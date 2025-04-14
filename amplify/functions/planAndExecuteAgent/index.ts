// YAML形式の文字列化、データバリデーション用のライブラリをインポート
import { stringify } from "yaml"
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// リソース定義のスキーマをインポート
import { Schema } from '../../data/resource';

// Amazon BedrockとLangChainの関連ライブラリをインポート
import { ChatBedrockConverse } from "@langchain/aws";
import { BaseMessage, AIMessage, ToolMessage, AIMessageChunk, HumanMessage, isAIMessageChunk, BaseMessageChunk } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { END, START, StateGraph, Annotation, CompiledStateGraph, StateDefinition } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { RetryPolicy } from "@langchain/langgraph"

// Amplify関連のユーティリティ関数とGraphQL mutationsをインポート
import { AmplifyClientWrapper, getLangChainMessageTextContent, stringifyLimitStringLength } from '../utils/amplifyUtils'
import { publishResponseStreamChunk, updateChatSession } from '../graphql/mutations'

// 計算機能とカスタムツールをインポート
import { Calculator } from "@langchain/community/tools/calculator";

import { queryGQLToolBuilder } from './toolBox'
import { isValidJSON } from "../../../src/utils/amplify-utils";

// リトライの最大回数を設定
const MAX_RETRIES = 3

// プランのステップを定義するスキーマ
// 各ステップには タイトル、役割(AI/人間)、説明、ツール呼び出し、結果 が含まれる
const PlanStepSchema = z.object({
    title: z.string(),
    role: z.enum(['ai', 'human']),
    description: z.string(),
    toolCalls: z.array(z.any()).optional(),
    result: z.string().optional()
});

// PlanStepSchemaから型を生成
type PlanStep = z.infer<typeof PlanStepSchema>;

// プランの実行状態を管理するための状態定義
// input: ユーザーからの入力
// plan: 実行予定のステップ
// pastSteps: 実行済みのステップ
// response: 応答内容
const PlanExecuteState = Annotation.Root({
    input: Annotation<string>({
        reducer: (x, y) => y ?? x ?? "",
    }),
    plan: Annotation<PlanStep[]>({
        reducer: (x, y) => y ?? x ?? [],
    }),
    pastSteps: Annotation<PlanStep[]>({
        reducer: (x, y) => y ?? x ?? [],
    }),
    response: Annotation<string>({
        reducer: (x, y) => y ?? x,
    })
})

// 2つの配列が等しいかどうかを判定するヘルパー関数
function areListsEqual<T>(list1: T[] | undefined, list2: T[] | undefined): boolean {
    if (!list1 || !list2) return false;
    return list1.length === list2.length &&
        list1.every((value, index) => value === list2[index]);
}

// AIの応答をストリーミング形式でクライアントに送信する関数
async function publishTokenStreamChunk(props: { tokenStreamChunk: AIMessageChunk, tokenIndex: number, amplifyClientWrapper: AmplifyClientWrapper }) {
    const streamChunk = props.tokenStreamChunk
    const chunkContent = getLangChainMessageTextContent(streamChunk)

    if (chunkContent) {
        // GraphQL mutationを使用して応答チャンクをクライアントにストリーミング
        await props.amplifyClientWrapper.amplifyClient.graphql({
            query: publishResponseStreamChunk,
            variables: {
                chatSessionId: props.amplifyClientWrapper.chatSessionId,
                index: props.tokenIndex,
                chunk: chunkContent
            }
        })
    }
}

// Lambda関数のメインハンドラー。プランの作成と実行を管理する
export const handler: Schema["invokePlanAndExecuteAgent"]["functionHandler"] = async (event) => {

    // デバッグ用のログ出力（現在はコメントアウト）
    // console.log('event: ', event)
    // console.log('context: ', context)
    // console.log('Amplify env: ', env)
    // console.log('process.env: ', process.env)


    // 必須パラメータのバリデーション
    if (!(event.arguments.chatSessionId)) throw new Error("Event does not contain chatSessionId");
    if (!event.identity) throw new Error("Event does not contain identity");
    if (!('sub' in event.identity)) throw new Error("Event does not contain user");

    // Amplifyクライアントのラッパーを初期化
    // チャットセッションの管理や通信に使用
    const amplifyClientWrapper = new AmplifyClientWrapper({
        chatSessionId: event.arguments.chatSessionId,
        env: process.env
    })

    // ユーザーに処理開始を通知するための初期メッセージを送信
    await publishTokenStreamChunk({
        tokenStreamChunk: new AIMessageChunk({content: "Generating new plan ...\n\n"}),//This is just meant to show something is happening.
        tokenIndex: -1,
        amplifyClientWrapper: amplifyClientWrapper
    })

    try {
        // 現在のチャットセッション情報を取得
        const chatSession = await amplifyClientWrapper.getChatSession({ chatSessionId: event.arguments.chatSessionId })
        if (!chatSession) throw new Error(`Chat session ${event.arguments.chatSessionId} not found`)

        // チャットの履歴メッセージを取得
        console.log('Getting messages for chat session: ', event.arguments.chatSessionId)
        await amplifyClientWrapper.getChatMessageHistory({
            latestHumanMessageText: event.arguments.lastMessageText
        })

        // チャットセッションを更新し、ユーザーの意図（目標）を保存
        await amplifyClientWrapper.amplifyClient.graphql({
            query: updateChatSession,
            variables: {
                input: {
                    id: event.arguments.chatSessionId,
                    planGoal: event.arguments.lastMessageText
                }
            }
        })

        // エージェントへの入力を準備
        // - input: ユーザーからの最新のメッセージ
        // - plan: 現在のプランのステップ（JSON文字列から変換）
        // - pastSteps: 既に実行済みのステップ（JSON文字列から変換）
        const inputs = {
            input: event.arguments.lastMessageText,
            plan: chatSession?.planSteps?.map(step => JSON.parse(step || "") as PlanStep),
            pastSteps: chatSession?.pastSteps?.map(step => JSON.parse(step || "") as PlanStep),
        }

        // Amazon Bedrockを使用したチャットモデルを初期化
        // temperature: 0 は、より決定論的な（予測可能な）応答を生成
        const agentModel = new ChatBedrockConverse({
            model: process.env.MODEL_ID,
            temperature: 0
        });

        ///////////////////////////////////////////////
        ///////// Executor Agent Step /////////////////
        ///////////////////////////////////////////////
        // エージェントが使用できるツールを定義
    // - Calculator: 数学的な計算を実行するためのツール
    // - queryGQLToolBuilder: GraphQLクエリを実行するためのカスタムツール
    const agentExecutorTools = [
            new Calculator,
            queryGQLToolBuilder({
                amplifyClientWrapper: amplifyClientWrapper,
                chatMessageOwnerIdentity: event.identity.sub
            })
        ]

        // ReActフレームワークを使用したエージェントを作成
        // ReActは「Reasoning(推論) + Acting(行動)」の略で、
        // エージェントが問題を解決するために以下のステップを繰り返します:
        // 1. 状況を分析(Reasoning)
        // 2. 適切なツールを選択して実行(Acting)
        // 3. 結果を評価して次のステップを決定
        const agentExecutor = createReactAgent({
            llm: agentModel,  // 言語モデル（AI）を指定
            tools: agentExecutorTools,  // エージェントが使用できるツールを指定
        });

        // const dummyAgentExecutorResponse = await agentExecutor.invoke({
        //     messages: [new HumanMessage("who is the winner of the us open")],
        //   });
        // console.log("Dummy Agent Executor Response:\n", dummyAgentExecutorResponse.slice(-1)[0])

        ///////////////////////////////////////////////
        ///////// Planning Step ///////////////////////
        ///////////////////////////////////////////////

        // プランの構造をJSON Schemaとして定義
        // zodToJsonSchemaは、zodのスキーマ定義をJSON Schemaフォーマットに変換する
        // このスキーマは以下の構造を持つ:
        // - steps: PlanStepSchemaの配列
        // - 各ステップには title, role, description などが含まれる
        // - 配列の順序は実行順序を表す
        const plan = zodToJsonSchema(
            z.object({
                steps: z
                    .array(PlanStepSchema)
                    .describe("Different steps to follow. Sort in order of completion"),
            }),
        );

        const planningModel = agentModel.withStructuredOutput(plan);

        ///////////////////////////////////////////////
        ///////// Re-Planning Step ////////////////////
        ///////////////////////////////////////////////


        // プランを生成・更新するためのプロンプトテンプレートを定義
        // このプロンプトは以下の要素を含む:
        // - 目標(objective)に基づいて段階的なプランを作成
        // - 各ステップは具体的なタスクで、正しく実行すれば目標達成につながる
        // - 不要なステップは含めない
        // - 最終ステップの結果が最終的な回答となる
        // - 各ステップに必要な情報をすべて含める
        // - 利用可能なツールで解決できる場合は、AIよりも人間の役割を優先する
        const replannerPrompt = ChatPromptTemplate.fromTemplate(
            `For the given objective, come up with a simple step by step plan. 
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            Favor assigning the role of ai to human if an available tool may be able to resolve the step.
            
            Your objective was this:
            {objective}
            
            Your original plan (if any) was this:
            {plan}
            
            You have currently done the follow steps:
            {pastSteps}
            
            Update your plan accordingly.  
            Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.`.replace(/^\s+/gm, ''),
        );

        // プロンプトとプランニングモデルを組み合わせて、
        // プランを生成・更新するためのパイプラインを作成
        const replanner = replannerPrompt.pipe(planningModel);

        ///////////////////////////////////////////////
        ///////// Response Step ///////////////////////
        ///////////////////////////////////////////////

        // ユーザーへの応答を生成するためのプロンプトテンプレートを定義
        // - 入力された目標、次のステップ、完了したステップに基づいて
        // - マークダウン形式で応答を生成する
        const responderPrompt = ChatPromptTemplate.fromTemplate(
            `Respond to the user in markdown format based on the origional objective and completed steps.
            
            Your objective was this:
            {input}

            The next steps (if any) are this:
            {plan}
            
            You have currently done the follow steps:
            {pastSteps}
            `.replace(/^\s+/gm, ''),
        );


        // 応答の形式を定義するスキーマを作成
        // - response: マークダウン形式の文字列
        // zodToJsonSchemaを使用してZodスキーマをJSON Schemaに変換
        const response = zodToJsonSchema(
            z.object({
                response: z.string().describe("Response to user in markdown format."),
            }),
        );

        // AIモデルに応答の構造を指定
        // withStructuredOutputを使用して、モデルの出力を
        // 上で定義したスキーマの形式に制限する
        const responderModel = agentModel.withStructuredOutput(response);

        // プロンプトとモデルを組み合わせて
        // 応答生成のパイプラインを作成
        const responder = responderPrompt.pipe(responderModel)



        ///////////////////////////////////////////////
        ///////// Create the Graph ////////////////////
        ///////////////////////////////////////////////
        // カスタムハンドラーの定義：AIモデルからのトークン生成とチャットモデルの開始を処理
        const customHandler = {
            // 新しいトークンが生成されるたびに呼び出される関数
            handleLLMNewToken: async (token: string, idx: { completion: number, prompt: number }, runId: any, parentRunId: any, tags: any, fields: any) => {
                // 空のトークンの場合はランダムな数のドットで置き換え、そうでない場合はトークンをそのまま使用
                const tokenStreamChunk = new AIMessageChunk({ content: token.length > 0 ? token : '.'.repeat(Math.ceil(Math.random() * 5)) })

                // コンソールに緑色のテキストでトークンを出力
                process.stdout.write(`\x1b[32m${getLangChainMessageTextContent(tokenStreamChunk)}\x1b[0m`);

                // トークンをクライアントにストリーミング配信
                await publishTokenStreamChunk({
                    tokenStreamChunk: tokenStreamChunk,
                    tokenIndex: -2,
                    amplifyClientWrapper: amplifyClientWrapper
                })
            },
            // チャットモデルが開始されるときに呼び出される関数
            handleChatModelStart: async (llm: any, inputMessages: any, runId: any) => {
                console.log("Chat model start:\n", stringifyLimitStringLength(inputMessages));
            },
        };

        // プランの各ステップを実行する関数
        async function executeStep(
            state: typeof PlanExecuteState.State,
            config?: RunnableConfig,
        ): Promise<Partial<typeof PlanExecuteState.State>> {
            // 現在のステップから結果フィールドを除外して残りのタスク情報を取得
            const { result, ...task } = state.plan[0];

            // エージェントへの入力メッセージを作成
            // - 完了済みのステップ情報
            // - 現在実行すべきタスクの詳細
            // - ツールの使用方法に関する指示
            const inputs = {
                messages: [new HumanMessage(`
                    The following steps have been completed
                    <previousSteps>
                    ${stringify(state.pastSteps)}
                    </previousSteps>
                    
                    Now execute this task.
                    <task>
                    ${stringify(task)}
                    </task>

                    To make plots or tables, use the queryGQL tool.
                    When creating a table, never use the HTML format.
                    
                    Tool messages can contain visualizations, query results and tables.
                    If the tool message says it contains information which completes the task, return a summary to the user.
                    
                    Once you have a result for this task, respond with that result.
                    `)],
            };

            // エージェントを実行してタスクの結果を取得
            const { messages } = await agentExecutor.invoke(inputs, config);
            const resultText = getLangChainMessageTextContent(messages.slice(-1)[0]) || ""
            console.log("Execute Step Complete. Result Text:\n", resultText)

            // 実行結果を含む新しい状態を返す
            // - pastSteps: 完了したステップのリストに現在のタスクと結果を追加
            // - plan: 残りのステップ（現在のステップを除外）
            return {
                pastSteps: [
                    ...(state.pastSteps || []),
                    {
                        ...task,
                        result: resultText,
                    },
                ],
                plan: state.plan.slice(1),
            };
        }

        /**
         * プランを再評価・更新するための関数
         * @param state 現在の実行状態（入力、プラン、過去のステップなどを含む）
         * @param config 実行設定
         * @returns 更新された状態の一部
         */
        async function replanStep(
            state: typeof PlanExecuteState.State,
            config: RunnableConfig,
        ): Promise<Partial<typeof PlanExecuteState.State>> {

            // プランのステップがなく、過去のステップが変更されている場合は空オブジェクトを返す
            // （= プランが完了したことを示す）
            if (state.plan && state.plan.length === 0 && !areListsEqual(inputs.pastSteps, state.pastSteps)) return {}

            // 現在の過去のステップとプランステップを取得
            let pastSteps = state.pastSteps
            let planSteps = state.plan

            // ユーザーからの入力を処理する条件をチェック:
            // - プランが存在し、ステップがある
            // - 最初のステップが人間の役割
            // - 過去のステップがないか、入力の過去のステップと現在の過去のステップが同じ
            if (
                state.plan &&
                state.plan.length > 0 &&
                state.plan[0].role === "human" &&
                (
                    !state.pastSteps || 
                    areListsEqual(inputs.pastSteps, state.pastSteps)
                )
            ) {
                // ユーザーの入力を過去のステップとして記録
                pastSteps = [
                    ...(state.pastSteps ?? []),
                    {
                        ...state.plan[0],
                        result: event.arguments.lastMessageText
                    }
                ]

                // 処理済みのステップをプランから削除
                planSteps = planSteps.slice(1)

                console.log(`User responded to a step with the role human. New Past Steps: \n${stringify(pastSteps)} \n New plan steps:\n${planSteps}`)
            }

            /**
             * リプランナーを実行して新しいプランを生成する関数
             * 最大試行回数まで実行を試みる
             */
            const invokeReplanner = async () => {
                let newPlan: { steps: PlanStep[] }
                // 最大試行回数まで実行を繰り返す
                for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
                    try {
                        // リプランナーを設定して実行
                        newPlan = await replanner
                            .withConfig({
                                callbacks: [customHandler],
                                tags: ["replanner"],
                            })
                            .invoke(
                                {
                                    objective: state.input,
                                    plan: stringify(planSteps),
                                    // 試行回数に応じてフォーマット指示を追加
                                    pastSteps: stringify(pastSteps) + "\nMake sure you respond in the correct format".repeat(attempt)
                                },
                                config
                            ) as { steps: PlanStep[] };

                        // 返されたプランの検証
                        if (!newPlan.steps) throw new Error("No steps returned from replanner")

                        // 文字列として返された場合はJSONとしてパース
                        if (typeof newPlan.steps === 'string' && isValidJSON(newPlan.steps)) {
                            console.log("Steps are a string and valid JSON. Converting them to an object")
                            newPlan.steps = JSON.parse(newPlan.steps) as PlanStep[]
                        }

                        // プランのステップが正しい形式かチェック
                        if (
                            !Array.isArray(newPlan.steps) ||
                            !newPlan.steps.every((step: unknown) => (PlanStepSchema.safeParse(step).success)
                            )
                        ) {
                            console.warn(`Provided steps are not in the correct format.\n\nSteps: ${stringify(newPlan.steps)}\n\n`)
                        } else return newPlan

                    } catch (error) {
                        // 最大試行回数に達した場合はエラーを投げる
                        if (attempt === MAX_RETRIES - 1) {
                            throw new Error(`Failed to get valid output after ${MAX_RETRIES} attempts: ${error}`);
                        }
                    }
                }
            }

            // }

            // const newPlanFromInvoke = await replanner
            //     .withConfig({
            //         callbacks: [customHandler],
            //         // callbacks: config.callbacks!,
            //         // runName: "replanner",
            //         tags: ["replanner"],
            //     })
            //     .invoke(
            //         {
            //             objective: state.input,
            //             plan: stringify(planSteps),
            //             pastSteps: stringify(pastSteps)
            //         },
            //         config
            //     );
            // リプランナーを実行して新しいプランを取得
            const newPlanFromInvoke = await invokeReplanner()
            // プランが返されなかった場合はエラーを投げる
            if (!newPlanFromInvoke) throw new Error("No new plan returned from replanner")

            // 新しいプランをログに出力
            console.log("New Plan:\n", stringify(newPlanFromInvoke))

            // コメントアウトされた検証ロジック
            // - プランにステップが含まれているか確認
            // - 文字列として返された場合はJSONとしてパース
            // - プランのステップが正しい形式かチェック

            // プランの各ステップから'result'プロパティを除外
            // これにより、実行前の新しいプランステップだけが残る
            planSteps = newPlanFromInvoke.steps.map((step: PlanStep) => {
                const { result, ...planPart } = step
                return planPart
            })

            // 必須パラメータの存在チェック
            if (!(event.arguments.chatSessionId)) throw new Error("Event does not contain chatSessionId");
            if (!event.identity) throw new Error("Event does not contain identity");
            if (!('sub' in event.identity)) throw new Error("Event does not contain user");

            // 新しいプランのステップが存在する場合
            if (planSteps[0]) {
                // チャット履歴に新しいステップの開始を示すメッセージを送信
                // マークダウンの見出し(##)を使用してステップのタイトルを表示
                amplifyClientWrapper.publishMessage({
                    chatSessionId: event.arguments.chatSessionId,
                    owner: event.identity.sub,
                    message: new AIMessage({ content: `## ${planSteps[0].title}` }),
                    responseComplete: true
                })
            }


            // 更新された状態を返す
            // - plan: 新しいプランステップ
            // - pastSteps: これまでに完了したステップ
            return {
                plan: planSteps,
                pastSteps: pastSteps
            }
        }

        /**
         * ユーザーへの最終的な応答を生成するステップ
         * @param state 現在の実行状態(入力、プラン、過去のステップを含む)
         * @param config 実行設定
         * @returns 生成された応答を含む状態の一部
         */
        async function respondStep(
            state: typeof PlanExecuteState.State,
            config: RunnableConfig,
        ): Promise<Partial<typeof PlanExecuteState.State>> {
            // responderを使って応答を生成
            // - カスタムハンドラーを設定してコールバックを処理
            // - "responder"タグを付与して実行を追跡
            const response = await responder
                .withConfig({
                    callbacks: [customHandler],
                    tags: ["responder"],
                })
                .invoke({
                    input: state.input,
                    plan: stringify(state.plan),
                    pastSteps: stringify(state.pastSteps)
                },
                    config
                );

            // 生成された応答を返す
            return { response: response.response };
        }

        /**
         * ワークフローを終了するべきかを判断する関数
         * @param state 現在の実行状態
         * @returns "true"(終了する)または"false"(継続する)
         */
        function shouldEnd(state: typeof PlanExecuteState.State) {
            // 以下の条件のいずれかに該当する場合は終了
            if (!state.plan) return "false"                    // プランが存在しない場合は継続
            if (state.plan.length === 0) return "true"        // プランのステップがすべて完了した場合は終了
            if (state.plan[0].role === "human") return "true" // 次のステップが人間の入力を待つ場合は終了
            return "false";                                    // それ以外の場合は継続
        }

        // ワークフローの状態遷移グラフを定義
        const workflow = new StateGraph(PlanExecuteState)
            // 各ノード(処理ステップ)を追加。失敗時は最大2回まで再試行
            .addNode("agent", executeStep, { retryPolicy: { maxAttempts: 2 } })   // エージェントの実行ステップ
            .addNode("replan", replanStep, { retryPolicy: { maxAttempts: 2 } })   // プランの再評価ステップ
            .addNode("respond", respondStep, { retryPolicy: { maxAttempts: 2 } }) // 応答生成ステップ
            // ノード間の遷移を定義
            .addEdge(START, "replan")                         // 開始→プラン再評価
            .addEdge("agent", "replan")                      // エージェント実行→プラン再評価
            .addConditionalEdges("replan", shouldEnd, {      // プラン再評価から条件分岐
                true: "respond",                             // 終了条件満たす→応答生成
                false: "agent",                              // 継続→エージェント実行
            })
            .addEdge("respond", END);                        // 応答生成→終了

        // ワークフローをLangChainの実行可能な形式にコンパイル
        const agent = workflow.compile();

        ///////////////////////////////////////////////
        ///////// Invoke the Graph ////////////////////
        ///////////////////////////////////////////////

        // const stream = await agent.stream(inputs, {
        //     recursionLimit: 50,
        //     streamMode: "messages"
        // });

        const agentEventStream = agent.streamEvents(
            inputs,
            {
                version: "v2",
            }
        );

        // https://js.langchain.com/v0.2/docs/how_to/chat_streaming/#stream-events
        // https://js.langchain.com/v0.2/docs/how_to/streaming/#using-stream-events
        // const stream = executorAgent.streamEvents(input, { version: "v2" });

        console.log('Listening for stream events')
        // for await (const streamEvent of stream) {
        let currentChunkIndex = 10000 // This is meant to help if multiple agents are streaming at the same time to the client.
        for await (const streamEvent of agentEventStream) {
            // console.log('event: ', streamEvent.event)

            switch (streamEvent.event) {
                case "on_chat_model_stream":
                    const streamChunkText = getLangChainMessageTextContent(streamEvent.data.chunk as AIMessageChunk) || ""

                    //Write the blurb in blue
                    process.stdout.write(`\x1b[34m${streamChunkText}\x1b[0m`);

                    await publishTokenStreamChunk({
                        tokenStreamChunk: streamEvent.data.chunk,
                        tokenIndex: currentChunkIndex++,
                        amplifyClientWrapper: amplifyClientWrapper,
                    })
                    break
                case "on_chain_stream":
                    console.log('on_chain_stream: \n', stringifyLimitStringLength(streamEvent))
                    const chainStreamMessage = streamEvent.data.chunk
                    const chainMessageType = ("planner" in chainStreamMessage || "replan" in chainStreamMessage) ? "plan" :
                        ("agent" in chainStreamMessage) ? "agent" :
                            ("respond" in chainStreamMessage) ? "respond" :
                                "unknown"

                    switch (chainMessageType) {
                        case "plan":
                            const updatePlanResonseInput: Schema["ChatSession"]["updateType"] = {
                                id: event.arguments.chatSessionId,
                                planSteps: ((chainStreamMessage.planner || chainStreamMessage.replan) as typeof PlanExecuteState.State)
                                    .plan.map((step) => JSON.stringify(step, null, 2)),
                            }

                            //If the chatStreamMessage contains pastSteps, update the chat session with them.
                            if (chainStreamMessage.replan.pastSteps) {
                                updatePlanResonseInput.pastSteps = (chainStreamMessage.replan as typeof PlanExecuteState.State)
                                    .pastSteps.map((step) => JSON.stringify(step, null, 2))
                            }

                            const updatePlanResonse = await amplifyClientWrapper.amplifyClient.graphql({
                                query: updateChatSession,
                                variables: {
                                    input: updatePlanResonseInput
                                }
                            })

                            // console.log(`Update Plan Response:\n`, stringify(updatePlanResonse))
                            break
                        case "agent":
                            const executeAgentChatSessionUpdate = await amplifyClientWrapper.amplifyClient.graphql({
                                query: updateChatSession,
                                variables: {
                                    input: {
                                        id: event.arguments.chatSessionId,
                                        pastSteps: (chainStreamMessage.agent as typeof PlanExecuteState.State).pastSteps.map((step) => JSON.stringify(step, null, 2)),
                                        planSteps: (chainStreamMessage.agent as typeof PlanExecuteState.State).plan.map((step) => JSON.stringify(step, null, 2)),
                                    }
                                }
                            })
                            break
                        case "respond":
                            // console.log('Response Event: ', chainStreamMessage)
                            const responseAIMessage = new AIMessage({
                                content: chainStreamMessage.respond.response,
                            })

                            // console.log('Publishing AI Message: ', responseAIMessage, '. Content: ', responseAIMessage.content)

                            await amplifyClientWrapper.publishMessage({
                                chatSessionId: event.arguments.chatSessionId,
                                owner: event.identity.sub,
                                message: responseAIMessage,
                                responseComplete: true
                            })
                            break
                        default:
                            console.log('Unknown message type:\n', stringifyLimitStringLength(chainStreamMessage))
                            break
                    }
                    break
                // case "on_tool_end":
                case "on_chat_model_end":
                    const streamChunk = streamEvent.data.output as ToolMessage | AIMessageChunk
                    const textContent = getLangChainMessageTextContent(streamChunk)
                    if (textContent && textContent.length > 20) {
                        await amplifyClientWrapper.publishMessage({
                            chatSessionId: event.arguments.chatSessionId,
                            owner: event.identity.sub,
                            message: streamChunk
                        })
                    }
                    break

            }
        }

        return "Invocation Successful!";

    } catch (error) {

        console.log('Error: ', error)

        if (error instanceof Error) {
            //If there is an error
            const AIErrorMessage = new AIMessage({ content: error.message + `\n model id: ${process.env.MODEL_ID}` })
            await amplifyClientWrapper.publishMessage({
                chatSessionId: event.arguments.chatSessionId,
                owner: event.identity.sub,
                message: AIErrorMessage,
                responseComplete: true
            })
            return error.message
        }
        return `Error: ${JSON.stringify(error)}`
    }

};