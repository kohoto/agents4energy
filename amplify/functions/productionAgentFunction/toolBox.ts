import { stringify } from 'yaml'
import { z } from "zod";

import { BedrockAgentRuntimeClient, RetrieveCommand } from "@aws-sdk/client-bedrock-agent-runtime";
import { S3Client, GetObjectCommand, ListObjectsV2Command, ListObjectsV2CommandInput } from "@aws-sdk/client-s3";

import { tool } from "@langchain/core/tools";
import { env } from '$amplify/env/production-agent-function';

import { AmplifyClientWrapper, FieldDefinition } from '../utils/amplifyUtils'
import { processWithConcurrency, startQueryExecution, waitForQueryToComplete, getQueryResults, transformResultSet } from '../utils/sdkUtils'

import { ToolMessageContentType } from '../../../src/utils/types'

import { invokeBedrockWithStructuredOutput } from '../graphql/queries'

import { getStructuredOutputResponse } from '../getStructuredOutputFromLangchain'
import { HumanMessage } from "@langchain/core/messages";

const s3Client = new S3Client();

export async function queryKnowledgeBase(props: { knowledgeBaseId: string, query: string }) {
    console.log("Invoking productionAgent/queryKnowledgeBase with query: ", props.query)

    const bedrockRuntimeClient = new BedrockAgentRuntimeClient();

    const command = new RetrieveCommand({
        knowledgeBaseId: props.knowledgeBaseId,
        retrievalQuery: { text: props.query },
        retrievalConfiguration: {
            vectorSearchConfiguration: {
                numberOfResults: 5 // Adjust based on your needs
            }
        }
    });

    try {
        const response = await bedrockRuntimeClient.send(command);
        return response.retrievalResults;
    } catch (error) {
        console.error('Error querying knowledge base:', error);
        throw error;
    }
}

///////////////////////////////////////////////////////////////
////// Retrieve Petroleum Enginnering Knowledge Tool //////////
///////////////////////////////////////////////////////////////
const retrievePetroleumEngineeringKnowledgeSchema = z.object({
    concepts: z.string().describe(`どの概念について知りたいですか？`),
});

//https://js.langchain.com/docs/integrations/retrievers/bedrock-knowledge-bases/
export const retrievePetroleumEngineeringKnowledgeTool = tool(
    async ({ concepts }) => {

        const contextResponse = await queryKnowledgeBase({
            knowledgeBaseId: env.PETROLEUM_ENG_KNOWLEDGE_BASE_ID, // petroleum KB への問合せ
            query: concepts
        })

        if (!contextResponse) throw new Error(`No relevant tables found. Query: ${concepts}`)
        // console.log("Pet Eng KB response:\n", JSON.stringify(contextResponse, null, 2))

        return {
            messageContentType: 'tool_json',
            context: contextResponse
        } as ToolMessageContentType
    },
    {
        name: "retrievePetroleumEngineeringKnowledge",
        description: "以下の3つに関する情報を取得できます。: 1. 石油ガス開発における業界用語、石油ガス開発における修理/メンテナンス用語、petrophysicsの専門用語、掘削の専門用語 2. パイプラインの漏洩事故の詳細、修理の詳細、修理のコスト 3. ピグ作業の詳細、費用",
        schema: retrievePetroleumEngineeringKnowledgeSchema,
    }
);

//////////////////////////////////////////
////// Get table definiton tool //////////
//////////////////////////////////////////
const getTableDefinitionsSchema = z.object({
    tableFeatures: z.string().describe(`
        ユーザーの質問のどの特徴を見て、クエリを実行するテーブルを選択すべきですか？ キーワードと、可能性のあるSQLクエリのカラム名を含めてください。`),
});

//https://js.langchain.com/docs/integrations/retrievers/bedrock-knowledge-bases/
export const getTableDefinitionsTool = tool(
    async ({ tableFeatures }) => {
        // console.log('Getting relevant tables for table features:\n', tableFeatures)

        const relevantTables = await queryKnowledgeBase({
            knowledgeBaseId: env.AWS_KNOWLEDGE_BASE_ID,
            query: tableFeatures
        }
        )

        if (!relevantTables) throw new Error("No relevant tables found")
        // console.log("Text2Sql KB response:\n", JSON.stringify(relevantTables, null, 2))


        const tableDefinitions = relevantTables.map((result) =>
        ({
            ...JSON.parse(result?.content?.text || ""),
            score: result?.score
        }))

        // console.log('Table Definitions:\n', tableDefinitions)

        return {
            messageContentType: 'tool_json',
            tableDefinitions: tableDefinitions
        } as ToolMessageContentType
    },
    {
        name: "getTableDefinitionsTool",
        description: "データベースのSQLクエリを実行する前に必ずこのツールを呼び出してください。SQLクエリに利用可能なデータベーステーブルの定義を取得できます。",
        schema: getTableDefinitionsSchema,
    }
);

///////////////////////////////////////////////////
///// Execute SQL Statement Tool //////////////////
///////////////////////////////////////////////////

const executeSQLQuerySchema = z.object({
    query: z.string().describe(`
    実行するTrino SQLクエリ。 FROMエレメントにデータソース、データベース、テーブル名を含めてください。
    （例：FROM <dataSourceName>.production.daily） 
    
    全てのカラム名の周りに""を使用してください。 
    varchar型のカラムで日付関数を使用する場合は、まずカラムを日付型にキャストしてください。 
    DATE_SUB関数は利用できません。タイムスタンプに間隔値を追加する場合は、常にDATE_ADD(unit, value, timestamp)関数を使用してください。DATE_SUBは決して使用しないでください。 
    <利用不可能なSQL関数> DATE_SUB ILIKE </利用不可能なSQL関数> 
    DATE_TRUNC関数の使用例：DATE_TRUNC('month', CAST("firstDayOfMonth" AS DATE)) 
    WHEREまたはGROUP BY句では、SELECT句で定義されたカラムエイリアスを使用しないでください。 クエリ処理中にSELECT句の前に評価されるため、SELECT句で定義されたカラムエイリアスはWHEREまたはGROUP BY句で参照できません。 
    返される結果の最初のカラムがX軸のカラムとして使用されます。クエリに日付が含まれている場合は、それを最初のカラムとして設定してください。

    以下は日次の石油、ガス、水の総生産量に関するSQLクエリの例です。
        <exampleSqlQuery>
        SELECT
            DATE_TRUNC('day', CAST("firstDayOfMonth" AS DATE)) AS day,
            SUM("oil(bbls)") AS total_oil_production,
            SUM("gas(mcf)") AS total_gas_production,
            SUM("water(bbls)") AS total_water_production
        FROM "AwsDataCatalog"."<database_name>"."crawler_monthly_production"
        WHERE "well api" = '<example_well_api>'
            AND CAST("firstDayOfMonth" AS DATE) >= CAST('1990-01-01' AS DATE)
        GROUP BY DATE_TRUNC('day', CAST("firstDayOfMonth" AS DATE))
        ORDER BY day
        </exampleSqlQuery>
        `.replace(/^\s+/gm, '')),
});

function doesFromLineContainOneDot(sqlQuery: string): boolean {
    // Split the query into lines
    const lines = sqlQuery.split('\n');

    // Find the line that starts with "FROM" (case-insensitive)
    const fromLine = lines.find(line => line.trim().toUpperCase().startsWith('FROM'));

    // If there's no FROM line, return false
    if (!fromLine) {
        return false;
    }

    // Extract the part after "FROM"
    const afterFrom = fromLine.trim().substring(4).trim();

    // Count the number of dots
    const dotCount = (afterFrom.match(/\./g) || []).length;

    // Return true if there's exactly one dot
    return dotCount === 1;
}

export const executeSQLQueryTool = tool(
    async ({ query }) => {
        // console.log('Executing SQL Query:\n', query, '\nUsing workgroup: ', env.ATHENA_WORKGROUP_NAME)
        try {

            // See if the string date_sub is in the query sting
            if (query.toLowerCase().includes("date_sub")) {
                return {
                    messageContentType: 'tool_json',
                    error: `
                    DATE_SUB はこの SQL query では使用できません。クエリを書き直してください。 
                    また、新しい時刻をデータ間に追加する必要がある場合は、DATE_ADD(unit, value, timestamp) を使用してください。例: DATE_ADD('year', -5, CURRENT_DATE)
                    `.replace(/^\s+/gm, '')
                } as ToolMessageContentType
            }

            //Check if the datasource is included in the query
            if (doesFromLineContainOneDot(query)) {
                return {
                    messageContentType: 'tool_json',
                    error: `
                    The FROM line in the SQL query does not the data source.
                    Include the dataSource, database, and tableName in the FROM element (ex: FROM <dataSource>.production.daily)
                    `.replace(/^\s+/gm, '')
                } as ToolMessageContentType
            }

            const queryExecutionId = await startQueryExecution({
                query: query,
                workgroup: env.ATHENA_WORKGROUP_NAME,
            });
            await waitForQueryToComplete(queryExecutionId, env.ATHENA_WORKGROUP_NAME);
            const results = await getQueryResults(queryExecutionId);
            // console.log('Athena Query Result:\n', results);

            if (!results.ResultSet?.Rows) throw new Error("No results returned from Athena")

            const queryResponseData = transformResultSet(results.ResultSet)

            return {
                messageContentType: 'tool_table_trend',
                queryResponseData: queryResponseData,
            } as ToolMessageContentType

        } catch (error) {
            console.error('Error executing sql query:', error);
            return {
                messageContentType: 'tool_json',
                error: error instanceof Error ? error.message : `Error:\n ${JSON.stringify(error)}`
            } as ToolMessageContentType
        }
    },
    {
        name: "executeSQLQuery",
        description: `
        このツールは、構造化データを取得する際に使用します。（構造化データの例: 石油・ガス・水の生産量の数値）
        このツールを呼び出す前に、必ず getTableDefinitionsTool を呼び出してください。
        このツールは Trino SQL クエリを実行でき、結果をテーブル形式で返します。
        `.replace(/^\s+/gm, ''),
        schema: executeSQLQuerySchema,
    }
);

///////////////////////////////////////////////////
////////// Plot Table Tool ////////////////////////
///////////////////////////////////////////////////

const plotTableFromToolResponseSchema = z.object({
    chartTitle: z.string().describe("The title of the plot."),
    includePreviousDataTable: z.boolean().optional().describe("If true, the last table in the plot will be the data table. If false, the last table in the plot will be the event data table. Default is true."),
    includePreviousEventTable: z.boolean().optional().describe("If true, the last table in the plot will be the event data table. If false, the last table in the plot will be the data table. Default is true.")
    // numberOfPreviousTablesToInclude: z.number().int().optional().describe("The number of previous tables to include in the plot. Use at least 2 to include produciton and event data tables."),
});


export const plotTableFromToolResponseTool = tool(
    async ({ chartTitle, includePreviousDataTable = true, includePreviousEventTable = true }) => {

        return {
            messageContentType: 'tool_plot',
            // columnNameFromQueryForXAxis: columnNameFromQueryForXAxis,
            chartTitle: chartTitle,
            // numberOfPreviousTablesToInclude: numberOfPreviousTablesToInclude,
            includePreviousDataTable: includePreviousDataTable,
            includePreviousEventTable: includePreviousEventTable
            // chartData: queryResponseData
        } as ToolMessageContentType

    },
    {
        name: "plotTableFromToolResponseToolBuilder",
        description: "一つ前のツールからのメッセージに含まれるテーブル形式のデータをプロットします。",
        schema: plotTableFromToolResponseSchema,
    }
);


//////////////////////////////////////////
/////// Get Well File Info Tool //////////
//////////////////////////////////////////

const getS3KeyConentsSchema = z.object({
    s3Key: z.string().describe("The S3 key to get the contents of.")
});

export const getS3KeyConentsTool = tool(
    async ({ s3Key }) => {
        const getObjectResponse = await s3Client.send(new GetObjectCommand({
            Bucket: process.env.DATA_BUCKET_NAME,
            Key: s3Key
        }))

        const objectContent = await getObjectResponse.Body?.transformToString()

        if (!objectContent) {
            return {
                messageContentType: 'tool_json',
                error: `
                The S3 Key was not found. S3 Key: ${s3Key}
                `
            } as ToolMessageContentType
        }

        // return objectContent
        return {
            messageContentType: 'tool_json',
            objectContent: objectContent
        } as ToolMessageContentType
    },
    {
        name: "getS3ObjectContents",
        description: `
        個々のS3キーの内容を返すことができます。
        このツールは、wellTableToolから行のソースファイルについて学ぶためにのみ使用してください。
        このツールを使用する前に、必ずwellTableToolを呼び出す必要があります。`.replace(/^\s+/gm, ''),
        schema: getS3KeyConentsSchema,
    }
);


//////////////////////////////////////////
//////// PDF Reports to Table Tool ///////
//////////////////////////////////////////

const jsonSchemaTypes = z.enum(['string', 'integer', 'date', 'number', 'boolean', 'null'])

export const wellTableSchema = z.object({
    dataToExclude: z.string().optional().describe("List of criteria to exclude data from the table"),
    dataToInclude: z.string().optional().describe("List of criteria to include data in the table"),
    tableColumns: z.array(z.object({
        columnName: z.string().describe('The name of a column'),
        columnDescription: z.string().describe('A description of the information which this column contains.'),
        columnDataDefinition: z.object({
            type: z.union([
                jsonSchemaTypes,
                z.array(jsonSchemaTypes)
            ]),
            format: z.string().describe('The format of the column.').optional(),
            enum: z.array(z.string()).optional(),
            pattern: z.string().describe('The regex pattern for the column.').optional(),
            minimum: z.number().optional(),
            maximum: z.number().optional(),
        })//.optional()
    })).describe(`テーブルの各列の列名と説明。チャートのラベルに最適な列を最初の要素として選択してください。tableColumns のリストは4追加に絞ってください。
        以下はJSON形式のテーブル列引数の例です。
        <exampleTableColumns>
        {
            "tableColumns": [
                {
                    "columnName": "event",    
                    "columnDescription": "The type of well event that occurred",
                    "columnDataDefinition": {
                        "type": "string",
                        "enum": [
                            "修理",
                            "規制",
                            "点検",
                            "その他"
                        ]
                    }
                },
                {
                    "columnName": "description",
                    "columnDescription": "A description of the well event",
                    "columnDataDefinition": {
                        "type": "string"
                    }
                }
            ]
        }
        </exampleTableColumns>
        `.replace(/^\s+/gm, '')),
    wellApiNumber: z.string().describe('この設備管理番号（API番号）のガスパイプラインについての情報を検索します。')
});

async function listFilesUnderPrefix(
    props: {
        bucketName: string,
        prefix: string,
        suffix?: string
    }
): Promise<string[]> {
    const { bucketName, prefix, suffix } = props
    // Create S3 client
    const files: string[] = [];

    // Prepare the initial command input
    const input: ListObjectsV2CommandInput = {
        Bucket: bucketName,
        Prefix: prefix,
    };

    try {
        let isTruncated = true;

        while (isTruncated) {
            const command = new ListObjectsV2Command(input);
            const response = await s3Client.send(command);

            // Add only the files that match the suffix to our array
            response.Contents?.forEach((item) => {
                if (item.Key && item.Key.endsWith(suffix || "")) {
                    files.push(item.Key);
                }
            });

            // Check if there are more files to fetch
            isTruncated = response.IsTruncated || false;

            // If there are more files, set the continuation token
            if (isTruncated && response.NextContinuationToken) {
                input.ContinuationToken = response.NextContinuationToken;
            }
        }

        return files;
    } catch (error) {
        console.error('Error listing files:', error);
        throw error;
    }
}

function removeSpaceAndLowerCase(str: string): string {
    //return a string that matches regex pattern '^[a-zA-Z0-9_-]{1,64}$'
    let transformed = str.replaceAll(" ", "").toLowerCase()
    transformed = transformed.replaceAll(/[^a-zA-Z0-9_-]/g, '');
    transformed = transformed.slice(0, 64);

    return transformed;
}

async function listS3Folders(
    props: {
        bucketName: string,
        prefix: string
    },
): Promise<string[]> {
    const { bucketName, prefix } = props

    const s3Client = new S3Client({});

    // Add trailing slash if not present
    const normalizedPrefix = prefix.endsWith('/') ? prefix : `${prefix}/`;

    const input: ListObjectsV2CommandInput = {
        Bucket: bucketName,
        Delimiter: '/',
        Prefix: normalizedPrefix,
    };

    try {
        const command = new ListObjectsV2Command(input);
        const response = await s3Client.send(command);

        // console.log('list folders s3 response:\n',response)

        // Get common prefixes (folders)
        const folders = response.CommonPrefixes?.map(prefix => prefix.Prefix!.slice(normalizedPrefix.length)) || [];

        // console.log('folders: ', folders)

        // Filter out the current prefix itself and just get the part of the prefix after the normalizedPrefix
        return folders
            .filter(folder => folder !== normalizedPrefix)

    } catch (error) {
        console.error('Error listing S3 folders:', error);
        throw error;
    }
}
// /(amplifyClientWrapper: AmplifyClientWrapper) => 
export const wellTableTool = tool(
    async ({ dataToInclude, tableColumns, wellApiNumber, dataToExclude }) => {
        console.log("Well Table Tool Invoked")
        try {
            if (!process.env.DATA_BUCKET_NAME) throw new Error("DATA_BUCKET_NAME environment variable is not set")

            //If tableColumns contains a column with columnName date, remove it. The user may ask for one, and one will automatically be added later.
            tableColumns = tableColumns.filter(column => !(column.columnName.toLowerCase().includes('date')))
            // Here add in the default table columns date and excludeRow 
            tableColumns.unshift({
                columnName: 'includeScore',
                columnDescription: `
                    もし、JSON object が [${dataToExclude}] に関する情報を含んでいたら、点数を 1 追加します。
                    もし [${dataToExclude}] に関する情報を含んでおらず、かつ、[${dataToInclude}] に関する情報を含んでいれば、点数を 10 追加します。
                    大抵は、点数は 5 点程度になります。10 点のものは、特別なケースとして保持します。
                    `,
                columnDataDefinition: {
                    type: 'integer',
                    minimum: 0,
                    maximum: 10
                }
            })

            tableColumns.unshift({
                columnName: 'includeScoreExplanation',
                columnDescription: `なぜこの点数をつけたのですか?`,
                columnDataDefinition: {
                    type: 'string',
                }
            })

            tableColumns.unshift({
                columnName: 'relevantPartOfJsonObject',
                columnDescription: `どの object がこの点数をつける原因となりましたか？`,
                columnDataDefinition: {
                    type: 'string',
                }
            })

            tableColumns.unshift({
                columnName: 'date',
                columnDescription: `イベントの日付のフォーマットは YYYY-MM-DD です。もし日付がわからない場合は null で構いません。`,
                columnDataDefinition: {
                    type: ['string', 'null'],
                    format: 'date',
                    pattern: "^(?:\\d{4})-(?:(0[1-9]|1[0-2]))-(?:(0[1-9]|[12]\\d|3[01]))$"
                }
            })

            // console.log('Input Table Columns: ', tableColumns)

            // const correctedColumnNameMap = tableColumns.map(column => [removeSpaceAndLowerCase(column.columnName), column.columnName])
            const correctedColumnNameMap = Object.fromEntries(
                tableColumns
                    .filter(column => column.columnName !== removeSpaceAndLowerCase(column.columnName))
                    .map(column => [removeSpaceAndLowerCase(column.columnName), column.columnName])
            );

            const fieldDefinitions: Record<string, FieldDefinition> = {};
            for (const column of tableColumns) {
                const correctedColumnName = removeSpaceAndLowerCase(column.columnName)

                fieldDefinitions[correctedColumnName] = {
                    ...(column.columnDataDefinition ? column.columnDataDefinition : { type: 'string' }),
                    description: column.columnDescription
                };
            }
            const jsonSchema = {
                title: "getKeyInformation",
                description: "フォームから抽出したテキストをもとに、これらの引数を入力してください。",
                type: "object",
                properties: fieldDefinitions,
                required: Object.keys(fieldDefinitions).filter(key => key !== 'date'),
            };

            console.log('target json schema for row:\n', stringify(jsonSchema))

            let columnNames = tableColumns.map(column => column.columnName)
            //Add in the source and relevanceScore columns
            columnNames.push('s3Key')

            console.log('Generating column names: ', columnNames)

            const s3Prefix = `production-agent/well-files/field=SanJuanEast/api=${wellApiNumber}/`;
            const wellFiles = await listFilesUnderPrefix({
                bucketName: process.env.DATA_BUCKET_NAME,
                prefix: s3Prefix,
                suffix: '.yaml'
            })
            // console.log('Well Files: ', wellFiles)

            if (wellFiles.length === 0) {
                const oneLevelUpS3Prefix = s3Prefix.split('/').slice(0, -2).join('/')

                console.log('one level up s3 prefix: ', oneLevelUpS3Prefix)
                const s3Folders = await listS3Folders({
                    bucketName: process.env.DATA_BUCKET_NAME,
                    prefix: oneLevelUpS3Prefix
                })//await onFetchObjects(oneLevelUpS3Prefix)
                // const s3Folders = s3ObjectsOneLevelHigher.filter(s3Asset => s3Asset.IsFolder).map(s3Asset => s3Asset.Key)

                return {
                    messageContentType: 'tool_json',
                    error: `
                    このAPI番号の坑井に関するファイルは見つかりませんでした: ${wellApiNumber}
                    情報を取得できる坑井のAPI番号は以下です:\n${s3Folders.join('\n')}
                `
                } as ToolMessageContentType
            }

            const dataRows = await processWithConcurrency({
                items: wellFiles,
                concurrency: parseInt(env.FILE_PROCESSING_CONCURRENCY || '30', 10),
                fn: async (s3Key) => {
                    try {

                        const getObjectResponse = await s3Client.send(new GetObjectCommand({
                            Bucket: process.env.DATA_BUCKET_NAME,
                            Key: s3Key
                        }))

                        const objectContent = await getObjectResponse.Body?.transformToString()
                        if (!objectContent) throw new Error(`No object content for s3 key: ${s3Key}`)
                        if (objectContent.length < 25) {
                            console.log("Object Length too small. Not generating a response. Object:\n", objectContent)
                            return
                        } // If the file contents are empty, do not create a row for that file. The empty file has a length of 22


                        // リトライ用の関数
                        async function retryWithBackoff<T>(
                        fn: () => Promise<T>,
                        maxRetries: number = 5,
                        initialDelayMs: number = 1000
                        ): Promise<T> {
                            let attempt = 0;
                            let delay = initialDelayMs;

                            while (true) {
                                try {
                                return await fn();
                                } catch (error: any) {
                                    console.log('retryWithBackoff')
                                // Bedrockのレートリミットエラーに該当するか判定
                                const isRateLimit =
                                    error?.name === 'ThrottlingException' ||
                                    error?.$metadata?.httpStatusCode === 429 ||
                                    error?.name === 'ModelErrorException' ||
                                    error?.$metadata?.httpStatusCode === 424 || 
                                    error?.name === 'ServiceUnavailableException' ||
                                    error?.$metadata?.httpStatusCode === 503;

                                if (!isRateLimit || attempt >= maxRetries) {
                                    throw error;
                                }

                                // バックオフしてリトライ
                                await new Promise((resolve) => setTimeout(resolve, delay));
                                attempt++;
                                delay *= 2; // 指数バックオフ
                                }
                            }
                        }

                        const messageText = `
                        ユーザーは、あなたに情報をYAML形式で提供することを要求しています。
                        YAML型式のobjectは、ガスパイプラインに関する情報を含んでいます。
                        <YamlObject>
                        ${objectContent}
                        </YamlObject>
                        `
                        // 既存のgetStructuredOutputResponse呼び出し部分を以下のように変更
                        const fileDataResponse = await retryWithBackoff(() =>
                            getStructuredOutputResponse({
                                messages: [new HumanMessage({ content: messageText })],
                                outputStructure: jsonSchema,
                                modelId: env.STRUCTURED_OUTPUT_MODEL_ID
                            })
                        );
                        //Replace the keys in file Data with those from correctedColumnNameMap
                        Object.keys(fileDataResponse).forEach(key => {
                            if (key in correctedColumnNameMap) {
                                const correctedKey = correctedColumnNameMap[key]
                                fileDataResponse[correctedKey] = fileDataResponse[key]
                                delete fileDataResponse[key]
                            }
                        })

                        //Preserve ordering of columns
                        const sortedFileDataResponse = Object.fromEntries(columnNames.map(colName => [colName, fileDataResponse[colName]]))

                        const fileResponseData: Record<string, any> = {
                            ...sortedFileDataResponse,
                            s3Key: s3Key
                        }

                        return fileResponseData
                    } catch (error) {
                        console.error('Error:', error);
                        throw new Error(`Error: ${JSON.stringify(error)}`)
                        // return {
                        //     messageContentType: 'tool_json',
                        //     error: `Error: ${error}`
                        // } as ToolMessageContentType
                    }
                }
            })


            // console.log('data Rows: ', dataRows)

            //Sort the data rows by date (first column)
            dataRows.sort((a, b) => {
                if (!a || !a.date) return 0
                if (!b || !b.date) return 1

                return a?.date.localeCompare(b?.date)
            });

            // console.log('data Rows: ', dataRows)

            return {
                messageContentType: 'tool_table_events',
                queryResponseData: dataRows
            } as ToolMessageContentType
        }
        catch (error) {
            console.error('Error in WellTableTool invocation:', error);
            return {
                messageContentType: 'tool_json',
                error: `Error: ${error}`
            } as ToolMessageContentType
        }
    },
    {
        name: "wellTableTool",
        description: `
        このツールは、ガスパイプラインに関する特定の情報を抽出するためにガスパイプラインのメンテナンスと点検に関するドキュメントを検索します。 
        ガスパイプラインのメンテナンスと点検に関するドキュメントから知識を取得するにはこのツールを使用してください。
        石油・ガス・水の生産量の数値を照会する際にはこのツールを絶対に使用してはいけません。このツールは構造化されたデータソースを照会することはできません。
        `.replace(/^\s+/gm, ''),
        schema: wellTableSchema,
    }
);

