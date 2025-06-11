// import { BedrockAgent } from "@aws-sdk/client-bedrock-agent"
import outputs from '@/../amplify_outputs.json';

type BaseAgent = {
    name: string
    samplePrompts: string[]
    source: 'bedrockAgent' | 'graphql'
}

export type BedrockAgent = BaseAgent & {
    source: "bedrockAgent"
    agentId: string
    agentAliasId: string
}

export type LangGraphAgent = BaseAgent & {
    source: "graphql"
    invokeFieldName: string
}

export const defaultAgents: { [key: string]: BaseAgent | BedrockAgent | LangGraphAgent } = {
    PlanAndExecuteAgent: {
        name: `Production Agent`,
        source: `graphql`,
        samplePrompts: [
            `API番号30-045-29202の井戸が今朝、チュービングに穴が開いている兆候があり、ガスの生産が停止しました。
            坑井ファイルにある全ての運転イベントの表を作成してください。
            過去の月間生産量データを全て照会し、イベントデータと生産データの両方をプロットしてください。
            井戸の残存生産量の価値を見積もってください。 井戸を修理する手順を作成し、修理費用を見積もり、財務指標を計算してください。
            詳細な費用と手順データを含む井戸修理に関する経営報告書を作成してください。 
            `.replace(/^\s+/gm, ''),
            `坑井ファイルからAPI番号30-045-29202の坑井を検索し、作業の種類（掘削、仕上げ、改修、プラグ、その他）、作業詳細を記載した報告書からのテキスト、文書タイトルを含む表を作成してください。
            また、この坑井からの月間の石油、ガス、水の総生産量を取得するSQLクエリを実行してください。 
            イベントデータと生産データの両方を含むプロットを作成してください。`.replace(/^\s+/gm, ''), //This trims the white space at the start of each line
            `1900年以降のAPI番号30-045-29202の坑井における月間の石油、ガス、水の総生産量をプロットしてください。`
        ]
    },
    MaintenanceAgent: {
        name: "Maintenance Agent",
        source: "bedrockAgent",
        agentId: outputs.custom.maintenanceAgentId,
        agentAliasId: outputs.custom.maintenanceAgentAliasId,
        samplePrompts: [
            "社内データによると、バイオディーゼル製造装置には、いくつタンクがありますか？",
            "2024年9月、バイオディーゼル製造装置で発生した主なトラブルや事故、また、それに対して取られた対策をリストアップしてください。",
        ],
    } as BedrockAgent,
    RegulatoryAgent: {
        name: "Regulatory Agent",
        source: "bedrockAgent",
        agentId: outputs.custom.regulatoryAgentId,
        agentAliasId: outputs.custom.regulatoryAgentAliasId,
        samplePrompts: [
            "What are the requirements for fugitive emissions monitoring and reporting in the U.S.?",
            "What are the requirements for decomissioning an offshore oil well in Brazil?",
        ],
    } as BedrockAgent,
    PetrophysicsAgent: {
        name: "Petrophysics Agent",
        source: "bedrockAgent",
        agentId: outputs.custom.petrophysicsAgentId,
        agentAliasId: outputs.custom.petrophysicsAgentAliasId,
        samplePrompts: [
            "Give me a summary fluid substitution modeling",
            "Gassmann の式の入力変数を教えて",
            "What are AVO classes?",
            "Calculate the intercept and gradient value of the wet sandstone with vp=3.5 km/s, vs=1.95 km/s, bulk density=2.23 gm/cc when it is overlain by a shale? Determine the AVO class.",
            "A wet sandstone has vp=3.5 km/s, vs=1.95 km/s, bulk density=2.23 gm/cc. What are the expected seismic velocities of the sandstone if the desired ﬂuid saturation is 80% oil? Use standard assumptions."
            ],
    } as BedrockAgent,
    UpstreamAgent: {
        name: "石油上流工程 AIエージェント",
        source: "bedrockAgent",
        agentId: 'ZWXAIZT6GH',
        agentAliasId: 'IN0XDVNYP5',
        samplePrompts: [
            "Give me a summary fluid substitution modeling",
            "Gassmann の式の入力変数を教えて",
            ],
    } as BedrockAgent,
    SolarAgent: {
        name: "太陽光発電 AIエージェント",
        source: "bedrockAgent",
        agentId: '1CENEWD5QM',
        agentAliasId: 'C9RTQGISKU',
        samplePrompts: [
            "エネルギー消費予測を教えていただけますか？現在の使用量と比べてどうですか？私のIDは1です。",
            "8月の予測を更新していただけますか？旅行の予定があり、予測は50になります。私のIDは1です。",
        ],
    } as BedrockAgent,
    WhatIfSimAgent: {
        name: "供給計画 AIエージェント",
        source: "bedrockAgent",
        agentId: '1DX6TV4PSH',
        agentAliasId: 'LK9N8DCRMO',
        samplePrompts: [
            "製油所Xの稼働が停止したので、石油製品を輸入したいです、その場合の最適な輸入量を教えて",
            "油種振替をしたいです。最適な振替量を教えて",
            "夏の輸出入で在庫調整する際の熟練者からのアドバイスは？",
            "製油所Xの稼働が停止したので輸入か油種振替をして製油所の在庫を調整したいです。最適な施策はどれですか？",
        ],
    } as BedrockAgent

}