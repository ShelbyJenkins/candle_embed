// Easy to use model IDs
pub const DEFAULT_MODEL: &str = "WhereIsAI/UAE-Large-V1";
// Giant
pub const E5_MISTRAL_7B_INSTRUCT: &str = "intfloat/e5-mistral-7b-instruct";
pub const SFR_EMBEDDING_MISTRAL: &str = "Salesforce/SFR-Embedding-Mistral";
// Large
pub const SNOWFLAKE_ARCTIC_EMBED_L: &str = "Snowflake/snowflake-arctic-embed-l";
pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";
pub const MXBAI_EMBED_LARGE_V1: &str = "mixedbread-ai/mxbai-embed-large-v1";
// Medium
pub const SNOWFLAKE_ARCTIC_EMBED_M: &str = "Snowflake/snowflake-arctic-embed-m";
pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
// Small
pub const ALL_MINILM_L6_V2: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// `WithModel` is an enum that represents different preset embedding models or a custom model.
pub enum WithModel {
    E5Mistral7bInstruct,
    SfrEmbeddingMistral,
    SnowflakeArcticEmbedL,
    UaeLargeV1,
    MxbaiEmbedLargeV1,
    SnowflakeArcticEmbedM,
    BgeBaseEnV15,
    AllMinilmL6V2,
    Default,
    Custom(String),
}

impl WithModel {
    // Function to create an enum variant from a given model ID string
    pub fn get_model_id_enum(model_id: &str) -> Self {
        match model_id {
            E5_MISTRAL_7B_INSTRUCT => WithModel::E5Mistral7bInstruct,
            SFR_EMBEDDING_MISTRAL => WithModel::SfrEmbeddingMistral,
            SNOWFLAKE_ARCTIC_EMBED_L => WithModel::SnowflakeArcticEmbedL,
            UAE_LARGE_V1 => WithModel::UaeLargeV1,
            MXBAI_EMBED_LARGE_V1 => WithModel::MxbaiEmbedLargeV1,
            SNOWFLAKE_ARCTIC_EMBED_M => WithModel::SnowflakeArcticEmbedM,
            BGE_BASE_EN_V1_5 => WithModel::BgeBaseEnV15,
            ALL_MINILM_L6_V2 => WithModel::AllMinilmL6V2,
            _ => WithModel::Custom(model_id.to_string()),
        }
    }
    pub fn get_model_id_string(&self) -> String {
        match self {
            WithModel::E5Mistral7bInstruct => E5_MISTRAL_7B_INSTRUCT.to_string(),
            WithModel::SfrEmbeddingMistral => SFR_EMBEDDING_MISTRAL.to_string(),
            WithModel::SnowflakeArcticEmbedL => SNOWFLAKE_ARCTIC_EMBED_L.to_string(),
            WithModel::UaeLargeV1 => UAE_LARGE_V1.to_string(),
            WithModel::MxbaiEmbedLargeV1 => MXBAI_EMBED_LARGE_V1.to_string(),
            WithModel::SnowflakeArcticEmbedM => SNOWFLAKE_ARCTIC_EMBED_M.to_string(),
            WithModel::BgeBaseEnV15 => BGE_BASE_EN_V1_5.to_string(),
            WithModel::AllMinilmL6V2 => ALL_MINILM_L6_V2.to_string(),
            WithModel::Default => DEFAULT_MODEL.to_string(),
            WithModel::Custom(model_id) => model_id.to_string(),
        }
    }
}
