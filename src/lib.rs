use anyhow::{Error as E, Result as R};
use candle::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::panic::{self, AssertUnwindSafe};
use tokenizers::{PaddingParams, Tokenizer};

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

pub enum WithDevice {
    AnyCudaDevice,
    SpecificCudaDevice(usize),
    Cpu,
}

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
/// `CandleEmbedBuilder` is a builder struct for configuring and creating a `BasedBertEmbedder` instance.
pub struct CandleEmbedBuilder {
    pub embedding_model: WithModel,
    pub model_revision: String,
    pub approximate_gelu: bool,
    pub noramlize_embeddings: bool,
    pub with_device: WithDevice,
}

impl Default for CandleEmbedBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl CandleEmbedBuilder {
    /// Creates a new instance of `CandleEmbedBuilder` with default configuration
    pub fn new() -> Self {
        Self {
            embedding_model: WithModel::Default,
            model_revision: "main".to_string(),
            approximate_gelu: true,
            noramlize_embeddings: true,
            with_device: WithDevice::AnyCudaDevice,
        }
    }
    /// Sets the embedding model from predefined presets.
    ///
    /// # Arguments
    ///
    /// * `model` - The preset embedding model to use.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn set_model_from_presets(mut self, model: WithModel) -> Self {
        self.embedding_model = model;
        self
    }
    /// Sets a custom embedding model.
    ///
    /// # Arguments
    ///
    /// * `embedding_model` - The ID or name of the custom embedding model.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn custom_embedding_model(mut self, embedding_model: &str) -> Self {
        self.embedding_model = WithModel::Custom(embedding_model.to_string());
        self
    }
    /// Sets a custom model revision.
    ///
    /// # Arguments
    ///
    /// * `model_revision` - The revision of the model to use.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn custom_model_revision(mut self, model_revision: &str) -> Self {
        self.model_revision = model_revision.to_string();
        self
    }
    /// Specifies whether to use approximate GeLU activation function.
    ///
    /// # Arguments
    ///
    /// * `approximate_gelu` - A boolean indicating whether to use approximate GeLU.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn approximate_gelu(mut self, approximate_gelu: bool) -> Self {
        self.approximate_gelu = approximate_gelu;
        self
    }
    /// Specifies whether to normalize the embeddings.
    ///
    /// # Arguments
    ///
    /// * `normalize_embeddings` - A boolean indicating whether to normalize the embeddings.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn normalize_embeddings(mut self, normalize_embeddings: bool) -> Self {
        self.noramlize_embeddings = normalize_embeddings;
        self
    }
    /// Specifies to use the CPU as the device for the model.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn with_device_cpu(mut self) -> Self {
        self.with_device = WithDevice::Cpu;
        self
    }
    /// Specifies to use any available CUDA device for the model.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn with_device_any_cuda(mut self) -> Self {
        self.with_device = WithDevice::AnyCudaDevice;
        self
    }
    /// Specifies to use a specific CUDA device for the model.
    ///
    /// # Arguments
    ///
    /// * `ordinal` - The ordinal number of the CUDA device to use.
    ///
    /// # Returns
    ///
    /// Returns the updated `CandleEmbedBuilder` instance.
    pub fn with_device_specific_cuda(mut self, ordinal: usize) -> Self {
        self.with_device = WithDevice::SpecificCudaDevice(ordinal);
        self
    }
    fn get_model_id_revision(&self) -> (String, String) {
        let model_id = match &self.embedding_model {
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
        };
        (model_id, self.model_revision.clone())
    }
    /// Builds and returns a `BasedBertEmbedder` instance based on the configured settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `BasedBertEmbedder` instance, or an error if the build fails.
    pub fn build(self) -> R<BasedBertEmbedder> {
        let (model_id, model_revision) = self.get_model_id_revision();
        let repo = Repo::with_revision(model_id, RepoType::Model, model_revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };
        let config_data = std::fs::read_to_string(config_filename)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_data)?;
        let hidden_size = config_json
            .get("hidden_size")
            .ok_or_else(|| anyhow::Error::msg("hidden_size field missing"))?
            .as_u64()
            .ok_or_else(|| anyhow::Error::msg("hidden_size is not a u64"))?
            as usize;
        let mut config: Config = serde_json::from_value(config_json)?;
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        Ok(BasedBertEmbedder::new(
            config,
            tokenizer_filename,
            weights_filename,
            self.noramlize_embeddings,
            hidden_size,
            self.with_device,
        ))
    }
}

/// `BasedBertEmbedder` is a struct for creating embeddings.
/// It provides functionality to load the model into memory, embed single or multiple texts,
/// and unload the model from memory.
/// It is initialized with the [CandleEmbedBuilder] struct.
pub struct BasedBertEmbedder {
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
    config: Config,
    tokenizer_filename: std::path::PathBuf,
    weights_filename: std::path::PathBuf,
    normalize_embeddings: bool,
    with_device: WithDevice,
    pub dimensions: usize,
}

impl BasedBertEmbedder {
    fn new(
        config: Config,
        tokenizer_filename: std::path::PathBuf,
        weights_filename: std::path::PathBuf,
        normalize_embeddings: bool,
        hidden_size: usize,
        with_device: WithDevice,
    ) -> Self {
        Self {
            model: None,
            tokenizer: None,
            config,
            tokenizer_filename,
            weights_filename,
            normalize_embeddings,
            dimensions: hidden_size,
            with_device,
        }
    }

    /// Loads the BERT model and tokenizer.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the loading is successful, or an error wrapped in a `Box<dyn std::error::Error>`
    /// if an error occurs during loading.
    pub fn load(&mut self) -> R<()> {
        let device = self.init_device()?;
        let weights_filename = self.weights_filename.clone(); // Clone the weights_filename
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        self.tokenizer =
            Some(Tokenizer::from_file(self.tokenizer_filename.clone()).map_err(E::msg)?);
        self.model = Some(BertModel::load(vb, &self.config)?);
        Ok(())
    }

    /// Unloads the BERT model and tokenizer, freeing up memory.
    pub fn unload(&mut self) {
        self.model = None;
        self.tokenizer = None;
    }

    fn init_device(&self) -> R<Device> {
        if let WithDevice::Cpu = self.with_device {
            return Ok(Device::Cpu);
        } else {
            if !candle::utils::cuda_is_available() {
                eprintln!("CUDA is not available, falling back to CPU");
                return Ok(Device::Cpu);
            };
            if let WithDevice::SpecificCudaDevice(i) = self.with_device {
                if let Ok(cuda_device) = Device::cuda_if_available(i) {
                    return Ok(cuda_device);
                } else {
                    eprintln!("CUDA device {i} is not available, trying other cuda devices");
                };
            };
            for i in 0..6 {
                let result = panic::catch_unwind(AssertUnwindSafe(|| Device::cuda_if_available(i)));

                match result {
                    Ok(Ok(device)) => match device {
                        Device::Cuda(_) => return Ok(device),
                        _ => {
                            eprintln!("CUDA is not available, checked device {i}.");
                        }
                    },
                    Ok(Err(_)) => {
                        eprintln!("Error occurred while checking CUDA device {i}, trying other CUDA devices");
                    }
                    Err(_) => {
                        eprintln!("Unexpected error (panic) occurred while checking CUDA device {i}, trying other CUDA devices");
                    }
                }
            }
        };
        eprintln!("No CUDA devices available, falling back to CPU");
        Ok(Device::Cpu)
    }

    /// Embeds a single text using the loaded BERT model.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of floats representing the embedding,
    /// or an error wrapped in a `Box<dyn std::error::Error>` if an error occurs.
    pub fn embed_one(&mut self, text: &str) -> R<Vec<f32>> {
        self.embed_batch(vec![text])
            .map(|v| v.into_iter().next().unwrap())
    }

    /// Embeds a batch of texts using the loaded BERT model.
    ///
    /// # Arguments
    ///
    /// * `texts` - A vector of texts to embed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of vectors of floats, where each inner vector
    /// represents the embedding for a single text in the batch. If an error occurs, an error
    /// wrapped in a `Box<dyn std::error::Error>` is returned.
    pub fn embed_batch(&mut self, texts: Vec<&str>) -> R<Vec<Vec<f32>>> {
        if self.model.is_none() || self.tokenizer.is_none() {
            self.load()?;
        }
        let model = if let Some(model) = &self.model {
            model
        } else {
            panic!("Model did not load")
        };
        let mut tokenizer = if let Some(tokenizer) = &self.tokenizer {
            tokenizer.clone()
        } else {
            panic!("Tokenizer did not load")
        };
        let device = &model.device;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer.encode_batch(texts, true).map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = model.forward(&token_ids, &token_type_ids)?;
        // avg pooling
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        // normalization
        let embeddings = if self.normalize_embeddings {
            Self::normalize_l2(&embeddings)?
        } else {
            embeddings
        };
        Ok(embeddings.to_vec2::<f32>()?)
    }

    fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_default_embed() -> R<()> {
        let builder = CandleEmbedBuilder::new();
        let mut cembed = builder.build()?;
        assert_eq!(cembed.dimensions, 1024);
        assert!(cembed.normalize_embeddings);
        assert!(matches!(cembed.with_device, WithDevice::AnyCudaDevice));
        cembed.unload();
        Ok(())
    }

    #[test]
    fn test_build_custom_embed() -> R<()> {
        let builder = CandleEmbedBuilder::new()
            .custom_embedding_model("avsolatorio/GIST-small-Embedding-v0")
            .custom_model_revision("d6c4190")
            .approximate_gelu(false)
            .normalize_embeddings(false)
            .with_device_cpu();
        let mut cembed = builder.build()?;
        assert_eq!(cembed.dimensions, 384);
        assert!(!cembed.normalize_embeddings);
        assert!(matches!(cembed.with_device, WithDevice::Cpu));
        cembed.unload();
        Ok(())
    }

    #[test]
    fn test_create_embeddings() -> R<()> {
        let builder = CandleEmbedBuilder::new();
        let mut cembed = builder.build()?;
        let text = "This is a test sentence.";
        let embeddings = cembed.embed_one(text)?;
        assert_eq!(embeddings.len(), cembed.dimensions);
        cembed.unload();
        Ok(())
    }

    #[test]
    fn test_create_batch_embeddings() -> R<()> {
        let builder = CandleEmbedBuilder::new();
        let mut cembed = builder.build()?;
        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_embeddings = cembed.embed_batch(texts)?;
        assert_eq!(batch_embeddings.len(), 3);
        assert_eq!(batch_embeddings[0].len(), cembed.dimensions);
        cembed.unload();
        Ok(())
    }
}
