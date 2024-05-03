pub mod models;
use anyhow::{Error, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
pub use models::WithModel;
use std::{
    cell::RefCell,
    panic::{self, AssertUnwindSafe},
};
use tokenizers::{Encoding, Tokenizer};

pub enum WithDevice {
    AnyCudaDevice,
    SpecificCudaDevice(usize),
    Cpu,
}

/// `CandleEmbedBuilder` is a builder struct for configuring and creating a `BasedBertEmbedder` instance.
pub struct CandleEmbedBuilder {
    pub approximate_gelu: bool,
    pub embedding_model: WithModel,
    pub mean_pooling: bool,
    pub model_revision: String,
    pub noramlize_embeddings: bool,
    pub truncate_text_len_overflow: bool,
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
            approximate_gelu: false,
            embedding_model: WithModel::Default,
            mean_pooling: true,
            model_revision: "main".to_string(),
            noramlize_embeddings: false,
            truncate_text_len_overflow: true,
            with_device: WithDevice::AnyCudaDevice,
        }
    }
    /// Sets the embedding model from predefined presets using the WithModel enum.
    ///
    /// # Arguments
    ///
    /// * `model` - The preset embedding model to use.
    ///
    pub fn set_model_from_presets(mut self, model: WithModel) -> Self {
        self.embedding_model = model;
        self
    }
    /// Sets a custom embedding model.
    ///
    /// # Arguments
    ///
    /// * `embedding_model` - The repo name and the model name to use. See `src/models.rs` for syntax
    ///
    pub fn custom_embedding_model(mut self, embedding_model: &str) -> Self {
        self.embedding_model = WithModel::Custom(embedding_model.to_string());
        self
    }
    /// Sets a custom model revision. Default is "main".
    ///
    pub fn custom_model_revision(mut self, model_revision: &str) -> Self {
        self.model_revision = model_revision.to_string();
        self
    }
    /// Specifies whether to use approximate GeLU activation function.
    ///
    pub fn approximate_gelu(mut self, approximate_gelu: bool) -> Self {
        self.approximate_gelu = approximate_gelu;
        self
    }
    /// Specifies whether to normalize the embeddings.
    ///
    pub fn normalize_embeddings(mut self, normalize_embeddings: bool) -> Self {
        self.noramlize_embeddings = normalize_embeddings;
        self
    }
    /// Specifies whether to apply mean pooling to the embeddings. Otherwise, only the CLS token is used.
    ///
    pub fn mean_pooling(mut self, mean_pooling: bool) -> Self {
        self.mean_pooling = mean_pooling;
        self
    }
    /// Specifies whether to truncate the text length if it exceeds the maximum input size. Defaults to true.
    ///
    pub fn truncate_text_len_overflow(mut self, truncate_text_len_overflow: bool) -> Self {
        self.truncate_text_len_overflow = truncate_text_len_overflow;
        self
    }
    /// Specifies to use the CPU as the device for the model.
    ///
    pub fn with_device_cpu(mut self) -> Self {
        self.with_device = WithDevice::Cpu;
        self
    }
    /// Specifies to use any available CUDA device for the model. It tries ordinals one through six and uses the first available.
    /// If CUDA is not available, it falls back to the CPU.
    ///
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
    pub fn with_device_specific_cuda(mut self, ordinal: usize) -> Self {
        self.with_device = WithDevice::SpecificCudaDevice(ordinal);
        self
    }
    /// Builds and returns a `BasedBertEmbedder` instance based on the configured settings.
    ///
    pub fn build(self) -> Result<BasedBertEmbedder> {
        let model_id = self.embedding_model.get_model_id_string();
        let model_revision = self.model_revision;
        let repo = Repo::with_revision(model_id, RepoType::Model, model_revision.clone());
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
        let max_position_embeddings = config_json
            .get("max_position_embeddings")
            .ok_or_else(|| anyhow::Error::msg("max_position_embeddings field missing"))?
            .as_u64()
            .ok_or_else(|| anyhow::Error::msg("max_position_embeddings is not a u64"))?
            as usize;
        let mut config: Config = serde_json::from_value(config_json)?;
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        Ok(BasedBertEmbedder {
            config,
            model_dimensions: hidden_size,
            model_id: self.embedding_model,
            model_max_input: max_position_embeddings,
            model_rev: model_revision,
            mean_pooling: self.mean_pooling,
            model: RefCell::new(None),
            normalize_embeddings: self.noramlize_embeddings,
            tokenizer_filename,
            tokenizer: RefCell::new(None),
            truncate_text_len_overflow: self.truncate_text_len_overflow,
            weights_filename,
            with_device: self.with_device,
        })
    }
}

/// `BasedBertEmbedder` is a struct for creating embeddings.
/// It provides functionality to load the model into memory, embed single or multiple texts,
/// and unload the model from memory.
/// It is initialized with the [CandleEmbedBuilder] struct.
pub struct BasedBertEmbedder {
    config: Config,
    mean_pooling: bool,
    model: RefCell<Option<BertModel>>,
    normalize_embeddings: bool,
    pub model_dimensions: usize,
    pub model_id: WithModel,
    pub model_max_input: usize,
    pub model_rev: String,
    tokenizer_filename: std::path::PathBuf,
    tokenizer: RefCell<Option<Tokenizer>>,
    truncate_text_len_overflow: bool,
    weights_filename: std::path::PathBuf,
    with_device: WithDevice,
}

impl BasedBertEmbedder {
    /// Loads the tokenizer.
    ///
    pub fn load_tokenizer(&self) -> Result<()> {
        *self.tokenizer.borrow_mut() =
            Some(Tokenizer::from_file(self.tokenizer_filename.clone()).map_err(Error::msg)?);

        Ok(())
    }

    /// Loads the BERT model.
    ///
    pub fn load_model(&self) -> Result<()> {
        let device = self.init_device()?;
        let weights_filename = self.weights_filename.clone(); // Clone the weights_filename
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        *self.model.borrow_mut() = Some(BertModel::load(vb, &self.config)?);
        Ok(())
    }

    /// Unloads the BERT model and tokenizer, freeing up memory. Call this when you are done using the model.
    ///
    pub fn unload(&self) {
        *self.model.borrow_mut() = None;
        *self.tokenizer.borrow_mut() = None;
    }

    /// Initializes the device for model loading.
    ///
    fn init_device(&self) -> Result<Device> {
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

    /// Embeds a batch of texts using the loaded BERT model.
    ///
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: embed_batch called with empty texts",
            ));
        }
        let mut embeddings = vec![];
        for text in texts {
            let embedding = self.embed_one(text)?;
            embeddings.push(embedding);
        }
        Ok(embeddings)
    }

    /// Embeds a single text using the loaded BERT model.
    ///
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: embed_one called with empty text",
            ));
        }

        if !self.truncate_text_len_overflow {
            let token_count = self.token_count(text)?;
            if token_count > self.model_max_input {
                return Err(Error::msg(format!(
                    "CandleEmbed error: Text input size of {} exceeds maximum input size of {}",
                    token_count, self.model_max_input
                )));
            }
        }

        if self.model.borrow().is_none() {
            self.load_model()?;
        }

        let model = self.model.borrow();
        let model = if let Some(model) = model.as_ref() {
            model
        } else {
            panic!("Model did not load")
        };

        let device = &model.device;
        let encoding = self.encode_text(text, true)?;
        let tokens = encoding.get_ids().to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let outputs = model.forward(&token_ids, &token_type_ids)?;

        let embedding = if self.mean_pooling {
            // Mean pooling
            let (_n_sentence, n_tokens, _hidden_size) = outputs.dims3()?;
            (outputs.sum(1)? / (n_tokens as f64))?
        } else {
            // CLS only
            outputs.i((.., 0))?
        };

        // normalization
        let embedding = if self.normalize_embeddings {
            embedding.broadcast_div(&embedding.sqr()?.sum_keepdim(1)?.sqrt()?)?
        } else {
            embedding
        };

        Ok(embedding.i(0)?.to_vec1::<f32>()?)
    }

    /// Counts the number of tokens in a single text using the loaded tokenizer.
    ///
    pub fn token_count(&self, text: &str) -> Result<usize> {
        let encoding = self.encode_text(text, false)?;
        Ok(encoding.get_tokens().len())
    }
    /// Counts the number of tokens in a batch of texts using the loaded tokenizer.
    ///
    pub fn token_count_batch(&self, texts: &[&str]) -> Result<Vec<usize>> {
        let encodings = self.encode_texts(texts, false)?;
        Ok(encodings
            .iter()
            .map(|encoding| encoding.get_tokens().len())
            .collect())
    }
    /// Tokenizes a single text using the loaded tokenizer.
    ///
    pub fn tokenize_one(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: tokenize_one called with empty text",
            ));
        }
        let encoding = self.encode_text(text, false)?;
        let token_string = encoding.get_tokens().to_vec();
        Ok(token_string)
    }
    /// Tokenizes a batch of texts using the loaded tokenizer.
    ///
    pub fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        if texts.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: tokenize_batch called with empty texts",
            ));
        }
        let encodings = self.encode_texts(texts, false)?;
        let token_strings = encodings
            .iter()
            .map(|encoding| encoding.get_tokens().to_vec())
            .collect::<Vec<_>>();
        Ok(token_strings)
    }
    /// Encodes a batch of texts using the loaded tokenizer. Used for the tokenizer functions and for pre-checking the input size. It is not used to generate embeddings.
    ///
    fn encode_text(&self, text: &str, with_trunc_settings: bool) -> Result<Encoding> {
        if text.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: encode_text called with an empty input text",
            ));
        }
        if self.tokenizer.borrow().is_none() {
            self.load_tokenizer()?;
        }
        let mut tokenizer = self.tokenizer.borrow_mut();
        let tokenizer = if let Some(tokenizer) = tokenizer.as_mut() {
            tokenizer
        } else {
            panic!("Tokenizer did not load")
        };
        let tokenizer = if with_trunc_settings && self.truncate_text_len_overflow {
            tokenizer
                .with_padding(None)
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: self.model_max_input,
                    ..Default::default()
                }))
                .map_err(anyhow::Error::msg)?
                .clone()
        } else {
            tokenizer
                .with_padding(None)
                .with_truncation(None)
                .map_err(anyhow::Error::msg)?
                .clone()
        };
        let encoding: tokenizers::Encoding = tokenizer.encode(text, true).map_err(Error::msg)?;
        Ok(encoding)
    }
    /// Encodes a batch of texts using the loaded tokenizer. Used for the tokenizer functions and for pre-checking the input size. It is not used to generate embeddings.
    ///
    fn encode_texts(&self, texts: &[&str], with_trunc_settings: bool) -> Result<Vec<Encoding>> {
        if texts.iter().any(|text| text.is_empty()) {
            return Err(Error::msg(
                "CandleEmbed error: encode_texts called with an empty input text",
            ));
        }
        if self.tokenizer.borrow().is_none() {
            self.load_tokenizer()?;
        }
        let mut tokenizer = self.tokenizer.borrow_mut();
        let tokenizer = if let Some(tokenizer) = tokenizer.as_mut() {
            tokenizer
        } else {
            panic!("Tokenizer did not load")
        };
        let tokenizer = if with_trunc_settings && self.truncate_text_len_overflow {
            tokenizer
                .with_padding(None)
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: self.model_max_input,
                    ..Default::default()
                }))
                .map_err(anyhow::Error::msg)?
                .clone()
        } else {
            tokenizer
                .with_padding(None)
                .with_truncation(None)
                .map_err(anyhow::Error::msg)?
                .clone()
        };
        let encodings: Vec<tokenizers::Encoding> = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(Error::msg)?;
        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default_embed() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().build()?;
        assert_eq!(candle_embed.model_dimensions, 1024);
        assert_eq!(candle_embed.model_max_input, 512);
        assert!(!candle_embed.normalize_embeddings);
        assert!(candle_embed.mean_pooling);
        assert!(candle_embed.truncate_text_len_overflow);
        assert!(matches!(
            candle_embed.with_device,
            WithDevice::AnyCudaDevice
        ));
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn build_custom_embed() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new()
            .custom_embedding_model("avsolatorio/GIST-small-Embedding-v0")
            .custom_model_revision("d6c4190")
            .approximate_gelu(false)
            .normalize_embeddings(false)
            .with_device_cpu()
            .build()?;
        assert_eq!(candle_embed.model_dimensions, 384);
        assert!(!candle_embed.normalize_embeddings);
        assert!(matches!(candle_embed.with_device, WithDevice::Cpu));
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn embed_one() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;
        let text = "This is a test sentence ok dog?";
        let embeddings = candle_embed.embed_one(text)?;
        assert_eq!(embeddings.len(), candle_embed.model_dimensions);
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn batch_embeddings() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;
        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_embeddings = candle_embed.embed_batch(&texts)?;
        assert_eq!(batch_embeddings.len(), 3);
        assert_eq!(batch_embeddings[0].len(), candle_embed.model_dimensions);
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn exceeds_max_input_size() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new()
            .truncate_text_len_overflow(false)
            .build()?;

        // Should pass
        let overly_long_string = (0..candle_embed.model_max_input - 2)
            .map(|_| "a")
            .collect::<Vec<_>>()
            .join(" ");
        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            &overly_long_string,
        ];

        let batch_embeddings = candle_embed.embed_batch(&texts)?;
        assert_eq!(batch_embeddings.len(), 3);
        assert_eq!(batch_embeddings[0].len(), candle_embed.model_dimensions);

        // Should fail
        let overly_long_string = (0..candle_embed.model_max_input)
            .map(|_| "a")
            .collect::<Vec<_>>()
            .join(" ");
        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            &overly_long_string,
        ];

        match candle_embed.embed_batch(&texts) {
            Err(e) => {
                assert_eq!(
                    e.to_string(),
                    format!("CandleEmbed error: Text input size of 514 exceeds maximum input size of {}",candle_embed.model_max_input)
                );
            }
            Ok(_) => panic!("Expected error"),
        }

        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn tokenize_text() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;

        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_tokens = candle_embed.tokenize_batch(&texts)?;
        assert_eq!(batch_tokens.len(), 3);

        let text = "This is the first sentence.";
        let tokens = candle_embed.tokenize_one(text)?;
        assert!(!tokens.is_empty());
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn token_count() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;

        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_tokens = candle_embed.token_count_batch(&texts)?;
        assert_eq!(batch_tokens.len(), 3);

        let text = "This is the first sentence.";
        let tokens = candle_embed.token_count(text)?;
        assert!(tokens > 0);
        candle_embed.unload();
        Ok(())
    }
}
