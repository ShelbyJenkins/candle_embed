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
    pub embedding_model: WithModel,
    pub model_revision: String,
    pub approximate_gelu: bool,
    pub noramlize_embeddings: bool,
    pub with_device: WithDevice,
    pub mean_pooling: bool,
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
            approximate_gelu: false,
            noramlize_embeddings: true,
            mean_pooling: true,
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
    pub fn mean_pooling(mut self, mean_pooling: bool) -> Self {
        self.mean_pooling = mean_pooling;
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
    /// Builds and returns a `BasedBertEmbedder` instance based on the configured settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `BasedBertEmbedder` instance, or an error if the build fails.
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
            embed_model_dimensions: hidden_size,
            embed_model_max_input: max_position_embeddings,
            embed_model_id: self.embedding_model,
            embed_model_rev: model_revision,
            model: RefCell::new(None), // Fix: Wrap None in a RefCell
            normalize_embeddings: self.noramlize_embeddings,
            tokenizer_filename,
            tokenizer: RefCell::new(None), // Fix: Wrap None in a RefCell
            weights_filename,
            with_device: self.with_device,
            mean_pooling: self.mean_pooling,
        })
    }
}

/// `BasedBertEmbedder` is a struct for creating embeddings.
/// It provides functionality to load the model into memory, embed single or multiple texts,
/// and unload the model from memory.
/// It is initialized with the [CandleEmbedBuilder] struct.
pub struct BasedBertEmbedder {
    config: Config,
    pub embed_model_id: WithModel,
    pub embed_model_rev: String,
    model: RefCell<Option<BertModel>>,
    normalize_embeddings: bool,
    mean_pooling: bool,
    pub embed_model_dimensions: usize,
    pub embed_model_max_input: usize,
    tokenizer_filename: std::path::PathBuf,
    tokenizer: RefCell<Option<Tokenizer>>,
    weights_filename: std::path::PathBuf,
    with_device: WithDevice,
}

impl BasedBertEmbedder {
    /// Loads the BERT model and tokenizer.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the loading is successful, or an error wrapped in a `Box<dyn std::error::Error>`
    /// if an error occurs during loading.
    pub fn load_tokenizer(&self) -> Result<()> {
        *self.tokenizer.borrow_mut() =
            Some(Tokenizer::from_file(self.tokenizer_filename.clone()).map_err(Error::msg)?);

        Ok(())
    }

    /// Loads the BERT model.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the loading is successful, or an error wrapped in a `Box<dyn std::error::Error>`
    /// if an error occurs during loading.
    pub fn load_model(&self) -> Result<()> {
        let device = self.init_device()?;
        let weights_filename = self.weights_filename.clone(); // Clone the weights_filename
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        *self.model.borrow_mut() = Some(BertModel::load(vb, &self.config)?);
        Ok(())
    }

    /// Unloads the BERT model and tokenizer, freeing up memory.
    pub fn unload(&self) {
        *self.model.borrow_mut() = None;
        *self.tokenizer.borrow_mut() = None;
    }

    /// Initializes the device for model loading.
    ///
    /// # Returns
    ///
    /// Returns the initialized device.
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
    /// # Arguments
    ///
    /// * `texts` - A vector of texts to embed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of vectors of floats, where each inner vector
    /// represents the embedding for a single text in the batch. If an error occurs, an error
    /// wrapped in a `Box<dyn std::error::Error>` is returned.
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
    /// # Arguments
    ///
    /// * `text` - The text to embed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of floats representing the embedding,
    /// or an error wrapped in a `Box<dyn std::error::Error>` if an error occurs.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: embed_one called with empty text",
            ));
        }

        let encoding = self.encode_texts(&[text])?;

        if encoding[0].len() > self.embed_model_max_input {
            return Err(Error::msg(format!(
                "CandleEmbed error: Text input size of {} exceeds maximum input size of {}",
                encoding[0].len(),
                self.embed_model_max_input
            )));
        }

        if self.tokenizer.borrow().is_none() {
            self.load_tokenizer()?;
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

        let mut tokenizer = self.tokenizer.borrow_mut();
        let tokenizer = if let Some(tokenizer) = tokenizer.as_mut() {
            tokenizer
        } else {
            panic!("Tokenizer did not load")
        };

        let device = &model.device;

        let tokens = tokenizer
            .encode(text, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
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

    pub fn token_count(&self, text: &str) -> Result<usize> {
        let encoding = self.encode_texts(&[text])?;
        Ok(encoding[0].len())
    }
    /// Tokenizes a single text using the loaded tokenizer.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of strings representing the tokens,
    /// or an error wrapped in a `Box<dyn std::error::Error>` if an error occurs.
    pub fn tokenize_one(&self, text: &str) -> Result<Vec<String>> {
        self.tokenize_batch(&[text])
            .map(|v| v.into_iter().next().unwrap())
    }

    /// Tokenizes a batch of texts using the loaded tokenizer.
    ///
    /// # Arguments
    ///
    /// * `texts` - An array of texts to tokenize.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of vectors of strings, where each inner vector

    pub fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        if texts.is_empty() {
            return Err(Error::msg(
                "CandleEmbed error: tokenize_batch called with empty texts",
            ));
        }
        let encodings = self.encode_texts(texts)?;
        let token_strings = encodings
            .iter()
            .map(|encoding| encoding.get_tokens().to_vec())
            .collect::<Vec<_>>();
        Ok(token_strings)
    }

    /// Encodes a batch of texts using the loaded tokenizer. Use for the tokenizer functions and for pre-checking the input size. It is not used to generate embeddings.
    ///
    /// # Arguments
    ///
    /// * `texts` - An array of texts to encode.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of `Encoding` objects, where each `Encoding`
    /// represents the encoding for a single text in the batch.
    fn encode_texts(&self, texts: &[&str]) -> Result<Vec<Encoding>> {
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
        tokenizer.with_padding(None);
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
        assert_eq!(candle_embed.embed_model_dimensions, 1024);
        assert_eq!(candle_embed.embed_model_max_input, 512);
        assert!(candle_embed.normalize_embeddings);
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
        assert_eq!(candle_embed.embed_model_dimensions, 384);
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
        assert_eq!(embeddings.len(), candle_embed.embed_model_dimensions);
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
        assert_eq!(
            batch_embeddings[0].len(),
            candle_embed.embed_model_dimensions
        );
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn batch_embeddings_where_text_at_max_input_size() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;

        let overly_long_string = (0..candle_embed.embed_model_max_input - 2)
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
        assert_eq!(
            batch_embeddings[0].len(),
            candle_embed.embed_model_dimensions
        );
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn batch_embeddings_where_text_exceeds_max_input_size() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;

        let overly_long_string = (0..candle_embed.embed_model_max_input + 1)
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
                    format!("CandleEmbed error: Text input size of 515 exceeds maximum input size of {}",candle_embed.embed_model_max_input)
                );
            }
            Ok(_) => panic!("Expected error"),
        }

        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn batch_tokens() -> Result<()> {
        let candle_embed = CandleEmbedBuilder::new().with_device_cpu().build()?;

        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_tokens = candle_embed.tokenize_batch(&texts)?;
        assert_eq!(batch_tokens.len(), 3);

        candle_embed.unload();
        Ok(())
    }
}
