pub mod models;
use anyhow::{Error as E, Result as R};
use candle::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
pub use models::WithModel;
use std::{
    cell::RefCell,
    panic::{self, AssertUnwindSafe},
};
use tokenizers::{PaddingParams, Tokenizer};

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
    /// Builds and returns a `BasedBertEmbedder` instance based on the configured settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `BasedBertEmbedder` instance, or an error if the build fails.
    pub fn build(self) -> R<BasedBertEmbedder> {
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
        let mut config: Config = serde_json::from_value(config_json)?;
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        Ok(BasedBertEmbedder {
            config,
            embed_model_dimensions: hidden_size,
            embed_model_id: self.embedding_model,
            embed_model_rev: model_revision,
            model: RefCell::new(None), // Fix: Wrap None in a RefCell
            normalize_embeddings: self.noramlize_embeddings,
            tokenizer_filename,
            tokenizer: RefCell::new(None), // Fix: Wrap None in a RefCell
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
    pub embed_model_id: WithModel,
    pub embed_model_rev: String,
    model: RefCell<Option<BertModel>>,
    normalize_embeddings: bool,
    pub embed_model_dimensions: usize,
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
    pub fn load(&self) -> R<()> {
        let device = self.init_device()?;
        let weights_filename = self.weights_filename.clone(); // Clone the weights_filename
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        *self.tokenizer.borrow_mut() =
            Some(Tokenizer::from_file(self.tokenizer_filename.clone()).map_err(E::msg)?);
        *self.model.borrow_mut() = Some(BertModel::load(vb, &self.config)?);

        Ok(())
    }

    /// Unloads the BERT model and tokenizer, freeing up memory.
    pub fn unload(&self) {
        *self.model.borrow_mut() = None;
        *self.tokenizer.borrow_mut() = None;
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
    pub fn embed_one(&self, text: &str) -> R<Vec<f32>> {
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
    pub fn embed_batch(&self, texts: Vec<&str>) -> R<Vec<Vec<f32>>> {
        if self.model.borrow().is_none() || self.tokenizer.borrow().is_none() {
            self.load()?;
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
        let mut candle_embed = builder.build()?;
        assert_eq!(candle_embed.embed_model_dimensions, 1024);
        assert!(candle_embed.normalize_embeddings);
        assert!(matches!(
            candle_embed.with_device,
            WithDevice::AnyCudaDevice
        ));
        candle_embed.unload();
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
        let mut candle_embed = builder.build()?;
        assert_eq!(candle_embed.embed_model_dimensions, 384);
        assert!(!candle_embed.normalize_embeddings);
        assert!(matches!(candle_embed.with_device, WithDevice::Cpu));
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn test_create_embeddings() -> R<()> {
        let builder = CandleEmbedBuilder::new();
        let mut candle_embed = builder.build()?;
        let text = "This is a test sentence.";
        let embeddings = candle_embed.embed_one(text)?;
        assert_eq!(embeddings.len(), candle_embed.embed_model_dimensions);
        candle_embed.unload();
        Ok(())
    }

    #[test]
    fn test_create_batch_embeddings() -> R<()> {
        let builder = CandleEmbedBuilder::new();
        let mut candle_embed = builder.build()?;
        let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
        let batch_embeddings = candle_embed.embed_batch(texts)?;
        assert_eq!(batch_embeddings.len(), 3);
        assert_eq!(
            batch_embeddings[0].len(),
            candle_embed.embed_model_dimensions
        );
        candle_embed.unload();
        Ok(())
    }
}
