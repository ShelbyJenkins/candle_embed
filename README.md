# CandleEmbed

CandleEmbed is a Rust library for creating embeddings using BERT-based models. It provides a convenient way to load pre-trained models, embed single or multiple texts, and customize the embedding process. It's basically the same code as the [candle example for embeddings]('https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert'), but with a nice wrapper. This exists because I wanted to play with Candle, and [fastembed.rs]('https://github.com/Anush008/fastembed-rs') doesn't support custom models.fs

Features

- Enums for most popular embedding models

- Specify custom model from HF

- Support for CUDA devices (requires feature flag)

Installation

Add the following to your Cargo.toml file:

```toml
[dependencies]
candle_embed = "0.1.0"
```

If you want to use CUDA devices, enable the cuda feature flag:

```toml
[dependencies]
candle_embed = { version = "0.1.0", features = ["cuda"] }
```

Usage

```rust

use candle_embed::{CandleEmbedBuilder, WithModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a builder with default settings
    let builder = CandleEmbedBuilder::new();
    
    // Customize the builder (optional)
    let builder = builder
        .set_model_from_presets(WithModel::UaeLargeV1)
        .normalize_embeddings(true)
        .approximate_gelu(true);

    // Set model from preset
    builder
        .set_model_from_presets(WithModel::UaeLargeV1);

    // Or use a custom model and revision
    builder
        .custom_embedding_model("avsolatorio/GIST-small-Embedding-v0")
        .custom_model_revision("d6c4190");

    // Will use the first available CUDA device (Default)
    builder.with_device_any_cuda();

    // Use a specific CUDA device failing
    builder.with_device_specific_cuda(ordinal: usize);

    // Use CPU (CUDA options fail over to this)
    builder.with_device_cpu();

    // Build the embedder
    let mut cembed = builder.build()?;
    
    // This loads the model and tokenizer into memory
    // Upon first running 'embed' this function is called
    // Documenting here for clarity
    cembed.load();

    // Embed a single text
    let text = "This is a test sentence.";
    let embeddings = cembed.embed_one(text)?;
    
    // Embed a batch of texts
    let texts = vec![
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
    ];
    let batch_embeddings = cembed.embed_batch(texts)?;
    
    // Unload the model dropping it from memory
    cembed.unload();
    
    Ok(())
}
```

Feature Flags

    cuda: Enables CUDA support for using GPU devices. Requires the candle/cuda, candle/cudnn, and candle_nn/cuda dependencies.
    default: No additional features are enabled by default.

License

This project is licensed under the MIT License.
Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.