# CandleEmbed 🧨

Text embeddings with any model on hugging face. Embedded in your app. Replace your API bill with a GPU.

### Features

- Enums for most popular embedding models OR specify custom models from HF (check out the [leaderboard](https://huggingface.co/spaces/mteb/leaderboard))

- GPU support with CUDA

- Builder with easy access to configuration settings

### Installation

Add the following to your Cargo.toml file:

```toml
[dependencies]
candle_embed = "*"

[dependencies]
candle_embed = { version = "*", features = ["cuda"] } // For CUDA support
```

### Basics 🫡

```rust

use candle_embed::{CandleEmbedBuilder, WithModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a builder with default settings
    //
    let candle_embed = CandleEmbedBuilder::new().build()?;
    
    // Embed a single text
    //
    let text = "This is a test sentence.";
    let embeddings = candle_embed.embed_one(text)?;
    
    // Embed a batch of texts
    //
    let texts = vec![
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
    ];
    let batch_embeddings = candle_embed.embed_batch(texts)?;
    

    Ok(())
}
```

### Custom 🤓

```rust
    // Custom settings
    //
    builder
        .approximate_gelu(false)
        .mean_pooling(true)
        .normalize_embeddings(false)
        .truncate_text_len_overflow(true)

    // Set model from preset
    //
    builder
        .set_model_from_presets(WithModel::UaeLargeV1);

    // Or use a custom model and revision
    //
    builder
        .custom_embedding_model("avsolatorio/GIST-small-Embedding-v0")
        .custom_model_revision("d6c4190")

    // Will use the first available CUDA device (Default)
    //
    builder.with_device_any_cuda()

    // Use a specific CUDA device
    //
    builder.with_device_specific_cuda(ordinal: usize);

    // Use CPU (CUDA options fail over to this)
    //
    builder.with_device_cpu()

    // Unload the model and tokenizer, dropping them from memory
    //
    candle_embed.unload();

    // ---

    // These are automatically loaded from the model's `config.json` after builder init

    // model_dimensions
    // This is the same as "hidden_size"
    //
    let dimensions = candle_embed.model_dimensions;

    // model_max_input 
    // This is the same as "max_position_embeddings"
    // If `truncate_text_len_overflow == false`, and your input exceeds this a panic will result
    // If you don't want to worry about this, the default truncation strategy will just chop the end off the input
    // However, you lose accuracy by mindlessly truncating your inputs
    //
    let max_position_embeddings = candle_embed.model_max_input;
    

    // ---
```

### Tokenization 🧮

```rust
    // Generate tokens using the model

    let texts = vec![
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ];
    let text = "This is the first sentence.";

    // Get tokens from a batch of texts
    //
    let batch_tokens = candle_embed.tokenize_batch(&texts)?;
    assert_eq!(batch_tokens.len(), 3);
    // Get tokens from a single text
    //
    let tokens = candle_embed.tokenize_one(text)?;
    assert!(!tokens.is_empty());

    // Get a count of tokens
    // This is important to use if you are using your own chunking strategy
    // For example, using a text splitter on any text string whose token count exceeds candle_embed.model_max_input

    // Get token counts from a batch of texts
    //
    let batch_tokens = candle_embed.token_count_batch(&texts)?;
   
    // Get token count from a single text
    //
    let tokens = candle_embed.token_count(text)?;
    assert!(tokens > 0);
```

### How is this differant than text-embeddings-inference 🤗

- TEI implements a client-server model. This requires running it as it's own external server, a docker container, or locally as a server.
- CandleEmbed is made to be embedded: it can be installed as a crate and runs in process.
- TEI is very well optimized and very scalable.
- CandleEmbed is fast (with a GPU), but was not created for serving at the scale, of say, HuggingFace's text embeddings API.


[text-embeddings-inference]('https://github.com/huggingface/text-embeddings-inference') is a more established project, and well respected. I recommend you check it out!

### How is this differant than fastembed.rs 🦀

- Both are usable as a crate!
- Custom models downloaded and ran from hf-hub by entering their `repo_name/model_id`.
- CandleEmbed is designed so projects can implement their own truncation and/or chunking strategies. 
- CandleEmbed uses CUDA. FastEmbed uses ONNX.
- And finaly.. CandleEmbed uses [Candle](https://github.com/huggingface/candle).

[fastembed.rs]('https://github.com/Anush008/fastembed-rs') is a more established project, and well respected. I recommend you check it out!


### Roadmap

- Multi-GPU support
- Benchmarking system

### License

This project is licensed under the MIT License.

### Contributing

My motivation for publishing is for someone to point out if I'm doing something wrong!