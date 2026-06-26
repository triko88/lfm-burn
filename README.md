# lfm-rs

> [!NOTE]
> **I have halted the progress of this project** after testing my engine on
> LFM-2.5. The engine takes a minute or two when cold starting an SLM. While
> the steady state show promising results, I find this moment to be a good
> opportunity to explore kernel dispatch. I've started a separate project in C++23
> to address this issue.

A lightweight inference engine for Liquid AI's Liquid Foundation Models (LFM),
built on [Burn](https://burn.dev). Inspired by SQLite's software philosophy,
`lfm-rs` runs language models *as a library*: embedded, in-process, and with
no server to deploy.

## Philosophy

SQLite isn't a database server you connect to; it's a library you link against,
and your program *is* the database engine. `lfm-rs` takes the same stance toward
language models:

- **Library, not a service.** No daemon, no RPC, no separate process. You call a
  function and get tokens back, in your own process.
- **Embedded and self-contained.** A model is a local directory of weights and a
  tokenizer. Point the library at it and go — no network at runtime.
- **Backend-agnostic.** Because it's built on Burn, the same code runs on CPU
  (`NdArray`), GPU (`Wgpu`), and future backends without source changes.
- **Small, auditable surface.** A handful of public types, typed errors, and
  no hidden global state.

## Supported models

The project is in its early stages. The inference core targets LFM2-style hybrid
architectures (interleaved short-convolution and grouped-query attention blocks,
RMSNorm, RoPE, tied embeddings).

**Available today**

- [x] LFM2-style **text** models — greedy decoding with a KV cache, loaded from a
      Hugging Face style local directory.

**Roadmap**

- [x] LFM 2.5 350M (text)
- [ ] LFM 2.5 1.2B — Instruct and Thinking (text)
- [ ] LFM 2.5 Audio
- [ ] LFM 2.5 Vision
- [ ] LFM 2 — text, vision, and audio modalities

The architecture is designed so new model versions add a backbone and a single
dispatch arm, and new modalities (vision, audio) add a processor and pipeline
type — without changing the text path, the generation core, or the streaming
machinery.

## Quick start

Add the crate and a Burn backend to your `Cargo.toml`, then point `LFMText` at a
local model directory containing `config.json`, `model.safetensors`, and
`tokenizer.json`:

```rust
use burn::backend::NdArray;
use lfm_rs::{LFMText, LFMError};

#[tokio::main]
async fn main() -> Result<(), LFMError> {
    let device = Default::default();

    let lfm = LFMText::<NdArray>::from_pretrained("path/to/model", &device)?
        .with_max_tokens(256);

    let output = lfm.prompt("The capital of France is").await?;
    println!("{output}");

    Ok(())
}
```

Swap `NdArray` for `Wgpu` to run on the GPU — the rest of the code is unchanged.

## Design notes

- **Prefill + KV cache.** The prompt is processed in a single forward pass;
  per-token decoding reuses the cache rather than rescanning the prefix.
- **Async without blocking the runtime.** Compute-bound forward passes run on a
  blocking worker (`spawn_blocking`) so the async runtime stays responsive.
- **Typed errors.** Fallible operations return `Result<_, LFMError>`; loading an
  unsupported or malformed model returns an error rather than panicking.
- **Backend generic.** Public types are parameterized over `Backend`, so callers
  pick the compute backend.

Incremental token streaming (`Stream<Item = String>`), multi-turn sessions, and
sampling strategies beyond greedy and are on
the path to v1.

## Building

```sh
cargo build
cargo test
```

The test suite runs against a fixture model directory (`test_repo`).

## License

Licensed under the [MIT License](LICENSE).
