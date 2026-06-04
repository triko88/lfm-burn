use burn::backend::NdArray;
// To profile on the GPU instead, swap the backend below and rebuild:
//   use burn::backend::Wgpu;
// GPU memory is reported when `nvidia-smi` or `rocm-smi` is available.
use std::path::Path;
use lfm_rs::{LFMError, LFMText};

#[tokio::main]
async fn main() -> Result<(), LFMError> {
    // Flags can appear anywhere; remaining positionals are [model_dir] [prompt].
    let mut steady_state = false;
    let mut gemv = false;
    let mut positionals = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--steady-state" => steady_state = true,
            "--gemv" => gemv = true,
            _ => positionals.push(arg),
        }
    }
    let mut positionals = positionals.into_iter();
    let source = positionals.next().unwrap_or_else(|| "test_repo".to_string());
    let prompt = positionals.next().unwrap_or_else(|| "The capital of France is".to_string());

    let dir = resolve_model_dir(&source)?;

    let device = Default::default();
    let lfm = LFMText::<NdArray>::from_pretrained(&dir, &device)?.with_max_tokens(64);

    // Warmup/measured-iteration counts for the microbenchmarks.
    const WARMUP: usize = 5;
    const ITERS: usize = 50;

    if gemv {
        println!("=== GEMV microbench (warmup={WARMUP}, iters={ITERS}) ===");
        for result in lfm.gemv_microbench(WARMUP, ITERS) {
            println!("{result}");
        }
        println!();
    }

    if steady_state {
        let report = lfm.steady_state_latency(&prompt, WARMUP, ITERS)?;
        println!("=== steady-state latency (warmup={WARMUP}, iters={ITERS}) ===\n{report}\n");
    }

    // Default end-to-end profile runs unless a microbench mode was requested.
    if !gemv && !steady_state {
        let (text, report) = lfm.prompt_profiled(&prompt).await?;
        println!("=== output ===\n{text}\n");
        println!("=== profile ===\n{report}");
    }

    Ok(())
}

// Local path wins if it exists; otherwise treat `source` as a Hugging Face
// model id (e.g. "LiquidAI/LFM2-350M") and pull the three required files into
// the hf-hub cache, returning the snapshot directory they share.
fn resolve_model_dir(source: &str) -> Result<String, LFMError> {
    if Path::new(source).is_dir() {
        return Ok(source.to_string());
    }

    println!("Local dir '{source}' not found - fetching from Hugging Face...");

    let api = hf_hub::api::sync::Api::new()
        .map_err(|err| LFMError::IO(format!("hf-hub init: {err}")))?;
    let repo = api.model(source.to_string());

    let mut snapshot_dir = None;
    for file in ["config.json", "model.safetensors", "tokenizer.json"] {
        let path = repo.get(file)
            .map_err(|err| LFMError::IO(format!("hf-hub fetch {file}: {err}")))?;
        if snapshot_dir.is_none() {
            snapshot_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    let dir = snapshot_dir
        .ok_or_else(|| LFMError::IO("hf-hub returned no snapshot directory".into()))?;
    Ok(dir.to_string_lossy().into_owned())
}
