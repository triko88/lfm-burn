use burn::backend::NdArray;
// To profile on the GPU instead, swap the backend below and rebuild:
//   use burn::backend::Wgpu;
// GPU memory is reported when `nvidia-smi` or `rocm-smi` is available.
use lfm_rs::{LFMError, LFMText};

#[tokio::main]
async fn main() -> Result<(), LFMError> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "test_repo".to_string());
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".to_string());

    let device = Default::default();
    let lfm = LFMText::<NdArray>::from_pretrained(&dir, &device)?.with_max_tokens(64);

    let (text, report) = lfm.prompt_profiled(&prompt).await?;

    println!("=== output ===\n{text}\n");
    println!("=== profile ===\n{report}");
    Ok(())
}
