use burn::backend::{NdArray, ndarray::NdArrayDevice};
use lfm_rs::LFMText;

// A.8 — prompt() returns a non-empty string
#[tokio::test]
async fn prompt_returns_non_empty_string() {
    let device = NdArrayDevice::Cpu;
    let lfm = LFMText::<NdArray>::from_pretrained("test_repo", &device)
        .expect("from_pretrained should succeed")
        .with_max_tokens(4);
    let out = lfm.prompt("hello").await.expect("prompt should succeed");
    assert!(!out.is_empty(), "expected non-empty completion, got empty string");
}

// A.8 — max_tokens is respected
#[tokio::test]
async fn prompt_respects_max_tokens() {
    let device = NdArrayDevice::Cpu;
    let lfm = LFMText::<NdArray>::from_pretrained("test_repo", &device)
        .expect("from_pretrained should succeed")
        .with_max_tokens(2);
    let out = lfm.prompt("hello").await.expect("prompt should succeed");
    // 2 tokens × ~10 chars/token is a conservative ceiling.
    assert!(out.len() < 100, "output suggests max_tokens was ignored: {:?}", out);
}
