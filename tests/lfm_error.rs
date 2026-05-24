use burn::backend::{NdArray, ndarray::NdArrayDevice};
use lfm_rs::{Lfm, LfmError};

// A.7 — LfmError is a typed error returned from from_pretrained
#[test]
fn from_pretrained_missing_dir_returns_typed_error() {
    let device = NdArrayDevice::Cpu;
    let err = Lfm::<NdArray>::from_pretrained("does/not/exist", &device).unwrap_err();
    assert!(matches!(
        err,
        LfmError::Config(_) | LfmError::Weights(_) | LfmError::Io(_)
    ));
}
