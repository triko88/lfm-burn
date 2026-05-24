// End-to-end integration test: load the test_repo fixture and run forward on a
// small token tensor. The crate must expose `LFMText` publicly for this file to
// compile — that export is a green-phase prerequisite tracked alongside the
// other compile-error preconditions in the test plan.

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Int, Tensor};

use lfm_rs::LFMText;

type TB = NdArray;

#[test]
fn loads_test_repo_and_runs_forward() {
    let device = NdArrayDevice::Cpu;
    let model: LFMText<TB> =
        LFMText::from_pretrained("test_repo", &device).expect("from_pretrained");

    let ids: Tensor<TB, 2, Int> = Tensor::zeros([1, 4], &device);
    let y = model.forward(ids, None);

    let dims = y.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 4);
    // Final dim is either hidden_size (4) or vocab_size (8) depending on whether
    // green phase folds in the tied embedding head; accept either.
    assert!(
        dims[2] == 4 || dims[2] == 8,
        "expected last dim 4 (hidden) or 8 (vocab); got {}",
        dims[2],
    );
}
