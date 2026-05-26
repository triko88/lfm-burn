use std::rc::Rc;

use burn::tensor::DType;
use burn_store::{ModuleAdapter, TensorSnapshot};

/// Adapter that converts BF16 tensors to F32, passing all other dtypes through unchanged.
///
/// This is needed because `burn-ndarray` does not support BF16, but many Hugging Face
/// models (including LFM) store their weights in BF16 format.
#[derive(Debug, Clone)]
pub struct Bf16ToF32Adapter;

impl ModuleAdapter for Bf16ToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        match snapshot.dtype {
            DType::BF16 => {
                let target_dtype = DType::F32;
                let original_data_fn = snapshot.clone_data_fn();

                let cast_data_fn = Rc::new(move || {
                    let data = original_data_fn()?;
                    Ok(data.convert_dtype(target_dtype))
                });

                TensorSnapshot::from_closure(
                    cast_data_fn,
                    target_dtype,
                    snapshot.shape.clone(),
                    snapshot.path_stack.clone().unwrap_or_default(),
                    snapshot.container_stack.clone().unwrap_or_default(),
                    snapshot.tensor_id.unwrap_or_default(),
                )
            }
            _ => snapshot.clone(),
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
