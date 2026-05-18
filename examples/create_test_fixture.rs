use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn zeros(count: usize) -> Vec<u8> {
    vec![0u8; count * 4]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tiny dims: hidden=4, vocab=8, conv_kernel=3
    let h = 4usize;
    let v = 8usize;
    let k = 3usize;

    let d2_hxh = zeros(h * h);
    let d2_vxh = zeros(v * h);
    let d1_h = zeros(h);
    let d3_hxhxk = zeros(h * h * k);

    let mut tensors: HashMap<String, TensorView> = HashMap::new();

    tensors.insert(
        "model.embed_tokens.weight".into(),
        TensorView::new(Dtype::F32, vec![v, h], &d2_vxh)?,
    );
    tensors.insert(
        "model.embed_norm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_h)?,
    );

    // MLP fields — loader uses literal "{x}", not a format string
    for key in ["w1", "w2", "w3"] {
        tensors.insert(
            format!("model.layers.{{x}}.feed_forward.{key}.weight"),
            TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
        );
    }
    tensors.insert(
        "model.layers.{x}.ffn_norm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_h)?,
    );
    tensors.insert(
        "model.layers.{x}.operator_norm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_h)?,
    );

    // Attention sequence — loader uses literal "{x}"
    for key in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        tensors.insert(
            format!("model.layers.{{x}}.self_attn.{key}.weight"),
            TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
        );
    }
    tensors.insert(
        "model.layers.{x}.self_attn.q_layernorm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_h)?,
    );
    tensors.insert(
        "model.layers.{x}.self_attn.k_layernorm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_h)?,
    );

    // Conv sequence — loader uses literal "{x}"
    tensors.insert(
        "model.layers.{x}.conv.in_proj.weight".into(),
        TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
    );
    tensors.insert(
        "model.layers.{x}.conv.out.weight".into(),
        TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
    );
    tensors.insert(
        "model.layers.{x}.conv.conv.weight".into(),
        TensorView::new(Dtype::F32, vec![h, h, k], &d3_hxhxk)?,
    );

    let bytes = safetensors::serialize(&tensors, None)?;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_repo/model.safetensors");
    fs::write(&path, &bytes)?;
    println!("Wrote {} bytes to {}", bytes.len(), path.display());

    Ok(())
}
