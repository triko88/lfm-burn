use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn zeros(count: usize) -> Vec<u8> {
    vec![0u8; count * 4]
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tiny dims: hidden=4, vocab=8, conv_kernel=3
    let h = 4usize;
    let v = 8usize;
    let k = 3usize;

    // Must match test_repo/config.json layer_types
    let layer_types = ["conv", "full_attention"];

    let d2_hxh = zeros(h * h);
    let d1_h = zeros(h);

    // Craft a deterministic, runnable toy model so generation yields real text:
    //   - per-layer operator_norm/ffn_norm are zero (gamma=0), making every
    //     decoder an exact identity, so the hidden stream stays = embed(token);
    //   - the final embedding_norm (embed_norm) is ones so it doesn't zero out;
    //   - embed_tokens row 4 ('a') dominates, so logits always argmax to token 4
    //     (a non-special token). Generation therefore emits "a" repeatedly.
    let mut embed = vec![0.1f32; v * h];
    for c in 0..h {
        embed[4 * h + c] = 10.0;
    }
    let d2_vxh = f32_bytes(&embed);
    let d1_ones = f32_bytes(&vec![1.0f32; h]);
    // Depthwise conv weight: [out_channels=h, in_channels/groups=1, kernel=k].
    let d3_hx1xk = zeros(h * 1 * k);
    // Gated in_proj (h -> 3h), stored PyTorch-orientation [out=3h, in=h]; the
    // PyTorchToBurnAdapter transposes it to burn's [h, 3h] on load.
    let d2_3hxh = zeros(3 * h * h);

    let mut tensors: HashMap<String, TensorView> = HashMap::new();

    tensors.insert(
        "model.embed_tokens.weight".into(),
        TensorView::new(Dtype::F32, vec![v, h], &d2_vxh)?,
    );
    tensors.insert(
        "model.embed_norm.weight".into(),
        TensorView::new(Dtype::F32, vec![h], &d1_ones)?,
    );

    for (x, layer_type) in layer_types.iter().enumerate() {
        for key in ["w1", "w2", "w3"] {
            tensors.insert(
                format!("model.layers.{x}.feed_forward.{key}.weight"),
                TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
            );
        }
        tensors.insert(
            format!("model.layers.{x}.ffn_norm.weight"),
            TensorView::new(Dtype::F32, vec![h], &d1_h)?,
        );
        tensors.insert(
            format!("model.layers.{x}.operator_norm.weight"),
            TensorView::new(Dtype::F32, vec![h], &d1_h)?,
        );

        match *layer_type {
            "conv" => {
                tensors.insert(
                    format!("model.layers.{x}.conv.in_proj.weight"),
                    TensorView::new(Dtype::F32, vec![3 * h, h], &d2_3hxh)?,
                );
                tensors.insert(
                    format!("model.layers.{x}.conv.out_proj.weight"),
                    TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
                );
                tensors.insert(
                    format!("model.layers.{x}.conv.conv.weight"),
                    TensorView::new(Dtype::F32, vec![h, 1, k], &d3_hx1xk)?,
                );
            }
            "full_attention" => {
                for key in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    tensors.insert(
                        format!("model.layers.{x}.self_attn.{key}.weight"),
                        TensorView::new(Dtype::F32, vec![h, h], &d2_hxh)?,
                    );
                }
                tensors.insert(
                    format!("model.layers.{x}.self_attn.q_layernorm.weight"),
                    TensorView::new(Dtype::F32, vec![h], &d1_h)?,
                );
                tensors.insert(
                    format!("model.layers.{x}.self_attn.k_layernorm.weight"),
                    TensorView::new(Dtype::F32, vec![h], &d1_h)?,
                );
            }
            other => return Err(format!("unknown layer_type: {other}").into()),
        }
    }

    let bytes = safetensors::serialize(&tensors, None)?;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_repo/model.safetensors");
    fs::write(&path, &bytes)?;
    println!("Wrote {} bytes to {}", bytes.len(), path.display());

    Ok(())
}
