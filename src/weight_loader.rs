use std::{
    path::Path,
    fs::{File, read_to_string},
};
use burn::tensor::{
    backend::Backend,
    TensorData,
    Tensor,
};
use safetensors::{
    SafeTensors,
    tensor::TensorView,
};
use memmap2::Mmap;

use crate::{
    self_attn::SelfAttn,
    short_conv::ShortConv,
    transformer::{
        Transformer,
        Sequence,
    },
    lfm_text::Model,
    config::{
        LFMConfig,
        ConfigLayer,
    },
};

fn to_tensor<Bknd: Backend, const Dim: usize>(view: &TensorView, device: &Bknd::Device)
-> Tensor<Bknd, Dim> {

    let bytes = view.data();
    let byte_vec: Vec<f32> = bytemuck::cast_slice(bytes).to_vec();

    let shape = view.shape();
    let tensor_data = TensorData::new(byte_vec, shape);

    Tensor::<Bknd, Dim>::from_data(tensor_data, device)
}

fn build_layers<Bknd: Backend>(config: LFMConfig, tensors: &SafeTensors, device: &Bknd::Device) 
-> Result<Vec<Transformer<Bknd>>, Box<dyn std::error::Error>> {
    let mut res = vec![];
    let mut x = 0;

    for layer in config.layer_types {
        let sequence = match layer {
            ConfigLayer::Conv => {
                Sequence::Conv(ShortConv::<Bknd> {
                    in_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.conv.in_proj.weight"))?, device),
                    out_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.conv.out_proj.weight"))?, device),
                    conv: to_tensor::<Bknd, 3>(&tensors.tensor(&format!("model.layers.{x}.conv.conv.weight"))?, device),
                })
            },
            ConfigLayer::Attention => {
                Sequence::Attention(SelfAttn::<Bknd> {
                    q_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.self_attn.q_proj.weight"))?, device),
                    k_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.self_attn.k_proj.weight"))?, device),
                    v_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.self_attn.v_proj.weight"))?, device),
                    out_proj: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.self_attn.out_proj.weight"))?, device),
                    q_norm: to_tensor::<Bknd, 1>(&tensors.tensor(&format!("model.layers.{x}.self_attn.q_layernorm.weight"))?, device),
                    k_norm: to_tensor::<Bknd, 1>(&tensors.tensor(&format!("model.layers.{x}.self_attn.k_layernorm.weight"))?, device),

                    n_heads: config.num_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.hidden_size / config.num_attention_heads,
                })
            },
        };

        let layer = Transformer::<Bknd> {
            w1: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.feed_forward.w1.weight"))?, device),
            w2: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.feed_forward.w2.weight"))?, device),
            w3: to_tensor::<Bknd, 2>(&tensors.tensor(&format!("model.layers.{x}.feed_forward.w3.weight"))?, device),
            ffn_norm: to_tensor::<Bknd, 1>(&tensors.tensor(&format!("model.layers.{x}.ffn_norm.weight"))?, device),
            operator_norm: to_tensor::<Bknd, 1>(&tensors.tensor(&format!("model.layers.{x}.operator_norm.weight"))?, device),
            sequence
        };

        x += 1;
        res.push(layer);
    }

    println!("layers: {}", res.len());
    Ok(res)
}

pub fn load_model<Bknd: Backend>(repo_path: &Path, device: &Bknd::Device) 
-> Result<Model<Bknd>, Box<dyn std::error::Error>> {
    let model_path = repo_path.join("model.safetensors");
    let config_path = repo_path.join("config.json");

    let model_config: LFMConfig = serde_json::from_str(&read_to_string(config_path)?)?;

    let model_file = File::open(model_path)?;
    let model_mmap = unsafe { Mmap::map(&model_file)? };

    let tensors = SafeTensors::deserialize(&model_mmap)?;
    let layers = build_layers(model_config, &tensors, device)?;

    let tokens = to_tensor::<Bknd, 2>(&tensors.tensor("model.embed_tokens.weight")?, device);
    let norm = to_tensor::<Bknd, 1>(&tensors.tensor("model.embed_norm.weight")?, device);

    Ok(
        Model {
            embed_token: tokens,
            embed_norm: norm,
            layers: layers,
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use std::path::PathBuf;

    fn repo_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_repo")
    }

    #[test]
    fn test_load_model_opens_file() {
        let device = NdArrayDevice::Cpu;
        let result: Result<Model<NdArray>, _> = load_model(&repo_path(), &device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_embed_token_is_2d() {
        let device = NdArrayDevice::Cpu;
        let model: Model<NdArray> = load_model(&repo_path(), &device).unwrap();
        assert_eq!(model.embed_token.shape().num_dims(), 2);
    }

    #[test]
    fn test_embedding_norm_is_1d() {
        let device = NdArrayDevice::Cpu;
        let model: Model<NdArray> = load_model(&repo_path(), &device).unwrap();
        assert_eq!(model.embed_norm.shape().num_dims(), 1);
    }

    #[test]
    fn test_model_has_layers() {
        let device = NdArrayDevice::Cpu;
        let model: Model<NdArray> = load_model(&repo_path(), &device).unwrap();
        assert!(!model.layers.is_empty());
    }
}
