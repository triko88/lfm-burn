use burn::{
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        RmsNorm,
        RmsNormConfig,
    },
    tensor::{
        Tensor,
        backend::Backend,
    },
};

use crate::{
    config::LFMTextConfig,
    utils::Block,
};

#[derive(Module, Debug, Clone)]
pub struct SelfAttn<Bknd: Backend> {
    q_proj: Linear<Bknd>,
    k_proj: Linear<Bknd>,
    v_proj: Linear<Bknd>,
    out_proj: Linear<Bknd>,
    q_norm: RmsNorm<Bknd>,
    k_norm: RmsNorm<Bknd>,

    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) num_groups: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f32,
}

impl<Bknd: Backend> Block<Bknd, 3> for SelfAttn<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let _ = input;
        todo!()
    }
}

impl <Bknd: Backend> SelfAttn<Bknd> {
    pub fn new(config: &LFMTextConfig, device: &Bknd::Device) -> Self {
        let h = config.hidden_size;
        let d = config.head_dim();
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;

        Self {
            q_proj: LinearConfig::new(h, nq * d).with_bias(false).init(device),
            k_proj: LinearConfig::new(h, nkv * d).with_bias(false).init(device),
            v_proj: LinearConfig::new(h, nkv * d).with_bias(false).init(device),
            out_proj: LinearConfig::new(nq * d, h).with_bias(false).init(device),

            q_norm: RmsNormConfig::new(d).with_epsilon(config.norm_eps).init(device),
            k_norm: RmsNormConfig::new(d).with_epsilon(config.norm_eps).init(device),

            num_q_heads: nq,
            num_kv_heads: nkv,
            num_groups: config.num_kv_groups(),
            head_dim: d,
            scale: (d as f32).powf(-0.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::nn::{Initializer, LinearConfig, RmsNormConfig};
    use burn::tensor::{Distribution, Tensor};

    use crate::config::{LFMTextConfig, RopeParameters};
    use crate::utils::Block;

    type TB = NdArray;

    fn small_config(num_attention_heads: usize, num_key_value_heads: usize, hidden_size: usize) -> LFMTextConfig {
        LFMTextConfig {
            hidden_size,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_heads: num_attention_heads,
            num_key_value_heads,
            num_attention_heads,
            vocab_size: 8,
            max_position_embeddings: 16,
            layer_types: vec!["full_attention".to_string()],
            conv_L_cache: 3,
            conv_bias: false,
            norm_eps: 1e-5,
            rope_params: RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: 1_000_000.0,
            },
            tie_embedding: true,
            use_pos_end: true,
        }
    }

    fn zero_self_attn(config: &LFMTextConfig, device: &NdArrayDevice) -> SelfAttn<TB> {
        let h = config.hidden_size;
        let d = config.head_dim();
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        SelfAttn {
            q_proj: LinearConfig::new(h, nq * d).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            k_proj: LinearConfig::new(h, nkv * d).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            v_proj: LinearConfig::new(h, nkv * d).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            out_proj: LinearConfig::new(nq * d, h).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            q_norm: RmsNormConfig::new(d).with_epsilon(config.norm_eps).init(device),
            k_norm: RmsNormConfig::new(d).with_epsilon(config.norm_eps).init(device),
            num_q_heads: nq,
            num_kv_heads: nkv,
            num_groups: config.num_kv_groups(),
            head_dim: d,
            scale: (d as f32).powf(-0.5),
        }
    }

    #[test]
    fn forward_preserves_shape_3d() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = attn.forward(x);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    #[test]
    fn forward_seq_len_one() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([1, 1, config.hidden_size], Distribution::Default, &device);
        let y = attn.forward(x);
        assert_eq!(y.dims(), [1, 1, config.hidden_size]);
    }

    #[test]
    fn forward_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let y = attn.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_zero_weights_yield_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = zero_self_attn(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = attn.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_is_causal() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let h = config.hidden_size;
        let seq = 6usize;
        let t = 3usize;

        let x_a: Tensor<TB, 3> = Tensor::random([1, seq, h], Distribution::Default, &device);
        let perturbation: Tensor<TB, 3> = Tensor::random([1, seq - (t + 1), h], Distribution::Default, &device);

        let x_b = x_a.clone().slice_assign(
            [0..1, (t + 1)..seq, 0..h],
            perturbation,
        );

        let y_a = attn.forward(x_a);
        let y_b = attn.forward(x_b);

        let prefix_a = y_a.slice([0..1, 0..(t + 1), 0..h]).into_data();
        let prefix_b = y_b.slice([0..1, 0..(t + 1), 0..h]).into_data();

        assert_eq!(prefix_a.as_slice::<f32>().unwrap(), prefix_b.as_slice::<f32>().unwrap());
    }

    #[test]
    fn forward_gqa_broadcast() {
        // nq=4, nkv=2, head_dim=4, hidden=16
        let device = NdArrayDevice::Cpu;
        let config = small_config(4, 2, 16);
        let attn = SelfAttn::<TB>::new(&config, &device);

        assert_eq!(attn.num_groups, 2);
        assert_eq!(attn.head_dim, 4);

        let x: Tensor<TB, 3> = Tensor::random([1, 3, config.hidden_size], Distribution::Default, &device);
        let y = attn.forward(x);
        assert_eq!(y.dims(), [1, 3, config.hidden_size]);
    }
}
