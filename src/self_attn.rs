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
        activation::softmax,
    },
};

use crate::{
    config::LFMTextConfig,
    layer::{AttnContext, Block, LayerCache, apply_rope},
};

#[derive(Debug)]
pub struct AttnCache<Bknd: Backend> {
    k: Option<Tensor<Bknd, 4>>,
    v: Option<Tensor<Bknd, 4>>,
    len: usize,
}

impl <Bknd: Backend> AttnCache<Bknd> {
    fn append(&mut self, k_new: Tensor<Bknd, 4>, v_new: Tensor<Bknd, 4>)
    -> (Tensor<Bknd, 4>, Tensor<Bknd, 4>) {
        let k = match self.k.take() {
            Some(k) => Tensor::cat(vec![k, k_new], 2),
            None => k_new,
        };

        let v = match self.v.take() {
            Some(v) => Tensor::cat(vec![v, v_new], 2),
            None => v_new,
        };

        self.len = k.dims()[2];
        self.k = Some(k.clone());
        self.v = Some(v.clone());

        (k, v)
    }
}

#[derive(Module, Debug, Clone)]
pub struct SelfAttn<Bknd: Backend> {
    q_proj: Linear<Bknd>,
    k_proj: Linear<Bknd>,
    v_proj: Linear<Bknd>,
    out_proj: Linear<Bknd>,
    q_norm: RmsNorm<Bknd>,
    k_norm: RmsNorm<Bknd>,

    pub(crate) num_heads: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) num_groups: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f32,
}

impl<Bknd: Backend> Block<Bknd> for SelfAttn<Bknd> {
    fn forward(
        &self,
        input: Tensor<Bknd, 3>,
        ctx: Option<AttnContext<Bknd>>,
        cache: Option<&mut LayerCache<Bknd>>,
    ) -> Tensor<Bknd, 3> {
        let [batch, seqlen, hidden] = input.dims();
        let qh = self.num_q_heads;
        let kvh = self.num_kv_heads;
        let h = self.head_dim;

        let ctx = ctx.expect("AttnContext required");

        let q = self.q_proj.forward(input.clone()).reshape([batch, seqlen, qh, h]);
        let k = self.k_proj.forward(input.clone()).reshape([batch, seqlen, kvh, h]);
        let v = self.v_proj.forward(input).reshape([batch, seqlen, kvh, h]);

        let q = apply_rope(self.q_norm.forward(q.swap_dims(1, 2)), ctx.cos.clone(), ctx.sin.clone());
        let k = apply_rope(self.k_norm.forward(k.swap_dims(1, 2)), ctx.cos.clone(), ctx.sin.clone());
        let v = v.swap_dims(1, 2);

        let (k, v) = match cache {
            Some(LayerCache::AttnCache(c)) => c.append(k, v),
            _ => (k, v),
        };

        let expanded_dims = [batch, kvh, self.num_groups, seqlen, h];
        let repeat_dims = [batch, self.num_heads, seqlen, h];

        let k = k.unsqueeze_dim::<5>(2).expand(expanded_dims).reshape(repeat_dims);
        let v = v.unsqueeze_dim::<5>(2).expand(expanded_dims).reshape(repeat_dims);

        let mut scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        if let Some(mask) = ctx.mask.clone() {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        let attn = softmax(scores, 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([batch * seqlen, self.num_heads * h]);

        self.out_proj.forward(out).reshape([batch, seqlen, hidden])
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

            num_heads: config.num_heads,
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
    use crate::layer::{causal_mask, AttnContext};

    fn placeholder_ctx(seq: usize, head_dim: usize, device: &NdArrayDevice) -> AttnContext<TB> {
        let cos: Tensor<TB, 3> = Tensor::ones([1, seq, head_dim], device);
        let sin: Tensor<TB, 3> = Tensor::zeros([1, seq, head_dim], device);
        let mask = causal_mask::<TB>(seq, 0, device);
        AttnContext { cos, sin, mask: Some(mask) }
    }

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
            num_heads: config.num_heads,
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
        let ctx = placeholder_ctx(5, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    #[test]
    fn forward_seq_len_one() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([1, 1, config.hidden_size], Distribution::Default, &device);
        let ctx = placeholder_ctx(1, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
        assert_eq!(y.dims(), [1, 1, config.hidden_size]);
    }

    #[test]
    fn forward_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(2, 2, 4);
        let attn = SelfAttn::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let ctx = placeholder_ctx(5, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
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
        let ctx = placeholder_ctx(5, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
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

        let ctx_a = placeholder_ctx(seq, config.head_dim(), &device);
        let ctx_b = placeholder_ctx(seq, config.head_dim(), &device);
        let y_a = attn.forward(x_a, Some(ctx_a), None);
        let y_b = attn.forward(x_b, Some(ctx_b), None);

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
        let ctx = placeholder_ctx(3, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
        assert_eq!(y.dims(), [1, 3, config.hidden_size]);
    }
}
