use burn::{
    module::Module,
    nn::{
        RmsNorm,
        RmsNormConfig,
        SwiGlu,
        SwiGluConfig,
    },
    tensor::{
        Tensor,
        backend::Backend,
    },
};

use crate::{
    config::LFMTextConfig,
    layer::{AttnContext, Block, LayerCache},
    self_attn::SelfAttn,
    short_conv::ShortConv,
};

#[derive(Module, Debug, Clone)]
pub struct LFMDecoder<Bknd: Backend> {
    conv: Option<ShortConv<Bknd>>,
    self_attn: Option<SelfAttn<Bknd>>,
    feed_forward: SwiGlu<Bknd>,
    operator_norm: RmsNorm<Bknd>,
    ffn_norm: RmsNorm<Bknd>,
}

impl<Bknd: Backend> Block<Bknd> for LFMDecoder<Bknd> {
    fn forward(
        &self,
        input: Tensor<Bknd, 3>,
        ctx: Option<AttnContext<Bknd>>,
        cache: Option<&mut LayerCache<Bknd>>,
    ) -> Tensor<Bknd, 3> {
        let _ = (input, ctx, cache);
        todo!()
    }
}

impl <Bknd: Backend> LFMDecoder<Bknd> {
    pub fn new(config: &LFMTextConfig, layer_idx: usize, device: &Bknd::Device) -> Self {
        let (conv, self_attn) = match config.layer_types[layer_idx].as_str() {
            "conv" => (Some(ShortConv::new(config, device)), None),
            "full_attention" => (None, Some(SelfAttn::new(config, device))),
            _ => panic!("unknown layer_type at index {layer_idx}"),
        };

        Self {
            conv,
            self_attn,
            feed_forward: SwiGluConfig::new(config.hidden_size, config.intermediate_size)
                .init(device),
            operator_norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.norm_eps).init(device),
            ffn_norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.norm_eps).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Distribution, Tensor};

    use crate::config::{LFMTextConfig, RopeParameters};
    use crate::layer::AttnContext;
    use burn::tensor::{Bool, TensorData};

    fn placeholder_ctx(seq: usize, head_dim: usize, device: &NdArrayDevice) -> AttnContext<TB> {
        let cos: Tensor<TB, 3> = Tensor::ones([1, seq, head_dim], device);
        let sin: Tensor<TB, 3> = Tensor::zeros([1, seq, head_dim], device);
        let mask_data: Vec<bool> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| j <= i))
            .collect();
        let mask: Tensor<TB, 4, Bool> =
            Tensor::from_data(TensorData::new(mask_data, [1, 1, seq, seq]), device);
        AttnContext { cos, sin, mask: Some(mask) }
    }

    type TB = NdArray;

    fn small_config(layer_types: Vec<&str>) -> LFMTextConfig {
        LFMTextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: layer_types.len(),
            num_heads: 2,
            num_key_value_heads: 2,
            num_attention_heads: 2,
            vocab_size: 8,
            max_position_embeddings: 16,
            layer_types: layer_types.into_iter().map(String::from).collect(),
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

    #[test]
    fn forward_preserves_shape_3d_conv_layer() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = dec.forward(x, None, None);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    #[test]
    fn forward_preserves_shape_3d_attn_layer() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["full_attention"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let ctx = placeholder_ctx(5, config.head_dim(), &device);
        let y = dec.forward(x, Some(ctx), None);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    #[test]
    #[should_panic(expected = "unknown layer_type")]
    fn forward_unknown_layer_type_panics() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["banana"]);
        let _ = LFMDecoder::<TB>::new(&config, 0, &device);
    }

    #[test]
    fn forward_is_causal_conv_layer() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let h = config.hidden_size;
        let seq = 6usize;
        let t = 3usize;

        let x_a: Tensor<TB, 3> = Tensor::random([1, seq, h], Distribution::Default, &device);
        let perturbation: Tensor<TB, 3> = Tensor::random([1, seq - (t + 1), h], Distribution::Default, &device);

        let x_b = x_a.clone().slice_assign([0..1, (t + 1)..seq, 0..h], perturbation);

        let y_a = dec.forward(x_a, None, None);
        let y_b = dec.forward(x_b, None, None);

        let prefix_a = y_a.slice([0..1, 0..(t + 1), 0..h]).into_data();
        let prefix_b = y_b.slice([0..1, 0..(t + 1), 0..h]).into_data();

        assert_eq!(prefix_a.as_slice::<f32>().unwrap(), prefix_b.as_slice::<f32>().unwrap());
    }

    #[test]
    fn forward_is_causal_attn_layer() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["full_attention"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let h = config.hidden_size;
        let seq = 6usize;
        let t = 3usize;

        let x_a: Tensor<TB, 3> = Tensor::random([1, seq, h], Distribution::Default, &device);
        let perturbation: Tensor<TB, 3> = Tensor::random([1, seq - (t + 1), h], Distribution::Default, &device);

        let x_b = x_a.clone().slice_assign([0..1, (t + 1)..seq, 0..h], perturbation);

        let ctx_a = placeholder_ctx(seq, config.head_dim(), &device);
        let ctx_b = placeholder_ctx(seq, config.head_dim(), &device);
        let y_a = dec.forward(x_a, Some(ctx_a), None);
        let y_b = dec.forward(x_b, Some(ctx_b), None);

        let prefix_a = y_a.slice([0..1, 0..(t + 1), 0..h]).into_data();
        let prefix_b = y_b.slice([0..1, 0..(t + 1), 0..h]).into_data();

        assert_eq!(prefix_a.as_slice::<f32>().unwrap(), prefix_b.as_slice::<f32>().unwrap());
    }

    // TODO(green-phase): forward_residual_passthrough_when_norms_zero
    // Once the residual structure is finalized, force operator_norm and ffn_norm
    // weights to zero and verify that the decoder reduces to identity on its input.

    #[test]
    fn hydrates_conv_branch_for_conv_layer_type() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        assert!(dec.conv.is_some(), "conv branch must be Some for layer_type='conv'");
        assert!(dec.self_attn.is_none(), "self_attn branch must be None for layer_type='conv'");
    }

    #[test]
    fn hydrates_attn_branch_for_full_attention_layer_type() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["full_attention"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        assert!(dec.conv.is_none(), "conv branch must be None for layer_type='full_attention'");
        assert!(dec.self_attn.is_some(), "self_attn branch must be Some for layer_type='full_attention'");
    }

    #[test]
    fn hydrated_conv_branch_accepts_hidden_size_input() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let conv = dec.conv.as_ref().expect("conv hydrated");
        let x: Tensor<TB, 3> = Tensor::zeros([1, 3, config.hidden_size], &device);
        let y = conv.forward(x, None, None);
        assert_eq!(y.dims(), [1, 3, config.hidden_size]);
    }

    #[test]
    fn hydrated_attn_branch_accepts_hidden_size_input() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["full_attention"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let attn = dec.self_attn.as_ref().expect("self_attn hydrated");
        let x: Tensor<TB, 3> = Tensor::zeros([1, 3, config.hidden_size], &device);
        let ctx = placeholder_ctx(3, config.head_dim(), &device);
        let y = attn.forward(x, Some(ctx), None);
        assert_eq!(y.dims(), [1, 3, config.hidden_size]);
    }

    #[test]
    fn hydrates_feed_forward_hidden_to_intermediate() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([1, 2, config.hidden_size], &device);
        let y = dec.feed_forward.forward(x);
        assert_eq!(y.dims(), [1, 2, config.intermediate_size]);
    }

    #[test]
    fn hydrates_operator_norm_with_hidden_size_io() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let x: Tensor<TB, 3> = Tensor::random([1, 2, config.hidden_size], Distribution::Default, &device);
        let y = dec.operator_norm.forward(x);
        assert_eq!(y.dims(), [1, 2, config.hidden_size]);
    }

    #[test]
    fn hydrates_ffn_norm_with_hidden_size_io() {
        let device = NdArrayDevice::Cpu;
        let config = small_config(vec!["conv"]);
        let dec = LFMDecoder::<TB>::new(&config, 0, &device);

        let x: Tensor<TB, 3> = Tensor::random([1, 2, config.hidden_size], Distribution::Default, &device);
        let y = dec.ffn_norm.forward(x);
        assert_eq!(y.dims(), [1, 2, config.hidden_size]);
    }

}
