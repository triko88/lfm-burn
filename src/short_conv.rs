use burn::{
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
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
pub struct ShortConv<Bknd: Backend> {
    conv: Conv1d<Bknd>,
    in_proj: Linear<Bknd>,
    out_proj: Linear<Bknd>,

    l_cache: usize,
    hidden_size: usize,
}

impl<Bknd: Backend> Block<Bknd, 3> for ShortConv<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let _ = input;
        todo!()
    }
}

impl<Bknd: Backend> ShortConv<Bknd> {
    pub fn new(config: &LFMTextConfig, device: &Bknd::Device) -> Self {
        let h = config.hidden_size;
        let l = config.conv_l_cache;

        let conv = Conv1dConfig::new(h, h, l)
            .with_groups(h)
            .with_padding(PaddingConfig1d::Explicit(l - 1, 0))
            .with_bias(config.conv_bias)
            .init(device);

        let in_proj = LinearConfig::new(h, 3 * h).with_bias(config.conv_bias).init(device);
        let out_proj = LinearConfig::new(h, h).with_bias(config.conv_bias).init(device);

        Self {
            conv, in_proj, out_proj,
            l_cache: l,
            hidden_size: h
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::nn::{
        Initializer, LinearConfig,
        conv::Conv1dConfig,
        PaddingConfig1d,
    };
    use burn::tensor::{Distribution, Tensor};

    use crate::config::{LFMTextConfig, RopeParameters};
    use crate::utils::Block;

    type TB = NdArray;

    fn small_config() -> LFMTextConfig {
        LFMTextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_heads: 2,
            num_key_value_heads: 2,
            num_attention_heads: 2,
            vocab_size: 8,
            max_position_embeddings: 16,
            layer_types: vec!["conv".to_string()],
            conv_l_cache: 3,
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

    fn zero_short_conv(config: &LFMTextConfig, device: &NdArrayDevice) -> ShortConv<TB> {
        let h = config.hidden_size;
        let l = config.conv_l_cache;
        let conv = Conv1dConfig::new(h, h, l)
            .with_groups(h)
            .with_padding(PaddingConfig1d::Explicit(l - 1, 0))
            .with_bias(config.conv_bias)
            .with_initializer(Initializer::Zeros)
            .init(device);
        ShortConv {
            conv,
            in_proj: LinearConfig::new(h, 3 * h)
                .with_bias(config.conv_bias)
                .with_initializer(Initializer::Zeros)
                .init(device),
            out_proj: LinearConfig::new(h, h)
                .with_bias(config.conv_bias)
                .with_initializer(Initializer::Zeros)
                .init(device),
            l_cache: l,
            hidden_size: h,
        }
    }

    #[test]
    fn forward_preserves_shape_3d() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = ShortConv::<TB>::new(&config, &device);

        for s in [1usize, 5, 8] {
            let x: Tensor<TB, 3> = Tensor::random([2, s, config.hidden_size], Distribution::Default, &device);
            let y = conv.forward(x);
            assert_eq!(y.dims(), [2, s, config.hidden_size]);
        }
    }

    #[test]
    fn forward_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = ShortConv::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let y = conv.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_zero_weights_yield_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = zero_short_conv(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = conv.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_is_causal() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = ShortConv::<TB>::new(&config, &device);

        let h = config.hidden_size;
        let seq = 6usize;
        let t = 3usize; // boundary: positions [0..=t] must be invariant to changes at t+1..

        let x_a: Tensor<TB, 3> = Tensor::random([1, seq, h], Distribution::Default, &device);
        let perturbation: Tensor<TB, 3> = Tensor::random([1, seq - (t + 1), h], Distribution::Default, &device);

        let x_b = x_a.clone().slice_assign(
            [0..1, (t + 1)..seq, 0..h],
            perturbation,
        );

        let y_a = conv.forward(x_a);
        let y_b = conv.forward(x_b);

        let prefix_a = y_a.slice([0..1, 0..(t + 1), 0..h]).into_data();
        let prefix_b = y_b.slice([0..1, 0..(t + 1), 0..h]).into_data();

        assert_eq!(prefix_a.as_slice::<f32>().unwrap(), prefix_b.as_slice::<f32>().unwrap());
    }

    #[test]
    fn forward_batch_dim_preserved() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = ShortConv::<TB>::new(&config, &device);

        let h = config.hidden_size;
        let row_0: Tensor<TB, 3> = Tensor::random([1, 4, h], Distribution::Default, &device);
        let row_1: Tensor<TB, 3> = Tensor::random([1, 4, h], Distribution::Default, &device);

        let batched = Tensor::cat(vec![row_0.clone(), row_1.clone()], 0);
        let y_batched = conv.forward(batched);

        let y_solo_0 = conv.forward(row_0);
        let y_solo_1 = conv.forward(row_1);

        let y_batched_0 = y_batched.clone().slice([0..1, 0..4, 0..h]).into_data();
        let y_batched_1 = y_batched.slice([1..2, 0..4, 0..h]).into_data();

        assert_eq!(y_batched_0.as_slice::<f32>().unwrap(), y_solo_0.into_data().as_slice::<f32>().unwrap());
        assert_eq!(y_batched_1.as_slice::<f32>().unwrap(), y_solo_1.into_data().as_slice::<f32>().unwrap());
    }
}
