use burn::{
    module::Module,
    nn::{
        Linear,
        LinearConfig,
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
pub struct MLP<Bknd: Backend> {
    w1: Linear<Bknd>,
    w2: Linear<Bknd>,
    w3: Linear<Bknd>,
}

impl<Bknd: Backend> Block<Bknd, 3> for MLP<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let _ = input;
        todo!()
    }
}

impl<Bknd: Backend> MLP<Bknd> {
    pub fn new(config: &LFMTextConfig, device: &Bknd::Device) -> Self {
        let ff_dim = config.adjusted_ff_dim();

        Self {
            w1: LinearConfig::new(config.hidden_size, ff_dim).with_bias(false).init(device),
            w2: LinearConfig::new(config.hidden_size, ff_dim).with_bias(false).init(device),
            w3: LinearConfig::new(config.hidden_size, ff_dim).with_bias(false).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::nn::{Initializer, LinearConfig};
    use burn::tensor::{Distribution, Tensor};

    use crate::config::{LFMTextConfig, RopeParameters};

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

    fn zero_mlp(config: &LFMTextConfig, device: &NdArrayDevice) -> MLP<TB> {
        let h = config.hidden_size;
        let ff = config.adjusted_ff_dim();
        MLP {
            w1: LinearConfig::new(h, ff).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            w2: LinearConfig::new(ff, h).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            w3: LinearConfig::new(h, ff).with_bias(false).with_initializer(Initializer::Zeros).init(device),
        }
    }

    #[test]
    fn forward_preserves_shape_3d() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        for (b, s) in [(1usize, 1usize), (1, 5), (2, 5)] {
            let x: Tensor<TB, 3> = Tensor::random([b, s, config.hidden_size], Distribution::Default, &device);
            let y = mlp.forward(x);
            assert_eq!(y.dims(), [b, s, config.hidden_size]);
        }
    }

    #[test]
    fn forward_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let y = mlp.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0), "expected all-zero output, got {:?}", slice);
    }

    #[test]
    fn forward_zero_weights_yield_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = zero_mlp(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = mlp.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0), "expected all-zero output, got {:?}", slice);
    }

    #[test]
    fn forward_batch_dim_preserved() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::random([3, 2, config.hidden_size], Distribution::Default, &device);
        let y = mlp.forward(x);
        assert_eq!(y.dims()[0], 3);
        assert_eq!(y.dims()[1], 2);
        assert_eq!(y.dims()[2], config.hidden_size);
    }
}
