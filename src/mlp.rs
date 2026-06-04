use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend, activation::silu},
};

use crate::config::TextModelConfig;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    w1: Linear<B>,
    w3: Linear<B>,
    w2: Linear<B>,
}

impl<Bknd: Backend> MLP<Bknd> {
    pub fn new(config: &TextModelConfig, device: &Bknd::Device) -> Self {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        Self {
            w1: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
            w2: LinearConfig::new(intermediate, hidden).with_bias(false).init(device),
            w3: LinearConfig::new(hidden, intermediate).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let w1_sliu = silu(self.w1.forward(input.clone()));
        let w3 = self.w3.forward(input);

        self.w2.forward(w1_sliu * w3)
    }

    /// GEMV `(name, k, n)` triples for this block's three projections, read from
    /// the actual weights. A `Linear` built with `new(in, out)` stores weight
    /// `[in, out]`, so `dims()` is exactly `(k, n)`.
    pub fn gemv_shapes(&self) -> [(&'static str, usize, usize); 3] {
        let dims = |l: &Linear<Bknd>| {
            let [k, n] = l.weight.val().dims();
            (k, n)
        };
        let (k1, n1) = dims(&self.w1);
        let (k3, n3) = dims(&self.w3);
        let (k2, n2) = dims(&self.w2);
        [("mlp.w1", k1, n1), ("mlp.w3", k3, n3), ("mlp.w2", k2, n2)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Distribution, Tensor, activation::silu};

    use crate::config::{TextModelConfig, RopeParameters};

    type TB = NdArray;

    fn small_config() -> TextModelConfig {
        TextModelConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_heads: 2,
            num_key_value_heads: 2,
            num_attention_heads: 2,
            vocab_size: 8,
            max_position_embeddings: 16,
            layer_types: vec!["conv".to_string()],
            conv_L_cache: 3,
            conv_bias: false,
            norm_eps: 1e-5,
            rope_parameters: RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: 1_000_000.0,
            },
            tie_embedding: true,
            use_pos_enc: true,
            eos_token_id: None,
        }
    }

    #[test]
    fn new_constructs_from_config() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let _mlp = MLP::<TB>::new(&config, &device);
    }

    #[test]
    fn forward_preserves_hidden_size_shape() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> =
            Tensor::random([2, 5, config.hidden_size], Distribution::Default, &device);
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    #[test]
    fn forward_zero_input_yields_zero_output() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 3, config.hidden_size], &device);
        let y = mlp.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(
            slice.iter().all(|&v| v == 0.0),
            "expected all-zero output (no biases on w1/w2/w3), got {:?}",
            slice,
        );
    }

    #[test]
    fn hydrates_w1_hidden_to_intermediate() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([1, 2, config.hidden_size], &device);
        let y = mlp.w1.forward(x);
        assert_eq!(y.dims(), [1, 2, config.intermediate_size]);
    }

    #[test]
    fn hydrates_w3_hidden_to_intermediate() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([1, 2, config.hidden_size], &device);
        let y = mlp.w3.forward(x);
        assert_eq!(y.dims(), [1, 2, config.intermediate_size]);
    }

    #[test]
    fn hydrates_w2_intermediate_to_hidden() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([1, 2, config.intermediate_size], &device);
        let y = mlp.w2.forward(x);
        assert_eq!(y.dims(), [1, 2, config.hidden_size]);
    }

    #[test]
    fn forward_matches_manual_swiglu_computation() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> =
            Tensor::random([1, 3, config.hidden_size], Distribution::Default, &device);

        let gate = silu(mlp.w1.forward(x.clone()));
        let up = mlp.w3.forward(x.clone());
        let expected = mlp.w2.forward(gate * up);

        let actual = mlp.forward(x);

        assert_eq!(
            actual.into_data().as_slice::<f32>().unwrap(),
            expected.into_data().as_slice::<f32>().unwrap(),
        );
    }

    #[test]
    fn forward_batch_independence() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let mlp = MLP::<TB>::new(&config, &device);

        let row_0: Tensor<TB, 3> =
            Tensor::random([1, 2, config.hidden_size], Distribution::Default, &device);
        let row_1: Tensor<TB, 3> =
            Tensor::random([1, 2, config.hidden_size], Distribution::Default, &device);
        let batched = Tensor::cat(vec![row_0.clone(), row_1.clone()], 0);

        let y_batched = mlp.forward(batched);
        let y_solo_0 = mlp.forward(row_0);
        let y_solo_1 = mlp.forward(row_1);

        let h = config.hidden_size;
        let y_batched_0 = y_batched.clone().slice([0..1, 0..2, 0..h]).into_data();
        let y_batched_1 = y_batched.slice([1..2, 0..2, 0..h]).into_data();

        assert_eq!(
            y_batched_0.as_slice::<f32>().unwrap(),
            y_solo_0.into_data().as_slice::<f32>().unwrap(),
        );
        assert_eq!(
            y_batched_1.as_slice::<f32>().unwrap(),
            y_solo_1.into_data().as_slice::<f32>().unwrap(),
        );
    }
}
