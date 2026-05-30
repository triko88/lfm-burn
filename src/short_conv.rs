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
    config::TextModelConfig,
    layer::{AttnContext, Block, LayerCache},
};

#[derive(Module, Debug)]
pub struct ShortConv<B: Backend> {
    conv: Conv1d<B>,
    in_proj: Linear<B>,
    out_proj: Linear<B>,

    l_cache: usize,
    hidden_size: usize,
}

#[derive(Clone, Debug)]
pub struct ConvCache<Bknd: Backend> {
    window: Tensor<Bknd, 3>,
    seeded: bool,
}

impl<Bknd: Backend> ConvCache<Bknd> {
    pub fn init(hidden_size: usize, l_cache: usize, device: &Bknd::Device) -> Self {
        Self {
            window: Tensor::zeros([1, hidden_size, l_cache - 1], device),
            seeded: false,
        }
    }

    fn push(&mut self, bx_col: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        // History holds the last l-1 input columns; append the current column to
        // form the full length-l convolution window [b, h, l].
        let full = Tensor::cat(vec![self.window.clone(), bx_col], 2);

        // Retain the most recent l-1 columns as history for the next step.
        let k = full.dims()[2];
        self.window = full.clone().narrow(2, 1, k - 1);

        full
    }

    fn seed(&mut self, bx: Tensor<Bknd, 3>, device: &Bknd::Device) {
        let [b, d, l] = bx.dims();
        let k = self.window.dims()[2];

        self.window = if l >= k {
            bx.narrow(2, l - k, k)
        } else {
            // Prefill shorter than the history window: left-pad with zeros to length k.
            Tensor::cat(vec![Tensor::zeros([b, d, k - l], device), bx], 2)
        };

        self.seeded = true;
    }
}

impl<Bknd: Backend> Block<Bknd> for ShortConv<Bknd> {
    fn forward(
        &self,
        input: Tensor<Bknd, 3>,
        ctx: Option<AttnContext<Bknd>>,
        cache: Option<&mut LayerCache<Bknd>>,
    ) -> Tensor<Bknd, 3> {
        let _ = ctx;
        let device = input.device();
        let seqlen = input.dims()[1];

        // Reshape it to [Bknd, 3H, T]
        let bcx = self.in_proj.forward(input).swap_dims(1, 2);

        let chunks  = bcx.chunk(3, 1);      // Converts 3 tensors shaped [b, h, t]
        let b_gate  = chunks[0].clone();
        let c_gate  = chunks[1].clone();
        let x_inner = chunks[2].clone();

        let bx = b_gate * x_inner;
        let conv_out = match cache {
            Some(LayerCache::ConvCache(c)) if c.seeded => {
                // Decode => calculate for Single token, depthwise dot product over the window.
                let window = c.push(bx);
                let [d, _, k] = self.conv.weight.val().dims();
                let kernel = self.conv.weight.val().reshape([1, d, k]);
                (window *kernel).sum_dim(2)
            },
            Some(LayerCache::ConvCache(c)) => {
                // Prefill => Run the convolution, seed the window
                let seqlen = bx.dims()[2];
                let out = self.conv.forward(bx.clone()).narrow(2, 0, seqlen);
                c.seed(bx, &device);

                out
            },
            _ => self.conv.forward(bx).narrow(2, 0, seqlen),
        };
        // forward makes [b, h, t + l - 1], narrow it to [b, h, t]

        let y = (c_gate * conv_out).swap_dims(1, 2); // [b, t, h]
        self.out_proj.forward(y)
    }
}

impl<Bknd: Backend> ShortConv<Bknd> {
    pub(crate) fn cache_dims(&self) -> (usize, usize) {
        (self.hidden_size, self.l_cache)
    }

    pub fn new(config: &TextModelConfig, device: &Bknd::Device) -> Self {
        let h = config.hidden_size;
        let l = config.conv_L_cache;

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

    fn zero_short_conv(config: &TextModelConfig, device: &NdArrayDevice) -> ShortConv<TB> {
        let h = config.hidden_size;
        let l = config.conv_L_cache;
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
            let y = conv.forward(x, None, None);
            assert_eq!(y.dims(), [2, s, config.hidden_size]);
        }
    }

    #[test]
    fn forward_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let config = small_config();
        let conv = ShortConv::<TB>::new(&config, &device);

        let x: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let y = conv.forward(x, None, None);
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
        let y = conv.forward(x, None, None);
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

        let y_a = conv.forward(x_a, None, None);
        let y_b = conv.forward(x_b, None, None);

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
        let y_batched = conv.forward(batched, None, None);

        let y_solo_0 = conv.forward(row_0, None, None);
        let y_solo_1 = conv.forward(row_1, None, None);

        let y_batched_0 = y_batched.clone().slice([0..1, 0..4, 0..h]).into_data();
        let y_batched_1 = y_batched.slice([1..2, 0..4, 0..h]).into_data();

        assert_eq!(y_batched_0.as_slice::<f32>().unwrap(), y_solo_0.into_data().as_slice::<f32>().unwrap());
        assert_eq!(y_batched_1.as_slice::<f32>().unwrap(), y_solo_1.into_data().as_slice::<f32>().unwrap());
    }
}
