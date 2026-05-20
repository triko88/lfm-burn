use burn::tensor::{
    Tensor,
    backend::Backend,
};

use crate::{
    self_attn::SelfAttn,
    short_conv::ShortConv,
    utils::Block,
};

#[derive(Clone, Debug)]
pub enum Sequence<Bknd: Backend> {
    Attention(SelfAttn<Bknd>),
    Conv(ShortConv<Bknd>),
}

#[derive(Clone, Debug)]
pub struct Transformer<Bknd: Backend> {
    pub w1: Tensor<Bknd, 2>,
    pub w2: Tensor<Bknd, 2>,
    pub w3: Tensor<Bknd, 2>,
    pub operator_norm: Tensor<Bknd, 1>,
    pub ffn_norm: Tensor<Bknd, 1>,
    pub sequence: Sequence<Bknd>,
}

impl<Bknd: Backend> Block<Bknd, 3> for Transformer<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::Tensor;

    fn zero_attn_seq(device: &NdArrayDevice) -> Sequence<NdArray> {
        Sequence::Attention(SelfAttn {
            q_proj: Tensor::zeros([4, 4], device),
            k_proj: Tensor::zeros([4, 4], device),
            v_proj: Tensor::zeros([4, 4], device),
            out_proj: Tensor::zeros([4, 4], device),
            q_norm: Tensor::zeros([4], device),
            k_norm: Tensor::zeros([4], device),
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
        })
    }

    fn zero_conv_seq(device: &NdArrayDevice) -> Sequence<NdArray> {
        Sequence::Conv(ShortConv {
            in_proj: Tensor::zeros([4, 4], device),
            out_proj: Tensor::zeros([4, 4], device),
            conv: Tensor::zeros([4, 1, 3], device),
        })
    }

    fn zero_transformer(device: &NdArrayDevice, seq: Sequence<NdArray>) -> Transformer<NdArray> {
        Transformer {
            w1: Tensor::zeros([4, 4], device),
            w2: Tensor::zeros([4, 4], device),
            w3: Tensor::zeros([4, 4], device),
            operator_norm: Tensor::zeros([4], device),
            ffn_norm: Tensor::zeros([4], device),
            sequence: seq,
        }
    }

    #[test]
    fn test_transformer_forward_preserves_shape_with_attention() {
        let device = NdArrayDevice::Cpu;
        let layer = zero_transformer(&device, zero_attn_seq(&device));
        let input: Tensor<NdArray, 3> = Tensor::zeros([2, 5, 4], &device);
        let out = layer.forward(input);
        assert_eq!(out.shape().dims(), [2, 5, 4]);
    }

    #[test]
    fn test_transformer_forward_preserves_shape_with_conv() {
        let device = NdArrayDevice::Cpu;
        let layer = zero_transformer(&device, zero_conv_seq(&device));
        let input: Tensor<NdArray, 3> = Tensor::zeros([2, 5, 4], &device);
        let out = layer.forward(input);
        assert_eq!(out.shape().dims(), [2, 5, 4]);
    }

    #[test]
    fn test_transformer_zero_weights_zero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let layer = zero_transformer(&device, zero_attn_seq(&device));
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 3, 4], &device);
        let out = layer.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_transformer_dispatches_to_conv_variant() {
        let device = NdArrayDevice::Cpu;
        let layer = zero_transformer(&device, zero_conv_seq(&device));
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 2, 4], &device);
        let out = layer.forward(input);
        assert_eq!(out.shape().dims(), [1, 2, 4]);
    }

    #[test]
    fn test_transformer_dispatches_to_attention_variant() {
        let device = NdArrayDevice::Cpu;
        let layer = zero_transformer(&device, zero_attn_seq(&device));
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 2, 4], &device);
        let out = layer.forward(input);
        assert_eq!(out.shape().dims(), [1, 2, 4]);
    }
}
