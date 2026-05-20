use burn::tensor::{
    Tensor,
    backend::Backend
};

use crate::utils::{
    Block,
    rms_norm,
};

#[derive(Clone, Debug)]
pub struct ShortConv<Bknd: Backend> {
    pub in_proj: Tensor<Bknd, 2>,
    pub out_proj: Tensor<Bknd, 2>,
    pub conv: Tensor<Bknd, 3>,
}

impl<Bknd: Backend> Block<Bknd, 3> for ShortConv<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Tensor, TensorData};

    fn zero_conv(device: &NdArrayDevice) -> ShortConv<NdArray> {
        ShortConv {
            in_proj: Tensor::zeros([4, 4], device),
            out_proj: Tensor::zeros([4, 4], device),
            conv: Tensor::zeros([4, 1, 3], device),
        }
    }

    #[test]
    fn test_short_conv_forward_preserves_shape() {
        let device = NdArrayDevice::Cpu;
        let conv = zero_conv(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([2, 5, 4], &device);
        let out = conv.forward(input);
        assert_eq!(out.shape().dims(), [2, 5, 4]);
    }

    #[test]
    fn test_short_conv_zero_input_zero_output() {
        let device = NdArrayDevice::Cpu;
        let conv = zero_conv(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 3, 4], &device);
        let out = conv.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_short_conv_zero_weights_nonzero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let conv = zero_conv(&device);
        let input_data = TensorData::new(vec![1.0_f32; 12], vec![1, 3, 4]);
        let input: Tensor<NdArray, 3> = Tensor::from_data(input_data, &device);
        let out = conv.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_short_conv_seq_len_one_shape() {
        let device = NdArrayDevice::Cpu;
        let conv = zero_conv(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 1, 4], &device);
        let out = conv.forward(input);
        assert_eq!(out.shape().dims(), [1, 1, 4]);
    }

    #[test]
    fn test_short_conv_batch_dim_preserved() {
        let device = NdArrayDevice::Cpu;
        let conv = zero_conv(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([3, 4, 4], &device);
        let out = conv.forward(input);
        assert_eq!(out.dims()[0], 3);
    }
}
