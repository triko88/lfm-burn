use burn::tensor::{
    Tensor,
    activation::softmax,
    backend::Backend,
    Bool,
};

use crate::utils::{
    Block,
    rms_norm,
};

#[derive(Clone, Debug)]
pub struct SelfAttn<Bknd: Backend> {
    pub q_proj: Tensor<Bknd, 2>,
    pub k_proj: Tensor<Bknd, 2>,
    pub v_proj: Tensor<Bknd, 2>,
    pub out_proj: Tensor<Bknd, 2>,
    pub q_norm: Tensor<Bknd, 1>,
    pub k_norm: Tensor<Bknd, 1>,

    // Read from config.json
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<Bknd: Backend> Block<Bknd, 3> for SelfAttn<Bknd> {
    fn forward(&self, input: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let [batch, seq_len, hidden] = input.dims();
        let device = input.device();

        let heads = self.n_heads;
        let kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;

        let n_groups = heads / kv_heads;

        let x = input.reshape([batch * seq_len, hidden]);

        let q = x.clone().matmul(self.q_proj.clone()).reshape([batch, seq_len, heads, head_dim]);
        let k = x.clone().matmul(self.k_proj.clone()).reshape([batch, seq_len, kv_heads, head_dim]);
        let v = x.clone().matmul(self.v_proj.clone()).reshape([batch, seq_len, kv_heads, head_dim]);

        let q = rms_norm(q, self.q_norm.clone(), 1e-6);
        let k = rms_norm(k, self.k_norm.clone(), 1e-6);

        let expanded = [batch, seq_len, kv_heads, n_groups, head_dim];
        let bcast_shape = [batch, seq_len, heads, head_dim];

        let k = k.unsqueeze_dim::<5>(3).expand(expanded.clone()).reshape(bcast_shape.clone());
        let v = v.unsqueeze_dim::<5>(3).expand(expanded).reshape(bcast_shape);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        let mask = Tensor::<Bknd, 2, Bool>::tril_mask([seq_len, seq_len], 0, &device);
        let scores = scores.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);

        let attn = softmax(scores, 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([batch * seq_len, heads * head_dim]);

        out.matmul(self.out_proj.clone()).reshape([batch, seq_len, hidden])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Tensor, TensorData};

    fn zero_attn(device: &NdArrayDevice) -> SelfAttn<NdArray> {
        SelfAttn {
            q_proj: Tensor::zeros([4, 4], device),
            k_proj: Tensor::zeros([4, 4], device),
            v_proj: Tensor::zeros([4, 4], device),
            out_proj: Tensor::zeros([4, 4], device),
            q_norm: Tensor::zeros([4], device),
            k_norm: Tensor::zeros([4], device),
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
        }
    }

    #[test]
    fn test_self_attn_forward_preserves_shape() {
        let device = NdArrayDevice::Cpu;
        let attn = zero_attn(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([2, 5, 4], &device);
        let out = attn.forward(input);
        assert_eq!(out.shape().dims(), [2, 5, 4]);
    }

    #[test]
    fn test_self_attn_zero_input_zero_output() {
        let device = NdArrayDevice::Cpu;
        let attn = zero_attn(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 3, 4], &device);
        let out = attn.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_self_attn_zero_weights_nonzero_input_yields_zero() {
        let device = NdArrayDevice::Cpu;
        let attn = zero_attn(&device);
        let input_data = TensorData::new(vec![1.0_f32; 12], vec![1, 3, 4]);
        let input: Tensor<NdArray, 3> = Tensor::from_data(input_data, &device);
        let out = attn.forward(input);
        let data = out.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_self_attn_single_token_seq() {
        let device = NdArrayDevice::Cpu;
        let attn = zero_attn(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([1, 1, 4], &device);
        let out = attn.forward(input);
        assert_eq!(out.shape().dims(), [1, 1, 4]);
    }

    #[test]
    fn test_self_attn_batch_dim_preserved() {
        let device = NdArrayDevice::Cpu;
        let attn = zero_attn(&device);
        let input: Tensor<NdArray, 3> = Tensor::zeros([3, 2, 4], &device);
        let out = attn.forward(input);
        assert_eq!(out.dims()[0], 3);
    }
}
