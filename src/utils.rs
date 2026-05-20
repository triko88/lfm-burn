use burn::tensor::{
    Tensor,
    backend::Backend,
};

pub fn rms_norm<Bknd: Backend> (x: Tensor<Bknd, 4>, scale: Tensor<Bknd, 1>, eps: f32) -> Tensor<Bknd, 4> {
    let head_dim    = scale.dims()[0];
    let variance    = x.clone().powf_scalar(2.0).mean_dim(3);
    let inv_rms     = variance.add_scalar(eps).sqrt().recip();
    let norm        = x * inv_rms;

    norm * scale.reshape([1, 1, 1, head_dim])
}

pub trait Block<Bknd: Backend, const Dim: usize> {
    fn forward(&self, input: Tensor<Bknd, Dim>) -> Tensor<Bknd, Dim>; 
}
