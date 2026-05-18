use burn::tensor::{
    Tensor,
    backend::Backend,
};

#[derive(Clone, Debug)]
pub struct SelfAttn<Bknd: Backend> {
    pub q_proj: Tensor<Bknd, 2>,
    pub k_proj: Tensor<Bknd, 2>,
    pub v_proj: Tensor<Bknd, 2>,
    pub out_proj: Tensor<Bknd, 2>,
    pub q_norm: Tensor<Bknd, 1>,
    pub k_norm: Tensor<Bknd, 1>,
}

#[derive(Clone, Debug)]
pub struct ShortConv<Bknd: Backend> {
    pub in_proj: Tensor<Bknd, 2>,
    pub out_proj: Tensor<Bknd, 2>,
    pub conv: Tensor<Bknd, 3>,
}

#[derive(Clone, Debug)]
pub enum Sequence<Bknd: Backend> {
    Attention(SelfAttn<Bknd>),
    Conv(ShortConv<Bknd>),
}

#[derive(Clone, Debug)]
pub struct LFMLayer<Bknd: Backend> {
    pub w1: Tensor<Bknd, 2>,
    pub w2: Tensor<Bknd, 2>,
    pub w3: Tensor<Bknd, 2>,
    pub operator_norm: Tensor<Bknd, 1>,
    pub ffn_norm: Tensor<Bknd, 1>,
    pub sequence: Sequence<Bknd>,
}

#[derive(Clone, Debug)]
pub struct Model<Bknd: Backend> {
    pub embed_token: Tensor<Bknd, 2>,
    pub layers: Vec<LFMLayer<Bknd>>,
    pub embed_norm: Tensor<Bknd, 1>,
}
