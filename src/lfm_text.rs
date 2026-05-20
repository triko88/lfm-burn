use burn::tensor::{
    Tensor,
    backend::Backend
};

use crate::transformer::Transformer;

#[derive(Clone, Debug)]
pub struct Model<Bknd: Backend> {
    pub embed_token: Tensor<Bknd, 2>,
    pub layers: Vec<Transformer<Bknd>>,
    pub embed_norm: Tensor<Bknd, 1>,
}

