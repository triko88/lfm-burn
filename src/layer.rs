use burn::tensor::{
    Tensor,
    backend::Backend,
    Bool,
};

use crate::{
    short_conv::ConvCache,
    self_attn::AttnCache,
};

pub struct AttnContext<Bknd: Backend> {
    pub cos: Tensor<Bknd, 3>,
    pub sin: Tensor<Bknd, 3>,
    pub mask: Option<Tensor<Bknd, 4, Bool>>,
}

pub enum LayerCache<Bknd: Backend> {
    ConvCache(ConvCache<Bknd>),
    AttnCache(AttnCache<Bknd>),
}

pub trait Block<Bknd: Backend> {
    fn forward(
        &self,
        input: Tensor<Bknd, 3>,
        ctx: Option<AttnContext<Bknd>>,
        cache: Option<&mut LayerCache<Bknd>>,
    ) -> Tensor<Bknd, 3>;
    // For both types of layers, the tensors are shaped (B, L, D) -> (B, L, D)
}
