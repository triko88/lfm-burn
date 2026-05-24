use burn::module::Module;
use burn::tensor::{
    Tensor,
    backend::Backend,
    Bool,
    Int,
};

use crate::{
    short_conv::{
        ConvCache,
        ShortConv,
    },
    self_attn::{
        AttnCache,
        SelfAttn,
    },
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

#[derive(Module, Debug, Clone)]
pub enum Layer<Bknd: Backend> {
    Conv(ShortConv<Bknd>),
    Attn(SelfAttn<Bknd>),
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

impl <Bknd: Backend> Block<Bknd> for Layer<Bknd> {
    fn forward(
        &self,
        input: Tensor<Bknd, 3>,
        ctx: Option<AttnContext<Bknd>>,
        cache: Option<&mut LayerCache<Bknd>>,
    ) -> Tensor<Bknd, 3> {
        match self {
            Layer::Conv(obj) => obj.forward(input, ctx, cache),
            Layer::Attn(obj) => obj.forward(input, ctx, cache),
        }
    }
}

fn rotate_half<Bknd: Backend>(x: Tensor<Bknd, 4>) -> Tensor<Bknd, 4> {
    let d = x.dims()[3];
    let x1 = x.clone().narrow(3, 0, d/2);
    let x2 = x.narrow(3, d/2, d/2);

    Tensor::cat(vec![-x2, x1], 3)
}

pub fn apply_rope<Bknd: Backend>(x: Tensor<Bknd, 4>, cos: Tensor<Bknd, 3>, sin: Tensor<Bknd, 3>) 
-> Tensor<Bknd, 4> {
    let cos = cos.unsqueeze_dim::<4>(1);
    let sin = sin.unsqueeze_dim::<4>(1);

    x.clone() * cos + rotate_half(x) * sin
}

pub fn causal_mask<Bknd: Backend>(
    seq: usize,
    past: usize,
    device: &Bknd::Device,
) -> Tensor<Bknd, 4, Bool> {
    Tensor::<Bknd, 2, Bool>::tril_mask([seq, past + seq], past as i64, device)
        .unsqueeze::<4>()
}

pub fn rope_tables<Bknd: Backend>(
    past: usize, seq: usize, d_h: usize, 
    theta: f64, dev: &Bknd::Device) 
-> (Tensor<Bknd, 3>, Tensor<Bknd, 3>) {

    let half = d_h / 2;
    let inv: Vec<f32> = (0..half)
        .map(|x| 1.0 / theta.powf(x as f64 / d_h as f64) as f32).collect();
    let inv = Tensor::<Bknd, 1>::from_floats(inv.as_slice(), dev).reshape([1, half]);
    let pos = Tensor::<Bknd, 1, Int>::arange(past as i64..(past + seq) as i64, dev)
        .float().reshape([seq, 1]);
    let freqs = pos.matmul(inv);
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);

    (emb.clone().cos().unsqueeze_dim::<3>(0), emb.sin().unsqueeze_dim::<3>(0))
}
