use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct LFMConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub conv_kernel_size: usize,
    pub layer_types: Vec<ConfigLayer>,
}

#[derive(Deserialize, Debug)]
pub enum ConfigLayer {
    #[serde(alias = "attention")]
    Attention,
    #[serde(alias = "conv")]
    Conv,
}
