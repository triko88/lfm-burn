use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct LFMConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub num_attention_heads: usize,
    pub conv_L_cache: usize,
    pub layer_types: Vec<ConfigLayer>,
}

#[derive(Deserialize, Debug)]
pub enum ConfigLayer {
    #[serde(alias = "full_attention")]
    Attention,
    #[serde(alias = "conv")]
    Conv,
}
