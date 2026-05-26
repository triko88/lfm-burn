use burn::config::Config;

#[derive(Config, Debug)]
pub struct RopeParameters {
    #[config(default = "\"default\".to_string()")]
    pub rope_type: String,
    #[config(default = "1_000_000.0")]
    pub rope_theta: f32,
}

#[derive(Config, Debug)]
pub struct TextModelConfig {
    // Dimensions
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub num_attention_heads: usize,

    // Vocab / context
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    // Layers for hybrid structure
    pub layer_types: Vec<String>,

    // ShortConv block
    #[config(default = "3")]
    #[allow(non_snake_case)]
    pub conv_L_cache: usize,
    #[config(default = "false")]
    pub conv_bias: bool,

    #[config(default = "1e-5")]
    pub norm_eps: f64,

    #[config(default = "RopeParameters { rope_type: \"default\".to_string(), rope_theta: 1_000_000.0 }")]
    pub rope_parameters: RopeParameters,

    // Output 
    #[config(default = "true")]
    pub tie_embedding: bool,
    #[config(default = "true")]
    pub use_pos_enc: bool,
    #[config(default = "None")]
    pub eos_token_id: Option<u32>,
}

impl TextModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn adjusted_ff_dim(&self) -> usize {
        self.intermediate_size
    }
}
