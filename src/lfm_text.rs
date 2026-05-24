use burn::{
    module::Module,
    config::Config,
    nn::{
        Embedding,
        EmbeddingConfig,
        RmsNorm,
        RmsNormConfig,
    },
    tensor::{
        Int,
        Tensor,
        backend::Backend,
    },
};

use crate::{
    decoder::LFMDecoder,
    config::LFMTextConfig,
};

use burn_store::{ModuleSnapshot, SafetensorsStore};

#[derive(Module, Debug, Clone)]
pub struct LFMText<Bknd: Backend> {
    embed_tokens: Embedding<Bknd>,
    layers: Vec<LFMDecoder<Bknd>>,
    embedding_norm: RmsNorm<Bknd>,
}

impl LFMTextConfig {
    pub fn init<Bknd: Backend>(&self, device: &Bknd::Device) -> LFMText<Bknd> {
        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size)
            .init(device);
        let layers = (0..self.num_hidden_layers)
            .map(|i| LFMDecoder::new(self, i, device))
            .collect();
        let embedding_norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.norm_eps)
            .init(device);

        LFMText {
            embed_tokens,
            layers,
            embedding_norm,
        }
    }
}

impl <Bknd: Backend> LFMText<Bknd> {
    pub fn forward(&self, token_ids: Tensor<Bknd, 2, Int>) -> Tensor<Bknd, 3> {
        let _ = token_ids;
        todo!()
    }

    pub fn from_pretrained(dir: &str, device: &Bknd::Device)
        -> Result<Self, Box<dyn std::error::Error>>
    {
        let config = LFMTextConfig::load(format!("{dir}/config.json"))?;
        let model = config.init::<Bknd>(device);

        let mut store = SafetensorsStore::from_file(format!("{dir}/model.safetensors"))
            .with_key_remapping(r"^model\.", "")
            .allow_partial(true);

        let mut model = model;
        <LFMText<Bknd> as ModuleSnapshot<Bknd>>::load_from(&mut model, &mut store)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::nn::{EmbeddingConfig, Initializer, RmsNormConfig};
    use burn::tensor::{Int, Tensor};

    use crate::config::{LFMTextConfig, RopeParameters};
    use crate::decoder::LFMDecoder;

    type TB = NdArray;

    fn hybrid_config() -> LFMTextConfig {
        LFMTextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_heads: 2,
            num_key_value_heads: 2,
            num_attention_heads: 2,
            vocab_size: 8,
            max_position_embeddings: 16,
            layer_types: vec!["conv".to_string(), "full_attention".to_string()],
            conv_L_cache: 3,
            conv_bias: false,
            norm_eps: 1e-5,
            rope_params: RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: 1_000_000.0,
            },
            tie_embedding: true,
            use_pos_end: true,
        }
    }

    fn build_synthetic_model(config: &LFMTextConfig, device: &NdArrayDevice, zero_embed: bool) -> LFMText<TB> {
        let embed_init = if zero_embed { Initializer::Zeros } else { Initializer::Normal { mean: 0.0, std: 1.0 } };
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .with_initializer(embed_init)
            .init(device);
        let layers: Vec<LFMDecoder<TB>> = (0..config.num_hidden_layers)
            .map(|i| LFMDecoder::<TB>::new(config, i, device))
            .collect();
        let embedding_norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.norm_eps)
            .init(device);

        LFMText {
            embed_tokens,
            layers,
            embedding_norm,
        }
    }

    fn assert_output_shape_is_hidden_or_vocab(dims: &[usize], expected_batch: usize, expected_seq: usize, config: &LFMTextConfig) {
        assert_eq!(dims[0], expected_batch);
        assert_eq!(dims[1], expected_seq);
        let last = dims[2];
        assert!(
            last == config.hidden_size || last == config.vocab_size,
            "expected last dim to be hidden_size ({}) or vocab_size ({}); got {}",
            config.hidden_size, config.vocab_size, last,
        );
    }

    #[test]
    fn forward_preserves_shape() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);

        let ids: Tensor<TB, 2, Int> = Tensor::zeros([2, 5], &device);
        let y = model.forward(ids);
        let dims = y.dims();
        assert_output_shape_is_hidden_or_vocab(&dims, 2, 5, &config);
    }

    #[test]
    fn forward_zero_embedding_yields_zero() {
        // TODO(green-phase): once we know if RmsNorm is applied to the embedding before
        // the residual stream, decide whether to also zero the embedding_norm scale.
        // For now the embedding is zero AND we expect the implementation to either
        // (a) emit zero, or (b) emit NaN/zero depending on norm placement.
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, true);

        let ids: Tensor<TB, 2, Int> = Tensor::zeros([1, 4], &device);
        let y = model.forward(ids);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert!(slice.iter().all(|&v| v == 0.0), "expected all-zero output, got {:?}", slice);
    }

    #[test]
    fn forward_is_causal_full_stack() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);

        let seq = 5usize;
        let t = 2usize;
        let h = config.hidden_size;

        let ids_a: Tensor<TB, 2, Int> = Tensor::from_data([[1i64, 2, 3, 4, 5]], &device);
        // Mutate position t+1 onward.
        let ids_b: Tensor<TB, 2, Int> = Tensor::from_data([[1i64, 2, 3, 7, 6]], &device);

        let y_a = model.forward(ids_a);
        let y_b = model.forward(ids_b);
        let last = y_a.dims()[2];

        let prefix_a = y_a.slice([0..1, 0..(t + 1), 0..last]).into_data();
        let prefix_b = y_b.slice([0..1, 0..(t + 1), 0..last]).into_data();

        assert_eq!(prefix_a.as_slice::<f32>().unwrap(), prefix_b.as_slice::<f32>().unwrap());
        let _ = seq;
        let _ = h;
    }

    #[test]
    fn forward_batch_dim_preserved() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);

        let row_0: Tensor<TB, 2, Int> = Tensor::from_data([[1i64, 2]], &device);
        let row_1: Tensor<TB, 2, Int> = Tensor::from_data([[3i64, 4]], &device);
        let batched = Tensor::cat(vec![row_0.clone(), row_1.clone()], 0);

        let y_batched = model.forward(batched);
        let y_solo_0 = model.forward(row_0);
        let y_solo_1 = model.forward(row_1);

        let last = y_batched.dims()[2];
        let y_batched_0 = y_batched.clone().slice([0..1, 0..2, 0..last]).into_data();
        let y_batched_1 = y_batched.slice([1..2, 0..2, 0..last]).into_data();

        assert_eq!(y_batched_0.as_slice::<f32>().unwrap(), y_solo_0.into_data().as_slice::<f32>().unwrap());
        assert_eq!(y_batched_1.as_slice::<f32>().unwrap(), y_solo_1.into_data().as_slice::<f32>().unwrap());
    }

    #[test]
    fn from_pretrained_loads_test_repo() {
        let device = NdArrayDevice::Cpu;
        let result: Result<LFMText<TB>, _> = LFMText::from_pretrained("test_repo", &device);
        assert!(result.is_ok(), "from_pretrained failed: {:?}", result.err().map(|e| e.to_string()));
    }

    #[test]
    fn from_pretrained_layer_count_matches_config() {
        let device = NdArrayDevice::Cpu;
        let model: LFMText<TB> = LFMText::from_pretrained("test_repo", &device).expect("load");
        // test_repo/config.json declares 2 layers (conv, full_attention).
        assert_eq!(model.layers.len(), 2);
    }
}
