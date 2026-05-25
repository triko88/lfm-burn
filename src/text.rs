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
    config::TextModelConfig,
    short_conv::ConvCache,
    self_attn::AttnCache,
    layer::{causal_mask, rope_tables, AttnContext, Block, LayerCache},
};

use burn_store::{ModuleSnapshot, SafetensorsStore};

#[derive(Module, Debug)]
pub struct TextModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<LFMDecoder<B>>,
    embedding_norm: RmsNorm<B>,

    // RoPE
    head_dim: usize,
    rope_theta: f32,
}

pub struct LFMCache<Bknd: Backend> {
    pub layers: Vec<LayerCache<Bknd>>,
    pub position: usize,
}

impl <Bknd: Backend> LFMCache <Bknd> {
    pub fn init(model: &TextModel<Bknd>, device: &Bknd::Device) -> Self {
        let layers = model.layers.iter().map(|dec| {
            if dec.is_attn() {
                LayerCache::AttnCache(AttnCache::init())
            } else {
                let (h, l) = dec.conv_cache_dims()
                    .expect("non-attn decoder must have conv cache dims");

                LayerCache::ConvCache(ConvCache::init(h, l, device))
            }
        }).collect();

        Self {
            layers,
            position: 0,
        }
    }
}

impl TextModelConfig {
    pub fn init<Bknd: Backend>(&self, device: &Bknd::Device) -> TextModel<Bknd> {
        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size)
            .init(device);
        let layers = (0..self.num_hidden_layers)
            .map(|i| LFMDecoder::new(self, i, device))
            .collect();
        let embedding_norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.norm_eps)
            .init(device);

        TextModel {
            embed_tokens,
            layers,
            embedding_norm,
            head_dim: self.head_dim(),
            rope_theta: self.rope_params.rope_theta,
        }
    }
}

impl <Bknd: Backend> TextModel<Bknd> {
    pub fn forward(
        &self,
        token_ids: Tensor<Bknd, 2, Int>,
        mut cache: Option<&mut LFMCache<Bknd>>,
    ) -> Tensor<Bknd, 3> {
        let seq = token_ids.dims()[1];
        let device = token_ids.device();

        let past = cache.as_deref().map_or(0, |c| c.position);
        let theta = self.rope_theta as f64;

        let x = self.embed_tokens.forward(token_ids);

        let (cos, sin) = rope_tables::<Bknd>(past, seq, self.head_dim, theta, &device);
        let mask = causal_mask::<Bknd>(seq, past, &device);

        let x = self.layers.iter().enumerate().fold(x, |x, (idx, layer)| {
            let local_cache = cache.as_deref_mut().map(|c| &mut c.layers[idx]);
            let ctx = layer.is_attn().then(|| AttnContext {
                cos: cos.clone(),
                sin: sin.clone(),
                mask: Some(mask.clone()),
            });

            layer.forward(x, ctx, local_cache)
        });

        let out = self.embedding_norm.forward(x);
        if let Some(c) = cache {
            c.position += seq;
        }

        out
    }

    pub fn lm_head(&self, hidden: Tensor<Bknd, 3>) -> Tensor<Bknd, 3> {
        let w = self.embed_tokens.weight.val();
        let [b, t, h] = hidden.dims();
        let v = w.dims()[0];

        hidden.reshape([b * t, h]).matmul(w.transpose()).reshape([b, t, v])
    }

    pub fn from_pretrained(dir: &str, device: &Bknd::Device)
        -> Result<Self, Box<dyn std::error::Error>>
    {
        let config = TextModelConfig::load(format!("{dir}/config.json"))?;
        let mut model = config.init::<Bknd>(device);

        // The checkpoint uses the HF LFM2 key layout; remap it onto this crate's
        // burn module paths:
        //   - strip the leading `model.`
        //   - the per-layer operator submodule is named `conv`/`self_attn` in the
        //     checkpoint but is a single `layer` field (a `Layer` enum) here
        //   - burn's RmsNorm stores its scale as `gamma`, not `weight`
        // `skip_enum_variants(true)` drops the `Conv`/`Attn` enum-variant names so
        // the operator path is just `layers.N.layer.<...>`.
        let mut store = SafetensorsStore::from_file(format!("{dir}/model.safetensors"))
            .with_key_remapping(r"^model\.", "")
            .with_key_remapping(r"^embed_norm\.", "embedding_norm.")
            .with_key_remapping(r"\.conv\.", ".layer.")
            .with_key_remapping(r"\.self_attn\.", ".layer.")
            .with_key_remapping(r"norm\.weight$", "norm.gamma")
            .skip_enum_variants(true)
            .allow_partial(false);

        let result = <TextModel<Bknd> as ModuleSnapshot<Bknd>>::load_from(&mut model, &mut store)?;
        if !result.missing.is_empty() || !result.unused.is_empty() || !result.errors.is_empty() {
            return Err(format!(
                "incomplete weight load: applied={}, missing={:?}, unused={:?}, errors={:?}",
                result.applied.len(),
                result.missing.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
                result.unused,
                result.errors,
            )
            .into());
        }

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

    use crate::config::{TextModelConfig, RopeParameters};
    use crate::decoder::LFMDecoder;
    use crate::layer::LayerCache;

    type TB = NdArray;

    fn hybrid_config() -> TextModelConfig {
        TextModelConfig {
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
            eos_token_id: None,
        }
    }

    fn build_synthetic_model(config: &TextModelConfig, device: &NdArrayDevice, zero_embed: bool) -> TextModel<TB> {
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

        TextModel {
            embed_tokens,
            layers,
            embedding_norm,
            head_dim: config.head_dim(),
            rope_theta: config.rope_params.rope_theta,
        }
    }

    fn assert_output_shape_is_hidden_or_vocab(dims: &[usize], expected_batch: usize, expected_seq: usize, config: &TextModelConfig) {
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
        let y = model.forward(ids, None);
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
        let y = model.forward(ids, None);
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

        let y_a = model.forward(ids_a, None);
        let y_b = model.forward(ids_b, None);
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

        let y_batched = model.forward(batched, None);
        let y_solo_0 = model.forward(row_0, None);
        let y_solo_1 = model.forward(row_1, None);

        let last = y_batched.dims()[2];
        let y_batched_0 = y_batched.clone().slice([0..1, 0..2, 0..last]).into_data();
        let y_batched_1 = y_batched.slice([1..2, 0..2, 0..last]).into_data();

        assert_eq!(y_batched_0.as_slice::<f32>().unwrap(), y_solo_0.into_data().as_slice::<f32>().unwrap());
        assert_eq!(y_batched_1.as_slice::<f32>().unwrap(), y_solo_1.into_data().as_slice::<f32>().unwrap());
    }

    #[test]
    fn from_pretrained_loads_test_repo() {
        let device = NdArrayDevice::Cpu;
        let result: Result<TextModel<TB>, _> = TextModel::from_pretrained("test_repo", &device);
        assert!(result.is_ok(), "from_pretrained failed: {:?}", result.err().map(|e| e.to_string()));
    }

    #[test]
    fn from_pretrained_layer_count_matches_config() {
        let device = NdArrayDevice::Cpu;
        let model: TextModel<TB> = TextModel::from_pretrained("test_repo", &device).expect("load");
        // test_repo/config.json declares 2 layers (conv, full_attention).
        assert_eq!(model.layers.len(), 2);
    }

    // A.1 — lm_head shape
    #[test]
    fn lm_head_projects_to_vocab() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);

        let hidden: Tensor<TB, 3> = Tensor::zeros([2, 5, config.hidden_size], &device);
        let logits = model.lm_head(hidden);
        assert_eq!(logits.dims(), [2, 5, config.vocab_size]);
    }

    // A.2 — lm_head uses tied embedding weights
    #[test]
    fn lm_head_matches_manual_weight_multiply() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config(); // hidden_size=4, vocab_size=8
        let model = build_synthetic_model(&config, &device, false);

        let hidden: Tensor<TB, 3> = Tensor::ones([1, 2, config.hidden_size], &device);
        let logits = model.lm_head(hidden.clone());

        // Manually reproduce: logits = hidden @ embed_tokens.weight.T
        let w = model.embed_tokens.weight.val(); // [V, H]
        let [b, t, h] = hidden.dims();
        let v = w.dims()[0];
        let expected = hidden
            .reshape([b * t, h])
            .matmul(w.transpose())
            .reshape([b, t, v]);

        assert_eq!(
            logits.into_data().as_slice::<f32>().unwrap(),
            expected.into_data().as_slice::<f32>().unwrap(),
        );
    }

    // A.3 — regression lock: forward returns hidden states, not logits
    #[test]
    fn forward_returns_hidden_states_not_logits() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);
        let ids: Tensor<TB, 2, Int> = Tensor::zeros([2, 5], &device);
        let y = model.forward(ids, None);
        assert_eq!(y.dims(), [2, 5, config.hidden_size]);
    }

    // A.4 — LFMCache::init layer count
    #[test]
    fn cache_empty_layer_count_matches_model() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config(); // 2 layers: conv + full_attention
        let model = build_synthetic_model(&config, &device, false);
        let cache = LFMCache::init(&model, &device);
        assert_eq!(cache.layers.len(), config.num_hidden_layers);
        assert_eq!(cache.position, 0);
    }

    // A.4 — LFMCache::init variant types
    #[test]
    fn cache_empty_variants_match_layer_types() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);
        let cache = LFMCache::init(&model, &device);
        assert!(matches!(cache.layers[0], LayerCache::ConvCache(_)));
        assert!(matches!(cache.layers[1], LayerCache::AttnCache(_)));
    }

    // A.5 — forward advances cache.position
    #[test]
    fn forward_with_cache_advances_position() {
        let device = NdArrayDevice::Cpu;
        let config = hybrid_config();
        let model = build_synthetic_model(&config, &device, false);
        let mut cache = LFMCache::init(&model, &device);
        assert_eq!(cache.position, 0);

        let ids: Tensor<TB, 2, Int> = Tensor::zeros([1, 5], &device);
        let _ = model.forward(ids, Some(&mut cache));
        assert_eq!(cache.position, 5);

        let next: Tensor<TB, 2, Int> = Tensor::zeros([1, 1], &device);
        let _ = model.forward(next, Some(&mut cache));
        assert_eq!(cache.position, 6);
    }
}
