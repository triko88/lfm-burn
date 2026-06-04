use std::{
    fmt,
    cell::RefCell,
};
use std::time::Instant;
use burn::tensor::{
    backend::Backend,
    Int,
    Tensor,
    TensorData,
};
use tokenizers::Tokenizer;

use crate::{
    config::TextModelConfig,
    profiling::{
        bench_gemv, gpu_mem_bytes, peak_cpu_rss_bytes, time_iters, GemvResult, GemvShape,
        LatencyStats, ProfileReport, SteadyStateReport,
    },
    text::{
        LFMCache,
        TextModel,
    },
};

#[derive(Debug)]
pub enum LFMError {
    Config(String),
    Weights(String),
    TokenizerLoad(String),
    Tokenizer(String),
    IO(String),
    Generation(String),
    Join(String),
    UntiedLMHeadUnspported,
}

impl fmt::Display for LFMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(s) => write!(f, "config error: {s}"),
            Self::Weights(s) => write!(f, "weights error: {s}"),
            Self::TokenizerLoad(s) => write!(f, "tokenizer load error: {s}"),
            Self::Tokenizer(s) => write!(f, "tokenizer error: {s}"),
            Self::IO(s) => write!(f, "I/O error: {s}"),
            Self::Generation(s) => write!(f, "generation error: {s}"),
            Self::Join(s) => write!(f, "join error: {s}"),
            Self::UntiedLMHeadUnspported => write!(f, "untied LM head is not supported"),
        }
    }
}

impl std::error::Error for LFMError {}

impl From<std::io::Error> for LFMError {
    fn from(err: std::io::Error) -> Self {
        Self::IO(err.to_string())
    }
}

fn argmax_last_token<Bknd: Backend> (logits: Tensor<Bknd, 3>) -> u32 {
    let [_, t, v] = logits.dims();

    // `iter::<i64>()` converts from the backend's int element type (i64 on
    // NdArray, i32 on Wgpu), so this stays backend-agnostic.
    logits.slice([0..1, (t-1)..t, 0..v])
        .argmax(2).into_data().iter::<i64>()
        .next().expect("argmax must produce a token") as u32
}

#[derive(Clone, Debug)]
pub struct LFMText<Bknd: Backend> {
    model: TextModel<Bknd>,
    cache: RefCell<LFMCache<Bknd>>,
    tokenizer: Tokenizer,
    device: Bknd::Device,
    max_tokens: usize,
    eos_id: Option<u32>,
}

impl <Bknd: Backend> LFMText<Bknd> {
    pub fn from_pretrained(dir: &str, device: &Bknd::Device) -> Result<Self, LFMError> {
        use burn::config::Config;

        let config = TextModelConfig::load(format!("{dir}/config.json"))
            .map_err(|err| LFMError::Config(err.to_string()))?;

        if !config.tie_embedding {
            return Err(LFMError::UntiedLMHeadUnspported);
        }

        let model = TextModel::from_pretrained(dir, device)
            .map_err(|err| LFMError::Weights(err.to_string()))?;

        let tokenizer = Tokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(|err| LFMError::TokenizerLoad(err.to_string()))?;

        let cache: RefCell<LFMCache<Bknd>> = LFMCache::init(&model, device).into();

        // Warm up the model on a throwaway cache so the persistent `cache` stays
        // pristine (position 0, conv windows unseeded) for the first generate call.
        // Forwarding into `cache` itself would seed the conv windows and advance
        // the attention position, making the first real prefill take the decode
        // path and fail with a window/kernel shape mismatch.
        let mut warmup_cache = LFMCache::init(&model, device);
        let _ = model.forward(Tensor::zeros([1, 100], device), Some(&mut warmup_cache));

        Ok(Self {
            model,
            cache,
            tokenizer,
            device: device.clone(),
            max_tokens: 128,
            eos_id: config.eos_token_id,
        })
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    fn apply_chat_template(user: &str) -> String {
        format!("<|startoftext|><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn generate_sync(&self, input: &str) -> Result<String, LFMError> {
        let encoding = self.tokenizer.encode(input, false)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let prefill_len = ids.len();
        let input: Tensor<Bknd, 2, Int> = Tensor::from_data(
                   TensorData::new(ids, [1, prefill_len]), &self.device);

        let hidden = self.model.forward(input, Some(&mut self.cache.borrow_mut()));
        let logits = self.model.lm_head(hidden);
        let mut next_id = argmax_last_token::<Bknd>(logits);

        let mut generated: Vec<u32> = vec![next_id];

        for _ in 1..self.max_tokens {
            if self.eos_id == Some(next_id) {
                break;
            }

            let step: Tensor<Bknd, 2, Int> = Tensor::from_data(
                TensorData::new(vec![next_id as i32], [1, 1]), &self.device);

            let hidden = self.model.forward(step, Some(&mut self.cache.borrow_mut()));
            let logits = self.model.lm_head(hidden);

            next_id = argmax_last_token::<Bknd>(logits);
            generated.push(next_id);
        }

        // If every generated id is a special token, `decode` with
        // skip_special_tokens=true would strip them all and hand the BPE decoder an
        // empty Vec, which underflows (tokens.len() - 1) and panics. Such a sequence
        // legitimately decodes to empty text, so short-circuit it.
        let specials = self.tokenizer.get_added_tokens_decoder();
        let has_text = generated
            .iter()
            .any(|id| specials.get(id).map_or(true, |tok| !tok.special));
        if !has_text {
            return Ok(String::new());
        }

        self.tokenizer.decode(&generated, true)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))
    }

    fn generate_sync_profiled(
        &self,
        input: &str,
    ) -> Result<(String, ProfileReport), LFMError> {
        let start = Instant::now();

        let encoding = self.tokenizer.encode(input, false)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let prefill_len = ids.len();
        let input: Tensor<Bknd, 2, Int> = Tensor::from_data(
                   TensorData::new(ids, [1, prefill_len]), &self.device);

        let mut cache = LFMCache::init(&self.model, &self.device);

        // Prefill: time the forward pass + first argmax, flushing the device
        // before and after so lazy backends (wgpu) are measured at completion.
        Bknd::sync(&self.device).ok();
        let prefill_start = Instant::now();
        let hidden = self.model.forward(input, Some(&mut cache));
        let logits = self.model.lm_head(hidden);
        let mut next_id = argmax_last_token::<Bknd>(logits);
        Bknd::sync(&self.device).ok();
        let prefill_time = prefill_start.elapsed();
        let ttft = start.elapsed();

        let mut generated: Vec<u32> = vec![next_id];

        let decode_start = Instant::now();
        for _ in 1..self.max_tokens {
            if self.eos_id == Some(next_id) {
                break;
            }

            let step: Tensor<Bknd, 2, Int> = Tensor::from_data(
                TensorData::new(vec![next_id as i32], [1, 1]), &self.device);

            let hidden = self.model.forward(step, Some(&mut cache));
            let logits = self.model.lm_head(hidden);

            next_id = argmax_last_token::<Bknd>(logits);
            generated.push(next_id);
        }
        Bknd::sync(&self.device).ok();
        let decode_time = decode_start.elapsed();

        let report = ProfileReport {
            prefill_tokens: prefill_len,
            decode_tokens: generated.len().saturating_sub(1),
            prefill_time,
            decode_time,
            ttft,
            peak_cpu_rss_bytes: peak_cpu_rss_bytes(),
            gpu_mem_bytes: gpu_mem_bytes(),
        };

        let specials = self.tokenizer.get_added_tokens_decoder();
        let has_text = generated
            .iter()
            .any(|id| specials.get(id).map_or(true, |tok| !tok.special));
        if !has_text {
            return Ok((String::new(), report));
        }

        let text = self.tokenizer.decode(&generated, true)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))?;
        Ok((text, report))
    }

    /// The model's `nn::Linear` projection shapes (all GEMVs at `m = 1`), read
    /// from the loaded weights via [`TextModel::linear_shapes`], followed by a
    /// generic square size sweep that maps the matmul latency curve independent
    /// of the model.
    fn gemv_shapes(&self) -> Vec<GemvShape> {
        let mut shapes: Vec<GemvShape> = self.model.linear_shapes()
            .into_iter()
            .map(|(name, k, n)| GemvShape { name, m: 1, k, n })
            .collect();

        for s in [512usize, 1024, 2048, 4096, 8192] {
            shapes.push(GemvShape { name: format!("square_{s}"), m: 1, k: s, n: s });
        }

        shapes
    }

    /// GEMV microbenchmark over the model's projection shapes plus a square
    /// sweep, timing both `nn::Linear` and raw `matmul` (see [`bench_gemv`]).
    pub fn gemv_microbench(&self, warmup: usize, iters: usize) -> Vec<GemvResult> {
        bench_gemv::<Bknd>(&self.device, &self.gemv_shapes(), warmup, iters)
    }

    /// Steady-state per-phase latency: the distribution of a single prefill
    /// forward pass (cold cache each sample) and of a single warm decode step,
    /// each measured `iters` times after `warmup` unmeasured runs.
    ///
    /// Uses a local cache so the persistent `self.cache` is left untouched.
    pub fn steady_state_latency(
        &self,
        input: &str,
        warmup: usize,
        iters: usize,
    ) -> Result<SteadyStateReport, LFMError> {
        let encoding = self.tokenizer.encode(input, false)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let prefill_len = ids.len();

        let sync = || {
            Bknd::sync(&self.device).ok();
        };

        // Prefill: each sample starts from a cold cache, matching a real first
        // turn, and runs the full prompt forward + lm_head.
        let prefill_samples = time_iters(warmup, iters, sync, || {
            let mut cache = LFMCache::init(&self.model, &self.device);
            let input: Tensor<Bknd, 2, Int> = Tensor::from_data(
                TensorData::new(ids.clone(), [1, prefill_len]), &self.device);
            let hidden = self.model.forward(input, Some(&mut cache));
            let _ = self.model.lm_head(hidden);
        });

        // Decode: prefill once into a warm cache, then time single-token steps.
        // Each step appends to the cache, so we feed a fixed token id and let the
        // cache grow; this measures the steady-state per-token decode latency.
        let mut cache = LFMCache::init(&self.model, &self.device);
        let input: Tensor<Bknd, 2, Int> = Tensor::from_data(
            TensorData::new(ids.clone(), [1, prefill_len]), &self.device);
        let hidden = self.model.forward(input, Some(&mut cache));
        let next_id = argmax_last_token::<Bknd>(self.model.lm_head(hidden));

        let decode_samples = time_iters(warmup, iters, sync, || {
            let step: Tensor<Bknd, 2, Int> = Tensor::from_data(
                TensorData::new(vec![next_id as i32], [1, 1]), &self.device);
            let hidden = self.model.forward(step, Some(&mut cache));
            let _ = self.model.lm_head(hidden);
        });

        Ok(SteadyStateReport {
            prefill_tokens: prefill_len,
            prefill: LatencyStats::from_durations(&prefill_samples),
            decode: LatencyStats::from_durations(&decode_samples),
        })
    }

    pub async fn prompt_profiled(&self, input: &str) -> Result<(String, ProfileReport), LFMError>
    where
        Bknd: Send + 'static,
        Bknd::Device: Send + Clone + 'static,
        TextModel<Bknd>: Send + 'static,
        Tokenizer: Send + 'static,
    {
        let input = Self::apply_chat_template(input);
        let this = self.clone();
        let input = input.to_owned();

        tokio::task::spawn_blocking(move || this.generate_sync_profiled(&input))
            .await.map_err(|err| LFMError::Join(err.to_string()))?
    }

    pub async fn prompt(&self, input: &str) -> Result<String, LFMError>
    where
        Bknd: Send + 'static,
        Bknd::Device: Send + Clone + 'static,
        TextModel<Bknd>: Send + 'static,
        Tokenizer: Send + 'static,
    {
        let input = Self::apply_chat_template(input);
        let this = self.clone();
        let input = input.to_owned();

        tokio::task::spawn_blocking(move || this.generate_sync(&input))
            .await.map_err(|err| LFMError::Join(err.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type TB = NdArray;

    fn load() -> LFMText<TB> {
        let device = NdArrayDevice::Cpu;
        LFMText::<TB>::from_pretrained("test_repo", &device).expect("load test_repo")
    }

    // GEMV shapes are read from the loaded model weights (all m == 1) and
    // include the generic square sweep.
    #[test]
    fn gemv_shapes_from_model_weights() {
        let lfm = load();
        let shapes = lfm.gemv_shapes();

        assert!(shapes.iter().all(|s| s.m == 1), "GEMV shapes must have m == 1");

        // 8 model-sourced (4 attn + 3 mlp + lm_head) + 5 square-sweep entries.
        assert_eq!(shapes.len(), 13);

        let find = |name: &str| shapes.iter().find(|s| s.name == name)
            .unwrap_or_else(|| panic!("missing shape {name}"));

        // test_repo: hidden_size = 4, vocab_size = 8.
        assert_eq!((find("lm_head").k, find("lm_head").n), (4, 8));
        assert_eq!(find("q_proj").k, 4);   // hidden
        assert_eq!(find("out_proj").n, 4); // hidden
        assert_eq!(find("mlp.w1").k, 4);   // hidden
        // intermediate_size comes from the actual weights (the safetensors
        // header can override config.json), so assert internal consistency
        // rather than a hardcoded value.
        assert_eq!(find("mlp.w1").n, find("mlp.w2").k);

        for s in [512usize, 1024, 2048, 4096, 8192] {
            let sq = find(&format!("square_{s}"));
            assert_eq!((sq.k, sq.n), (s, s));
        }
    }

    // [RED PHASE] steady_state_latency yields the requested number of samples
    // for both phases.
    #[test]
    fn steady_state_reports_requested_iters() {
        let lfm = load();
        let report = lfm.steady_state_latency("The capital of France is", 1, 3)
            .expect("steady-state run");

        assert_eq!(report.prefill.n, 3);
        assert_eq!(report.decode.n, 3);
        assert!(report.prefill_tokens > 0);
    }
}
