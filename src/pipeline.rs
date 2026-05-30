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
    profiling::{gpu_mem_bytes, peak_cpu_rss_bytes, ProfileReport},
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

        let cache = LFMCache::init(&model, device).into();

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
