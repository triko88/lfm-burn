use std::fmt;
use burn::tensor::{
    backend::Backend,
    Int,
    Tensor,
    TensorData,
};
use tokenizers::Tokenizer;

use crate::{
    config::TextModelConfig,
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

    logits.slice([0..1, (t-1)..t, 0..v])
        .argmax(2).into_data().as_slice::<i32>()
        .expect("argmax must produce i32")[0] as u32
}

#[derive(Clone, Debug)]
pub struct LFMText<Bknd: Backend> {
    model: TextModel<Bknd>,
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

        Ok(Self {
            model,
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

    fn generate_sync(&self, input: &str) -> Result<String, LFMError> {
        let encoding = self.tokenizer.encode(input, false)
            .map_err(|err| LFMError::Tokenizer(err.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let prefill_len = ids.len();
        let input: Tensor<Bknd, 2, Int> = Tensor::from_data(
                   TensorData::new(ids, [1, prefill_len]), &self.device);

        let mut cache = LFMCache::init(&self.model, &self.device);

        let hidden = self.model.forward(input, Some(&mut cache));
        let logits = self.model.lm_head(hidden);
        let mut next_id = argmax_last_token::<Bknd>(logits);

        let mut generated: Vec<u32> = vec![next_id];

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

    pub async fn prompt(&self, input: &str) -> Result<String, LFMError> 
    where
        Bknd: Send + 'static,
        Bknd::Device: Send + Clone + 'static,
        TextModel<Bknd>: Send + 'static,
        Tokenizer: Send + 'static,
    {
        let this = self.clone();
        let input = input.to_owned();

        tokio::task::spawn_blocking(move || this.generate_sync(&input))
            .await.map_err(|err| LFMError::Join(err.to_string()))?
    }
}
