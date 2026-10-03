//! sentrygate-core — the torch-free Laya guardrail, in Rust.
//!
//! A decision model (not a generative LLM) scores targeted binary questions about a
//! piece of text; the max score over an ensemble is the injection/jailbreak signal.
//! Tokenization uses the same HF `tokenizers` library Python wraps, and inference runs
//! through ONNX Runtime, so scores match the Python and TypeScript packages exactly.
//!
//! This is the shared backbone: `Scanner` here is what the PyO3 and (later) Node/wasm
//! bindings call, so there is one implementation of the hot path to trust.

use std::path::Path;
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

mod presets;
pub use presets::{INJECTION_QUESTIONS, INJECTION_THRESHOLD};

#[cfg(feature = "python")]
mod py;

#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("onnx runtime error: {0}")]
    Ort(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

// Fixed for the convaiinnovations/laya checkpoint (ModernBERT-large).
const CLS: i64 = 50281;
const SEP: i64 = 50282;
const MASK: i64 = 50284;
const PAD: i64 = 50283;
const MAX_LEN: usize = 512;
const HEAD_MAX_LEN: usize = 192;
// temperature_by_options["noul:2"] from the checkpoint config (a noul question has 2 options).
const NOUL_TEMP: f32 = 1.983_399_5;

/// One detection outcome, mirroring the Python/TS `Finding`.
#[derive(Debug, Clone)]
pub struct Finding {
    pub is_attack: bool,
    pub score: f32,
    pub threshold: f32,
    pub top_question: String,
    pub per_question: Vec<(String, f32)>,
    pub preview: String,
}

/// Loads the model once; scans text against an ensemble of binary questions.
pub struct Scanner {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    questions: Vec<String>,
    threshold: f32,
}

impl Scanner {
    /// `tokenizer_json` is the path to tokenizer.json; `onnx_path` is laya.onnx.
    pub fn new(
        tokenizer_json: impl AsRef<Path>,
        onnx_path: impl AsRef<Path>,
        questions: Vec<String>,
        threshold: f32,
    ) -> Result<Self, GuardError> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_json).map_err(|e| GuardError::Tokenizer(e.to_string()))?;
        let session = Session::builder()
            .map_err(|e| GuardError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GuardError::Ort(e.to_string()))?
            .commit_from_file(onnx_path)
            .map_err(|e| GuardError::Ort(e.to_string()))?;
        Ok(Self { tokenizer, session: Mutex::new(session), questions, threshold })
    }

    /// The ready-made injection/jailbreak guardrail (4 questions, MAX, threshold 0.20).
    pub fn injection(
        tokenizer_json: impl AsRef<Path>,
        onnx_path: impl AsRef<Path>,
    ) -> Result<Self, GuardError> {
        Self::new(
            tokenizer_json,
            onnx_path,
            INJECTION_QUESTIONS.iter().map(|s| s.to_string()).collect(),
            INJECTION_THRESHOLD,
        )
    }

    fn encode(&self, text: &str, max: Option<usize>) -> Result<Vec<i64>, GuardError> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| GuardError::Tokenizer(e.to_string()))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        if let Some(m) = max {
            ids.truncate(m);
        }
        Ok(ids)
    }

    // Port of laya.common.build_sequence for a noul question:
    //   [CLS] noul question: <ins> [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
    // noul options for criteria {true:"yes", false:"no"} render as ["false: no", "true: yes"].
    fn build_sequence(&self, state: &str, instructions: &str) -> Result<(Vec<i64>, Vec<usize>), GuardError> {
        let opts = ["false: no", "true: yes"];
        let mut head = self.encode(&format!("noul question: {instructions}"), None)?;
        let mut opt_ids: Vec<Vec<i64>> = Vec::with_capacity(2);
        for o in opts {
            let mut v = vec![MASK];
            v.extend(self.encode(&format!(" {o}"), Some(48))?);
            opt_ids.push(v);
        }

        let opt_total: usize = opt_ids.iter().map(|o| o.len()).sum();
        let mut opt_budget = HEAD_MAX_LEN as isize - opt_total as isize;
        if opt_budget < 16 {
            let per = std::cmp::max(4, (HEAD_MAX_LEN - 16) / std::cmp::max(1, opt_ids.len()));
            for o in opt_ids.iter_mut() {
                o.truncate(per);
            }
            let t: usize = opt_ids.iter().map(|o| o.len()).sum();
            opt_budget = HEAD_MAX_LEN as isize - t as isize;
        }
        head.truncate(std::cmp::max(8, opt_budget.max(0) as usize));

        let mut ids: Vec<i64> = Vec::with_capacity(MAX_LEN);
        ids.push(CLS);
        ids.extend(&head);
        ids.push(SEP);
        let mut markers = Vec::with_capacity(2);
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(SEP);

        let room = MAX_LEN.saturating_sub(ids.len() + 1);
        let state_ids = self.encode(state, None)?;
        let take = room.min(state_ids.len());
        ids.extend(&state_ids[..take]);
        ids.push(SEP);

        ids.truncate(MAX_LEN);
        let markers: Vec<usize> = markers.into_iter().filter(|&m| m < MAX_LEN).collect();
        Ok((ids, markers))
    }

    /// P(true) for each question against one state, in a single session run.
    pub fn score(&self, text: &str) -> Result<Vec<f32>, GuardError> {
        let rows: Vec<(Vec<i64>, Vec<usize>)> = self
            .questions
            .iter()
            .map(|q| self.build_sequence(text, q))
            .collect::<Result<_, _>>()?;

        let b = rows.len();
        let m = 2usize;
        let seq_len = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);

        let mut input_ids = vec![PAD; b * seq_len];
        let mut attn = vec![0i64; b * seq_len];
        let mut mpos = vec![0i64; b * m];
        let mmask = vec![true; b * m];
        let qtype = vec![2i64; b];

        for (bi, (ids, markers)) in rows.iter().enumerate() {
            for (i, &tok) in ids.iter().enumerate() {
                input_ids[bi * seq_len + i] = tok;
                attn[bi * seq_len + i] = 1;
            }
            for j in 0..m {
                mpos[bi * m + j] = *markers.get(j).unwrap_or(&0) as i64;
            }
        }

        let oe = |e: ort::Error| GuardError::Ort(e.to_string());
        let bb = b as i64;
        let ss = seq_len as i64;
        let mm = m as i64;
        let t_ids = Tensor::from_array((vec![bb, ss], input_ids)).map_err(oe)?;
        let t_attn = Tensor::from_array((vec![bb, ss], attn)).map_err(oe)?;
        let t_mpos = Tensor::from_array((vec![bb, mm], mpos)).map_err(oe)?;
        let t_mmask = Tensor::from_array((vec![bb, mm], mmask)).map_err(oe)?;
        let t_qtype = Tensor::from_array((vec![bb], qtype)).map_err(oe)?;

        let mut session = self.session.lock().expect("scanner session mutex poisoned");
        let outputs = session
            .run(ort::inputs![
                "input_ids" => t_ids,
                "attention_mask" => t_attn,
                "marker_pos" => t_mpos,
                "marker_mask" => t_mmask,
                "qtype" => t_qtype,
            ])
            .map_err(oe)?;

        let (_shape, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(oe)?;

        let mut scores = Vec::with_capacity(b);
        for bi in 0..b {
            let z0 = logits[bi * m] / NOUL_TEMP;
            let z1 = logits[bi * m + 1] / NOUL_TEMP;
            let mx = z0.max(z1);
            let e0 = (z0 - mx).exp();
            let e1 = (z1 - mx).exp();
            scores.push(e1 / (e0 + e1)); // P(true) = p[1]
        }
        Ok(scores)
    }

    /// Scan one text into a `Finding`.
    pub fn scan(&self, text: &str) -> Result<Finding, GuardError> {
        let scores = self.score(text)?;
        let top = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let round = |x: f32| (x * 1000.0).round() / 1000.0;
        Ok(Finding {
            is_attack: scores[top] >= self.threshold,
            score: round(scores[top]),
            threshold: self.threshold,
            top_question: self.questions[top].clone(),
            per_question: self
                .questions
                .iter()
                .cloned()
                .zip(scores.iter().map(|&s| round(s)))
                .collect(),
            preview: text.chars().take(120).collect(),
        })
    }
}
