//! Building the sequence a decision model reads.
//!
//! The format is the checkpoint's, not ours:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP]
//! [MASK] <option 0> [MASK] <option 1> ... [SEP]
//! <state> [SEP]
//! ```
//!
//! Each option is scored at its own `[MASK]`, so the positions of those
//! markers are the output contract and are returned alongside the ids. The
//! head — the question and its options — gets a fixed share of the sequence
//! and the state gets what is left, which is how a long document cannot push
//! the options out of the window.
//!
//! This lives on the gateway for the same reasons the chat template does: an
//! unanswerable request is rejected before any queue work, and the queued job
//! is self-describing.

use serde_json::Value;
use tokenizers::Tokenizer;
use vapi_core::{DecisionConfig, Error, QuestionType, Result};

use crate::TokenizerBundle;

/// The structural tokens an encoder's sequence is assembled from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpecialTokens {
    pub cls: Option<u32>,
    pub sep: Option<u32>,
    pub mask: Option<u32>,
    pub pad: Option<u32>,
    /// The mask token's text, needed to strip it out of user input.
    pub mask_text: Option<String>,
}

impl SpecialTokens {
    /// Resolve from `tokenizer_config.json`, falling back to the conventional
    /// spellings for a repo that does not name them.
    pub fn from_config(config: &Value, tokenizer: &Tokenizer) -> Self {
        let named = |key: &str, fallback: &str| -> Option<(String, u32)> {
            let text = config
                .get(key)
                .and_then(token_text)
                .unwrap_or_else(|| fallback.to_string());
            tokenizer.token_to_id(&text).map(|id| (text, id))
        };
        let mask = named("mask_token", "[MASK]");
        Self {
            cls: named("cls_token", "[CLS]").map(|(_, id)| id),
            sep: named("sep_token", "[SEP]").map(|(_, id)| id),
            mask: mask.as_ref().map(|(_, id)| *id),
            pad: named("pad_token", "[PAD]").map(|(_, id)| id),
            mask_text: mask.map(|(t, _)| t),
        }
    }
}

fn token_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// Longest an option's own text may be, before its marker.
const OPTION_TOKEN_CAP: usize = 48;
/// Below this much room left for the question itself, every option is shrunk.
const MIN_OPTION_BUDGET: usize = 16;
/// The question text never shrinks below this.
const MIN_HEAD_TOKENS: usize = 8;
/// The floor the checkpoint's own builder shrinks options to. Below this a
/// request cannot be built at all, and is refused.
const MIN_TOKENS_PER_OPTION: usize = 4;
/// Options cut below this stop being distinguishable from one another and
/// accuracy falls off a cliff — measured at 0.425 on a 77-option benchmark
/// against 0.870 for a model that gives every label room. Still answerable,
/// so it is reported rather than refused: see
/// [`BuiltSequence::option_tokens`].
pub const OPTIONS_ARE_CRAMPED_BELOW: usize = 8;

/// A built sequence and where its option markers landed.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltSequence {
    pub tokens: Vec<u32>,
    pub markers: Vec<u32>,
    /// Tokens each option was cut to, when they did not all fit. `None` means
    /// every option is intact.
    pub option_tokens: Option<usize>,
}

impl BuiltSequence {
    /// Whether the options were cut so far that the model can no longer tell
    /// them apart reliably. The caller decides what to do about it; what it
    /// must not do is stay quiet.
    pub fn options_are_cramped(&self) -> bool {
        self.option_tokens
            .is_some_and(|per| per < OPTIONS_ARE_CRAMPED_BELOW)
    }
}

/// Assembles sequences for one decision checkpoint.
#[derive(Clone, Debug)]
pub struct DecisionFormat {
    cls: u32,
    sep: u32,
    mask: u32,
    pub pad: u32,
    mask_text: String,
    pub config: DecisionConfig,
    /// Keep the *end* of an over-long state rather than the start. A support
    /// ticket's latest message is usually the one being judged.
    pub truncate_left: bool,
    /// Whether this checkpoint can read scripts other than Latin.
    ///
    /// A heuristic on the encoder's vocabulary size, because nothing in the
    /// checkpoint states it: the English encoder has 50,368 tokens, the
    /// multilingual one 256,000. Wrong only if someone ships an English model
    /// with a huge vocabulary, and the cost of that is a missing warning.
    pub multilingual: bool,
}

impl DecisionFormat {
    /// Fails when the vocabulary lacks the structural tokens — a decoder-only
    /// repo, loaded by mistake, rather than a decision checkpoint.
    pub fn new(special: &SpecialTokens, config: DecisionConfig) -> Result<Self> {
        let need = |id: Option<u32>, name: &str| -> Result<u32> {
            id.ok_or_else(|| {
                Error::Tokenizer(format!(
                    "this tokenizer has no {name} token, so it cannot be a decision model's"
                ))
            })
        };
        Ok(Self {
            cls: need(special.cls, "[CLS]")?,
            sep: need(special.sep, "[SEP]")?,
            mask: need(special.mask, "[MASK]")?,
            pad: special.pad.unwrap_or(0),
            mask_text: special
                .mask_text
                .clone()
                .unwrap_or_else(|| "[MASK]".to_string()),
            config,
            truncate_left: false,
            multilingual: false,
        })
    }

    /// Strip the mask token out of text that came from a request.
    ///
    /// Without this a state containing `[MASK]` mints an extra marker: the
    /// scorer would read a position nobody assigned an option to, and the
    /// answer would be drawn from the caller's own document. The checkpoint's
    /// builder does the same thing, and it is the one place in this file
    /// where the reason is security rather than fidelity.
    fn scrub(&self, text: &str) -> String {
        if text.contains(&self.mask_text) {
            text.replace(&self.mask_text, " ")
        } else {
            text.to_string()
        }
    }

    /// Build one question's sequence.
    ///
    /// `options` are the rendered option texts, in the order their
    /// probabilities will be reported.
    pub fn build(
        &self,
        tokenizer: &TokenizerBundle,
        qtype: QuestionType,
        instructions: &str,
        options: &[String],
        state: &str,
    ) -> Result<BuiltSequence> {
        if options.len() < 2 {
            return Err(Error::InvalidRequest(
                "a question needs at least two options".into(),
            ));
        }
        let max_len = self.config.max_len;
        let head_max = self.config.head_max_len.min(max_len);

        let question = format!("{qtype} question: {}", self.scrub(instructions));
        let mut head = tokenizer.encode(&question, false)?;

        let mut option_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
        for opt in options {
            // The leading space matters: this vocabulary is byte-level, so
            // "billing" and " billing" are different tokens, and the trained
            // format has the space.
            let mut ids = tokenizer.encode(&format!(" {}", self.scrub(opt)), false)?;
            ids.truncate(OPTION_TOKEN_CAP);
            let mut row = Vec::with_capacity(ids.len() + 1);
            row.push(self.mask);
            row.extend(ids);
            option_ids.push(row);
        }

        let used: usize = option_ids.iter().map(Vec::len).sum();
        let mut option_tokens = None;
        if head_max.saturating_sub(used) < MIN_OPTION_BUDGET {
            // Every option shrinks by the same amount: a request whose
            // options are all long should not have the last one erased.
            let room = head_max.saturating_sub(MIN_OPTION_BUDGET) / option_ids.len();
            if room < MIN_TOKENS_PER_OPTION {
                return Err(Error::InvalidRequest(format!(
                    "{} options leave {room} tokens each within head_max_len={head_max}, \
                     below the {MIN_TOKENS_PER_OPTION} an option needs to exist at all; \
                     raise head_max_len, shorten the option texts, or split the question \
                     into a coarse choice followed by a fine one",
                    option_ids.len()
                )));
            }
            for row in &mut option_ids {
                row.truncate(room);
            }
            option_tokens = Some(room);
        }
        let option_budget = head_max.saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        head.truncate(option_budget.max(MIN_HEAD_TOKENS));

        let mut tokens = Vec::with_capacity(max_len);
        tokens.push(self.cls);
        tokens.extend(head);
        tokens.push(self.sep);
        let mut markers = Vec::with_capacity(option_ids.len());
        for row in &option_ids {
            markers.push(tokens.len() as u32);
            tokens.extend(row);
        }
        tokens.push(self.sep);

        // One slot held back for the closing [SEP].
        let room = max_len.saturating_sub(tokens.len() + 1);
        let state_ids = tokenizer.encode(&self.scrub(state), false)?;
        let kept = if state_ids.len() <= room {
            &state_ids[..]
        } else if self.truncate_left {
            &state_ids[state_ids.len() - room..]
        } else {
            &state_ids[..room]
        };
        tokens.extend_from_slice(kept);
        tokens.push(self.sep);
        tokens.truncate(max_len);

        if markers.iter().any(|&m| m as usize >= tokens.len()) {
            // Only reachable if head_max_len exceeds max_len, which `new`
            // cannot see. Better a clear error than markers pointing at
            // whatever the truncation left.
            return Err(Error::InvalidRequest(format!(
                "the question and its options do not fit in max_len={max_len}"
            )));
        }
        Ok(BuiltSequence {
            tokens,
            markers,
            option_tokens,
        })
    }
}

/// Open a decision checkpoint's tokenizer and format.
///
/// The layout is the published one: `rl_agent_config.json` beside the
/// weights, and the tokenizer in a `tokenizer/` subdirectory — a subfolder
/// checkpoint in the same repo has its own copy of both.
pub fn open_decision_model(
    dir: impl AsRef<std::path::Path>,
) -> Result<(TokenizerBundle, DecisionFormat)> {
    let dir = dir.as_ref();
    let tok_dir = if dir.join("tokenizer/tokenizer.json").exists() {
        dir.join("tokenizer")
    } else {
        dir.to_path_buf()
    };
    let bundle = TokenizerBundle::from_dir(&tok_dir)?;
    let cfg_path = dir.join("rl_agent_config.json");
    let config = match std::fs::read_to_string(&cfg_path) {
        Ok(text) => DecisionConfig::from_json(&text)?,
        // A checkpoint with no config is not an error: the defaults are the
        // English root's own numbers. Say so in the logs rather than here.
        Err(_) => DecisionConfig::default(),
    };
    let mut format = DecisionFormat::new(&bundle.special, config)?;
    format.multilingual = std::fs::read_to_string(dir.join("encoder/config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|v| v.get("vocab_size").and_then(serde_json::Value::as_u64))
        .is_some_and(|v| v >= 100_000);
    Ok((bundle, format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    /// A word-level tokenizer over a fixed vocabulary, so a test can read the
    /// sequence back as words.
    fn bundle() -> TokenizerBundle {
        let words = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PAD]",
            "choice",
            "score",
            "noul",
            "question",
            ":",
            "which",
            "department",
            "?",
            "billing",
            "invoices",
            "refunds",
            "technical",
            "bugs",
            "sales",
            "other",
            "level",
            "0",
            "1",
            "2",
            "not",
            "urgent",
            "soon",
            "critical",
            "the",
            "user",
            "was",
            "billed",
            "twice",
            "and",
            "wants",
            "a",
            "refund",
            "today",
            "false",
            "true",
            "yes",
            "no",
        ];
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        TokenizerBundle::from_parts(tokenizer, None, vec![2])
    }

    fn format_with(max_len: usize, head_max_len: usize) -> (TokenizerBundle, DecisionFormat) {
        let b = bundle();
        let f = DecisionFormat::new(
            &b.special,
            DecisionConfig {
                max_len,
                head_max_len,
                ..Default::default()
            },
        )
        .unwrap();
        (b, f)
    }

    fn words(b: &TokenizerBundle, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&i| b.decode(&[i], false).unwrap_or_default())
            .collect()
    }

    #[test]
    fn the_sequence_has_the_shape_the_checkpoint_was_trained_on() {
        let (b, f) = format_with(64, 32);
        let out = f
            .build(
                &b,
                QuestionType::Choice,
                "which department ?",
                &["billing : invoices refunds".into(), "other".into()],
                "the user was billed twice",
            )
            .unwrap();
        let w = words(&b, &out.tokens);
        assert_eq!(w[0], "[CLS]");
        assert_eq!(w[1..4], ["choice", "question", ":"]);
        assert_eq!(*w.last().unwrap(), "[SEP]");
        // A marker sits on each option, and on nothing else.
        assert_eq!(out.markers.len(), 2);
        for &m in &out.markers {
            assert_eq!(w[m as usize], "[MASK]");
        }
        assert_eq!(
            w.iter().filter(|x| *x == "[MASK]").count(),
            2,
            "a stray marker would be scored as an option: {w:?}"
        );
        // The state follows the options, after a separator.
        let text = w.join(" ");
        assert!(text.contains("billing"), "{text}");
        assert!(text.contains("the user was billed twice"), "{text}");
    }

    #[test]
    fn a_mask_token_in_the_state_cannot_mint_a_marker() {
        // Otherwise a caller's document chooses where the answer is read
        // from, which is a confidently wrong answer rather than an error.
        let (b, f) = format_with(64, 32);
        let out = f
            .build(
                &b,
                QuestionType::Noul,
                "is the [MASK] happy ?",
                &["false : no".into(), "true : yes".into()],
                "the user [MASK] was [MASK] billed",
            )
            .unwrap();
        let masks = out
            .tokens
            .iter()
            .filter(|&&t| t == b.special.mask.unwrap())
            .count();
        assert_eq!(masks, 2, "{:?}", words(&b, &out.tokens));
        assert_eq!(out.markers.len(), 2);
    }

    #[test]
    fn a_long_state_is_cut_and_the_options_survive() {
        // The failure this prevents: a long document pushing the options out
        // of the window, leaving markers pointing past the end.
        let (b, f) = format_with(48, 24);
        let state = "the user was billed twice and wants a refund today ".repeat(40);
        let out = f
            .build(
                &b,
                QuestionType::Choice,
                "which department ?",
                &["billing".into(), "technical".into(), "sales".into()],
                &state,
            )
            .unwrap();
        assert_eq!(out.tokens.len(), 48);
        assert_eq!(out.markers.len(), 3);
        assert!(out.markers.iter().all(|&m| (m as usize) < out.tokens.len()));
        assert_eq!(*out.tokens.last().unwrap(), b.special.sep.unwrap());
    }

    #[test]
    fn keeping_the_tail_is_a_choice_the_caller_makes() {
        let (b, mut f) = format_with(40, 20);
        let head = "billing ".repeat(30);
        let tail = "critical";
        let state = format!("{head}{tail}");
        let opts = ["billing".to_string(), "other".to_string()];
        let front = f
            .build(&b, QuestionType::Choice, "which ?", &opts, &state)
            .unwrap();
        f.truncate_left = true;
        let back = f
            .build(&b, QuestionType::Choice, "which ?", &opts, &state)
            .unwrap();
        let ends_with_tail = |s: &BuiltSequence| words(&b, &s.tokens).contains(&tail.to_string());
        assert!(!ends_with_tail(&front));
        assert!(ends_with_tail(&back));
    }

    #[test]
    fn many_long_options_shrink_evenly_rather_than_dropping_the_last() {
        let (b, f) = format_with(256, 64);
        let opts: Vec<String> = (0..6)
            .map(|_| "billing invoices refunds technical bugs sales other".to_string())
            .collect();
        let out = f
            .build(&b, QuestionType::Choice, "which ?", &opts, "the user")
            .unwrap();
        assert_eq!(out.markers.len(), 6);
        let spans: Vec<u32> = out.markers.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            spans.windows(2).all(|w| w[0] == w[1]),
            "options were cut unevenly: {spans:?}"
        );
    }

    #[test]
    fn cramped_options_are_answered_but_reported() {
        // 0.425 accuracy against 0.870 for a model that gives every label
        // room. Answering is still the caller's right; saying nothing about
        // it is not ours.
        let (b, f) = format_with(512, 100);
        let opts: Vec<String> = (0..20)
            .map(|_| "billing invoices refunds technical bugs".to_string())
            .collect();
        let out = f
            .build(&b, QuestionType::Choice, "which ?", &opts, "the user")
            .unwrap();
        assert_eq!(out.markers.len(), 20);
        assert_eq!(out.option_tokens, Some(4));
        assert!(out.options_are_cramped());
    }

    #[test]
    fn options_that_cannot_exist_at_all_are_refused() {
        let (b, f) = format_with(512, 192);
        let opts: Vec<String> = (0..77).map(|i| format!("label {i} billing")).collect();
        let err = f
            .build(&b, QuestionType::Choice, "which ?", &opts, "the user")
            .unwrap_err()
            .to_string();
        assert!(err.contains("77 options"), "{err}");
        assert!(err.contains("head_max_len"), "{err}");
        assert!(err.contains("split the question"), "{err}");
    }

    #[test]
    fn short_options_are_not_refused_merely_for_being_many() {
        // 30 one-word labels fit comfortably; only truncation is the problem.
        let (b, f) = format_with(512, 192);
        let opts: Vec<String> = (0..30).map(|_| "billing".to_string()).collect();
        let out = f
            .build(&b, QuestionType::Choice, "which ?", &opts, "the user")
            .unwrap();
        assert_eq!(out.markers.len(), 30);
        assert_eq!(out.option_tokens, None, "nothing was cut");
        assert!(!out.options_are_cramped());
    }

    #[test]
    fn the_question_text_yields_before_the_options_do() {
        // The options are the answer space; the instructions are context.
        let (b, f) = format_with(128, 32);
        let long_question = "which department ".repeat(20);
        let out = f
            .build(
                &b,
                QuestionType::Choice,
                &long_question,
                &["billing".into(), "technical".into(), "sales".into()],
                "the user",
            )
            .unwrap();
        assert_eq!(out.markers.len(), 3);
        assert!(out.markers[0] as usize <= 32, "{:?}", out.markers);
    }

    #[test]
    fn a_decoder_only_tokenizer_is_rejected_at_load() {
        let special = SpecialTokens::default();
        let err = DecisionFormat::new(&special, DecisionConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("[CLS]"), "{err}");
    }

    #[test]
    fn the_type_word_is_the_one_the_model_was_trained_with() {
        let (b, f) = format_with(64, 32);
        for (t, word) in [
            (QuestionType::Choice, "choice"),
            (QuestionType::Score, "score"),
            (QuestionType::Noul, "noul"),
        ] {
            let out = f
                .build(&b, t, "which ?", &["billing".into(), "other".into()], "x")
                .unwrap();
            assert_eq!(words(&b, &out.tokens)[1], word);
        }
    }
}
