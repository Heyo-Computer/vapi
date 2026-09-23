//! The vocabulary of a single-pass decision request.
//!
//! A decision model is handed a *state* and a set of typed *questions*, and
//! answers all of them in one forward pass. Nothing here knows about tensors
//! or transport; it is the shared spelling of the three question primitives,
//! which the gateway, the worker and the model all have to agree on.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The three primitives a decision model answers.
///
/// The discriminants are not ours to choose: the model carries a trained
/// embedding indexed by exactly this order, so renumbering them would answer
/// every question as though it were a different type. [`QuestionType::index`]
/// is pinned by a test for that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    /// Pick one of a set of named options.
    Choice,
    /// An ordinal level, reported as the expectation over the levels.
    Score,
    /// A yes/no question, reported as the probability of "yes".
    Noul,
}

impl QuestionType {
    /// Index into the model's type embedding. Fixed by the checkpoint.
    pub fn index(self) -> u32 {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    pub fn from_index(i: u32) -> Option<Self> {
        match i {
            0 => Some(Self::Choice),
            1 => Some(Self::Score),
            2 => Some(Self::Noul),
            _ => None,
        }
    }

    /// Key into the calibration table.
    ///
    /// Temperature is fitted per (type, option count) because a two-option
    /// yes/no and a twenty-option choice are differently overconfident, and
    /// one scalar for both leaves the smaller one badly calibrated. The
    /// bucket edges match the fitted table shipped with the checkpoint.
    pub fn temperature_bucket(self, options: usize) -> String {
        let size = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        format!("{}:{}", self.as_str(), size)
    }
}

impl std::fmt::Display for QuestionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything about a decision checkpoint that is not weights.
///
/// Read from the model directory next to the tensors, because the budgets and
/// the temperatures were fitted against these weights: pairing one
/// checkpoint's weights with another's calibration produces confident
/// nonsense, and nothing downstream would notice.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionConfig {
    /// Longest sequence, in tokens.
    pub max_len: usize,
    /// Of that, the share the question and its options may spend.
    pub head_max_len: usize,
    pub calibration: Calibration,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            max_len: 512,
            head_max_len: 192,
            calibration: Calibration::default(),
        }
    }
}

impl DecisionConfig {
    /// Parse the checkpoint's own config, keeping the defaults for anything
    /// it does not mention.
    pub fn from_json(text: &str) -> crate::Result<Self> {
        #[derive(Deserialize)]
        struct Raw {
            max_len: Option<usize>,
            head_max_len: Option<usize>,
            temperature: Option<Vec<f32>>,
            temperature_by_options: Option<HashMap<String, f32>>,
        }
        let raw: Raw = serde_json::from_str(text)
            .map_err(|e| crate::Error::InvalidRequest(format!("decision config: {e}")))?;
        let d = Self::default();
        let mut per_type = d.calibration.per_type;
        if let Some(t) = raw.temperature {
            for (slot, v) in per_type.iter_mut().zip(t) {
                *slot = v;
            }
        }
        Ok(Self {
            max_len: raw.max_len.unwrap_or(d.max_len),
            head_max_len: raw.head_max_len.unwrap_or(d.head_max_len),
            calibration: Calibration {
                per_type,
                by_bucket: raw.temperature_by_options.unwrap_or_default(),
            },
        })
    }

    /// How much of the sequence is left for the state.
    pub fn state_budget(&self) -> usize {
        self.max_len.saturating_sub(self.head_max_len)
    }
}

/// The temperatures that turn raw marker logits into honest probabilities.
///
/// These are fitted, not learned: the checkpoint ships overconfident (mean ECE
/// 0.466 on the English weights) and refitting one temperature per (type,
/// option count) moves it to 0.081. They are configuration for that reason —
/// a deployment that cares about its probabilities refits them on its own
/// data, and a table baked into the code could not be refitted at all.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// One per [`QuestionType`], indexed by [`QuestionType::index`].
    pub per_type: [f32; 3],
    /// The finer table, keyed by [`QuestionType::temperature_bucket`]. Takes
    /// precedence where it has an entry.
    pub by_bucket: HashMap<String, f32>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            per_type: [1.0; 3],
            by_bucket: HashMap::new(),
        }
    }
}

impl Calibration {
    /// The divisor for a question of this type with this many options.
    pub fn temperature(&self, qtype: QuestionType, options: usize) -> f32 {
        self.by_bucket
            .get(&qtype.temperature_bucket(options))
            .copied()
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or_else(|| {
                let t = self.per_type[qtype.index() as usize];
                if t.is_finite() && t > 0.0 { t } else { 1.0 }
            })
    }
}

/// Temperature-scaled softmax over one question's markers.
pub fn calibrated_probabilities(logits: &[f32], temperature: f32) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = if temperature.is_finite() && temperature > 0.0 {
        temperature as f64
    } else {
        1.0
    };
    let z: Vec<f64> = logits.iter().map(|&l| l as f64 / t).collect();
    // Subtract the max before exponentiating: the head can emit logits in the
    // tens, and the masked-out markers are -1e4, which overflows to inf.
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<f64> = z.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = p.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for x in &mut p {
            *x /= sum;
        }
    } else {
        let uniform = 1.0 / p.len() as f64;
        p.iter_mut().for_each(|x| *x = uniform);
    }
    p
}

/// One minus the normalized entropy of an answer: 1.0 when the model is
/// certain, 0.0 when it spreads evenly over the options.
///
/// Normalized so that a two-option and a twenty-option question are on one
/// scale, which is what makes a single confidence threshold meaningful across
/// a mixed request.
pub fn confidence(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let ent: f64 = -p.iter().map(|&x| x * x.clamp(1e-12, 1.0).ln()).sum::<f64>();
    (1.0 - ent / (p.len() as f64).ln()).clamp(0.0, 1.0)
}

/// The expectation over ordinal levels — a score question's answer.
pub fn expected_level(p: &[f64]) -> f64 {
    p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_indices_match_the_trained_embedding() {
        // The checkpoint's QTYPES. A change here is a silently wrong answer,
        // not a crash: the model would read the wrong row of `type_emb`.
        assert_eq!(QuestionType::Choice.index(), 0);
        assert_eq!(QuestionType::Score.index(), 1);
        assert_eq!(QuestionType::Noul.index(), 2);
        for t in [
            QuestionType::Choice,
            QuestionType::Score,
            QuestionType::Noul,
        ] {
            assert_eq!(QuestionType::from_index(t.index()), Some(t));
        }
    }

    #[test]
    fn temperature_buckets_match_the_fitted_table() {
        // These keys index a table fitted elsewhere; a mismatch falls back to
        // the per-type scalar and quietly loses the calibration.
        assert_eq!(QuestionType::Noul.temperature_bucket(2), "noul:2");
        assert_eq!(QuestionType::Choice.temperature_bucket(2), "choice:2");
        assert_eq!(QuestionType::Choice.temperature_bucket(4), "choice:3-5");
        assert_eq!(QuestionType::Score.temperature_bucket(3), "score:3-5");
        assert_eq!(QuestionType::Choice.temperature_bucket(7), "choice:6-10");
        assert_eq!(QuestionType::Choice.temperature_bucket(77), "choice:11+");
    }

    #[test]
    fn a_fitted_bucket_beats_the_per_type_scalar() {
        let mut cal = Calibration {
            per_type: [1.6, 1.25, 1.98],
            by_bucket: HashMap::new(),
        };
        assert_eq!(cal.temperature(QuestionType::Choice, 4), 1.6);
        cal.by_bucket.insert("choice:3-5".into(), 1.76);
        assert_eq!(cal.temperature(QuestionType::Choice, 4), 1.76);
        // A bucket that was never fitted falls back rather than failing.
        assert_eq!(cal.temperature(QuestionType::Choice, 40), 1.6);
    }

    #[test]
    fn a_nonsense_temperature_falls_back_to_one() {
        // A zero divisor would turn every answer into a one-hot; a negative
        // one would invert it. Both are silent.
        let cal = Calibration {
            per_type: [0.0, f32::NAN, -1.0],
            by_bucket: HashMap::new(),
        };
        for t in [
            QuestionType::Choice,
            QuestionType::Score,
            QuestionType::Noul,
        ] {
            assert_eq!(cal.temperature(t, 3), 1.0);
        }
    }

    #[test]
    fn temperature_flattens_without_reordering() {
        let logits = [2.0f32, 0.5, -1.0];
        let hot = calibrated_probabilities(&logits, 1.0);
        let cool = calibrated_probabilities(&logits, 2.0);
        for p in [&hot, &cool] {
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert!(cool[0] < hot[0], "a larger divisor spreads the mass");
        assert!(cool[2] > hot[2]);
        // Calibration must never change which answer is given.
        assert_eq!(
            hot.iter().cloned().fold(f64::MIN, f64::max),
            hot[0],
            "argmax moved"
        );
        assert_eq!(cool.iter().cloned().fold(f64::MIN, f64::max), cool[0]);
    }

    #[test]
    fn masked_markers_do_not_overflow_the_softmax() {
        // Absent options are filled with -1e4; exp of the raw value is 0 or
        // inf depending on which way it is shifted.
        let p = calibrated_probabilities(&[3.0, -1e4, -1e4], 1.0);
        assert!((p[0] - 1.0).abs() < 1e-9, "{p:?}");
        assert!(p[1] == 0.0 && p[2] == 0.0);
    }

    #[test]
    fn confidence_spans_certain_to_uniform() {
        assert!((confidence(&[0.25, 0.25, 0.25, 0.25])).abs() < 1e-9);
        assert!(confidence(&[1.0, 0.0]) > 0.999);
        // Normalized, so the same shape scores the same at any cardinality.
        let two = confidence(&[0.9, 0.1]);
        let ten: Vec<f64> = std::iter::once(0.9)
            .chain(std::iter::repeat_n(0.1 / 9.0, 9))
            .collect();
        assert!(confidence(&ten) > two, "more options, same top mass");
    }

    #[test]
    fn a_score_answers_between_its_levels() {
        // The useful answer to an ordinal question is 1.44, which is not a
        // level; an argmax would report 2 and lose that.
        let p = calibrated_probabilities(&[0.0, 1.0, 1.55], 1.0);
        let s = expected_level(&p);
        assert!(s > 1.0 && s < 2.0, "{s}");
    }

    #[test]
    fn the_checkpoint_config_is_read_as_written() {
        let cfg = DecisionConfig::from_json(
            r#"{"encoder":"answerdotai/ModernBERT-large","max_len":512,"head_max_len":192,
                "temperature":[1.6369,1.2514,1.9834],
                "temperature_by_options":{"noul:2":1.9834,"choice:3-5":1.7602}}"#,
        )
        .unwrap();
        assert_eq!(cfg.max_len, 512);
        assert_eq!(cfg.head_max_len, 192);
        assert_eq!(cfg.state_budget(), 320);
        assert!((cfg.calibration.temperature(QuestionType::Choice, 4) - 1.7602).abs() < 1e-4);
        assert!((cfg.calibration.temperature(QuestionType::Score, 3) - 1.2514).abs() < 1e-4);
    }

    #[test]
    fn a_config_missing_everything_still_loads() {
        let cfg = DecisionConfig::from_json("{}").unwrap();
        assert_eq!(cfg, DecisionConfig::default());
        assert_eq!(cfg.calibration.temperature(QuestionType::Noul, 2), 1.0);
    }

    #[test]
    fn the_wire_spelling_is_lowercase() {
        let s = serde_json::to_string(&QuestionType::Noul).unwrap();
        assert_eq!(s, "\"noul\"");
    }
}
