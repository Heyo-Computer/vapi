//! The single-pass decision surface: a state, typed questions, calibrated
//! answers.
//!
//! Shaped after the published request format the decision models use, so a
//! client written against one can point at vapi without changes. Unlike chat,
//! there is no streaming and no sampling: the request is a pure function of
//! its inputs, and the response carries a probability distribution rather
//! than text.

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use vapi_core::{Error, QuestionType, Result};

use crate::Usage;

/// A JSON object that keeps the order it arrived in.
///
/// Question order is not cosmetic: the options of a `choice` go into the
/// prompt in this order, so reordering them changes what the model reads and
/// therefore the probabilities it reports. Answers come back in the same
/// order for the same reason a client expects its own question list back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ordered<V>(pub Vec<(String, V)>);

impl<V> Ordered<V> {
    pub fn iter(&self) -> impl Iterator<Item = &(String, V)> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

impl<V> FromIterator<(String, V)> for Ordered<V> {
    fn from_iter<I: IntoIterator<Item = (String, V)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<V: Serialize> Serialize for Ordered<V> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

impl<'de, V: Deserialize<'de>> Deserialize<'de> for Ordered<V> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V_<V>(std::marker::PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for V_<V> {
            type Value = Ordered<V>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut out = Vec::with_capacity(m.size_hint().unwrap_or(4));
                while let Some((k, v)) = m.next_entry()? {
                    out.push((k, v));
                }
                Ok(Ordered(out))
            }
        }
        d.deserialize_map(V_(std::marker::PhantomData))
    }
}

/// What a question's answer space is made of.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Criteria {
    /// `{"billing": "invoices and refunds", "technical": null}` — an ordered
    /// set of named options, each with an optional description.
    Named(Ordered<Option<String>>),
    /// `["not urgent", "soon", "critical"]` — the levels of a score, or a
    /// choice whose option names are their own description.
    List(Vec<String>),
}

/// One typed question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    /// What to ask. A string, or any JSON the caller would rather hand over
    /// verbatim.
    pub instructions: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Criteria>,
}

/// Serialize JSON the way the reference builder does.
///
/// `json.dumps` separates with `", "` and `": "`; serde's compact writer uses
/// `","` and `":"`. That is not cosmetic here — the state is tokenized, and a
/// missing space after every key is a different token sequence than the one
/// the model was trained and calibrated on. On the email fixture it is five
/// tokens of difference, silently.
pub fn python_json(value: &serde_json::Value) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
    match serde::Serialize::serialize(value, &mut ser) {
        Ok(()) => String::from_utf8(out).unwrap_or_default(),
        Err(_) => value.to_string(),
    }
}

/// Round a reported probability the way the published format does.
///
/// Four decimals is enough to act on and short enough to read; more digits
/// would imply a precision the calibration does not have.
pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// The default descriptions a `noul` question gets when the caller gives none.
/// Part of the trained prompt, not a nicety: the scorer read these strings
/// during training.
pub const NOUL_FALSE: &str = "no, the statement does not hold";
pub const NOUL_TRUE: &str = "yes, the statement holds";

impl Question {
    /// The instruction text, exactly as the model should read it.
    pub fn instruction_text(&self) -> String {
        match &self.instructions {
            serde_json::Value::String(s) => s.clone(),
            other => python_json(other),
        }
    }

    /// The option *labels* — what the answer is reported in terms of.
    ///
    /// For a choice these are the caller's own keys; for a score they are the
    /// level indices; for a noul they are false and true.
    pub fn labels(&self) -> Vec<String> {
        match (self.qtype, &self.criteria) {
            (QuestionType::Choice, Some(Criteria::Named(m))) => {
                m.keys().map(str::to_string).collect()
            }
            (QuestionType::Choice, Some(Criteria::List(v))) => v.clone(),
            (QuestionType::Score, Some(Criteria::List(v))) => {
                (0..v.len()).map(|i| i.to_string()).collect()
            }
            (QuestionType::Score, Some(Criteria::Named(m))) => {
                m.keys().map(str::to_string).collect()
            }
            (QuestionType::Noul, _) => vec!["false".into(), "true".into()],
            (_, None) => Vec::new(),
        }
    }

    /// What each level of a score question means, for the answer's legend.
    /// Empty for the other types, which report in terms of their own labels.
    pub fn legend(&self) -> Ordered<String> {
        match (self.qtype, &self.criteria) {
            (QuestionType::Score, Some(Criteria::List(v))) => v
                .iter()
                .enumerate()
                .map(|(i, c)| (i.to_string(), c.clone()))
                .collect(),
            (QuestionType::Score, Some(Criteria::Named(m))) => m
                .iter()
                .enumerate()
                .map(|(i, (k, _))| (i.to_string(), k.clone()))
                .collect(),
            _ => Ordered::default(),
        }
    }

    /// The option *texts* the model is shown, one per marker.
    ///
    /// These strings are part of the trained format down to the `level %d: `
    /// prefix; a paraphrase here is a silent accuracy loss, not an error.
    pub fn option_texts(&self) -> Result<Vec<String>> {
        let bad = |m: &str| Error::InvalidRequest(m.to_string());
        let texts = match (self.qtype, &self.criteria) {
            (QuestionType::Choice, Some(Criteria::Named(m))) => m
                .iter()
                .map(|(k, v)| match v.as_deref().filter(|d| !d.is_empty()) {
                    Some(d) => format!("{k}: {d}"),
                    None => k.clone(),
                })
                .collect(),
            (QuestionType::Choice, Some(Criteria::List(v))) => v.clone(),
            (QuestionType::Score, Some(Criteria::List(v))) => v
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {c}"))
                .collect(),
            (QuestionType::Score, Some(Criteria::Named(m))) => m
                .iter()
                .enumerate()
                .map(|(i, (k, v))| match v.as_deref().filter(|d| !d.is_empty()) {
                    Some(d) => format!("level {i}: {k}: {d}"),
                    None => format!("level {i}: {k}"),
                })
                .collect(),
            (QuestionType::Noul, crit) => {
                let named = match crit {
                    Some(Criteria::Named(m)) => Some(m),
                    _ => None,
                };
                let pick = |key: &str, default: &str| {
                    named
                        .and_then(|m| m.get(key))
                        .and_then(|v| v.clone())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| default.to_string())
                };
                vec![
                    format!("false: {}", pick("false", NOUL_FALSE)),
                    format!("true: {}", pick("true", NOUL_TRUE)),
                ]
            }
            (QuestionType::Choice, None) => {
                return Err(bad("a choice question needs criteria"));
            }
            (QuestionType::Score, None) => {
                return Err(bad("a score question needs its levels as criteria"));
            }
        };
        if texts.len() < 2 {
            return Err(bad(
                "a question needs at least two options to have an answer",
            ));
        }
        Ok(texts)
    }
}

/// `POST /v1/decisions`.
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// The thing being judged: free text, or any JSON document.
    pub state: serde_json::Value,
    pub questions: Ordered<Question>,
}

impl DecisionRequest {
    /// The state as the model reads it: a bare string stays as it is,
    /// anything else is serialized compactly.
    pub fn state_text(&self) -> String {
        match &self.state {
            serde_json::Value::String(s) => s.clone(),
            other => python_json(other),
        }
    }
}

/// One answer, shaped by the question's type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Choice {
        /// The most likely option's own key.
        choice: String,
        probabilities: Ordered<f64>,
        /// One minus the normalized entropy of the distribution: 1.0 when the
        /// model is certain, 0.0 when it is spreading evenly over the options.
        confidence: f64,
        act_probability: f64,
    },
    Score {
        /// The expectation over the levels, not the argmax — an ordinal
        /// question's useful answer is 1.4, which no single level is.
        score: f64,
        legend: Ordered<String>,
        probabilities: Ordered<f64>,
        confidence: f64,
        act_probability: f64,
    },
    Noul {
        /// The probability the statement holds.
        noul: f64,
        act_probability: f64,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct DecisionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub answers: Ordered<Answer>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> DecisionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn questions_and_options_keep_the_order_they_arrived_in() {
        // Alphabetical would be a plausible-looking bug: the answer keys
        // would still all be there, with the probabilities shuffled.
        let r = parse(
            r#"{"state":"hi","questions":{
                "zebra":{"type":"noul","instructions":"z?"},
                "apple":{"type":"choice","instructions":"a?",
                         "criteria":{"delta":null,"charlie":null,"bravo":null}}}}"#,
        );
        assert_eq!(r.questions.keys().collect::<Vec<_>>(), ["zebra", "apple"]);
        let apple = r.questions.get("apple").unwrap();
        assert_eq!(apple.labels(), ["delta", "charlie", "bravo"]);
    }

    #[test]
    fn choice_options_read_as_the_checkpoint_renders_them() {
        let r = parse(
            r#"{"state":"hi","questions":{"q":{"type":"choice","instructions":"which?",
                "criteria":{"billing":"invoices, payments, refunds","other":null}}}}"#,
        );
        let q = r.questions.get("q").unwrap();
        assert_eq!(
            q.option_texts().unwrap(),
            ["billing: invoices, payments, refunds", "other"]
        );
    }

    #[test]
    fn score_levels_are_numbered_the_way_the_model_was_trained() {
        let r = parse(
            r#"{"state":"hi","questions":{"q":{"type":"score","instructions":"how urgent?",
                "criteria":["not urgent","soon","critical"]}}}"#,
        );
        let q = r.questions.get("q").unwrap();
        assert_eq!(
            q.option_texts().unwrap(),
            ["level 0: not urgent", "level 1: soon", "level 2: critical"]
        );
        assert_eq!(q.labels(), ["0", "1", "2"]);
    }

    #[test]
    fn noul_gets_its_trained_defaults_and_false_comes_first() {
        // p[1] is the answer, so swapping these inverts every yes/no.
        let r = parse(r#"{"state":"hi","questions":{"q":{"type":"noul","instructions":"true?"}}}"#);
        let q = r.questions.get("q").unwrap();
        let texts = q.option_texts().unwrap();
        assert_eq!(texts[0], format!("false: {NOUL_FALSE}"));
        assert_eq!(texts[1], format!("true: {NOUL_TRUE}"));
        assert_eq!(q.labels(), ["false", "true"]);
    }

    #[test]
    fn a_noul_may_describe_its_own_two_sides() {
        let r = parse(
            r#"{"state":"hi","questions":{"q":{"type":"noul","instructions":"churn?",
                "criteria":{"true":"threatens to leave","false":"content"}}}}"#,
        );
        let texts = r.questions.get("q").unwrap().option_texts().unwrap();
        assert_eq!(texts, ["false: content", "true: threatens to leave"]);
    }

    #[test]
    fn a_choice_without_criteria_is_a_bad_request() {
        let r = parse(r#"{"state":"hi","questions":{"q":{"type":"choice","instructions":"?"}}}"#);
        let err = r.questions.get("q").unwrap().option_texts().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)), "{err}");
    }

    #[test]
    fn a_one_option_choice_is_a_bad_request() {
        // It has no answer to give: a softmax over one marker is always 1.0.
        let r = parse(
            r#"{"state":"hi","questions":{"q":{"type":"choice","instructions":"?",
                "criteria":{"only":null}}}}"#,
        );
        assert!(r.questions.get("q").unwrap().option_texts().is_err());
    }

    #[test]
    fn a_state_may_be_text_or_a_document() {
        assert_eq!(
            parse(r#"{"state":"just text","questions":{}}"#).state_text(),
            "just text"
        );
        let doc = parse(r#"{"state":{"subject":"x","n":2},"questions":{}}"#).state_text();
        assert!(doc.starts_with('{') && doc.contains("\"subject\""), "{doc}");
    }

    #[test]
    fn a_document_state_is_spelled_the_way_the_model_was_trained_to_read_it() {
        let r = parse(r#"{"state":{"from":"a@b.c","n":2,"tags":["x","y"]},"questions":{}}"#);
        assert_eq!(
            r.state_text(),
            r#"{"from": "a@b.c", "n": 2, "tags": ["x", "y"]}"#
        );
    }

    #[test]
    fn answers_serialize_under_their_question_ids() {
        let answers: Ordered<Answer> = [(
            "dept".to_string(),
            Answer::Noul {
                noul: 0.82,
                act_probability: 1.0,
            },
        )]
        .into_iter()
        .collect();
        let v = serde_json::to_value(&answers).unwrap();
        assert_eq!(v["dept"]["type"], "noul");
        assert_eq!(v["dept"]["noul"], 0.82);
    }
}
