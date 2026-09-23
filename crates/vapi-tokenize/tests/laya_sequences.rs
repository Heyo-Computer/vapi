//! The sequence builder against the checkpoint's own builder.
//!
//! Token-for-token, not approximately: the scorer reads specific positions,
//! so an off-by-one in the head is not a small quality loss but an answer
//! drawn from the wrong hidden state. Skipped when the model is not present.

use std::path::{Path, PathBuf};

use vapi_openai::decision::{Ordered, Question};
use vapi_tokenize::open_decision_model;

fn goldens(name: &str) -> Option<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens")
        .join(name);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn model_dir(g: &serde_json::Value) -> Option<PathBuf> {
    let dir = PathBuf::from(g["dir"].as_str()?);
    dir.join("tokenizer/tokenizer.json").exists().then_some(dir)
}

fn check(golden: &str) {
    let Some(g) = goldens(golden) else {
        eprintln!("skipping: tests/goldens/{golden} not generated");
        return;
    };
    let Some(dir) = model_dir(&g) else {
        eprintln!("skipping: {} not present", g["dir"]);
        return;
    };
    let (bundle, format) = open_decision_model(&dir).unwrap();
    assert_eq!(
        format.config.max_len,
        g["config"]["max_len"].as_u64().unwrap() as usize
    );

    let mut rows_checked = 0;
    for case in g["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let questions: Ordered<Question> =
            serde_json::from_value(case["questions"].clone()).unwrap();
        let state = match &case["state"] {
            serde_json::Value::String(s) => s.clone(),
            other => vapi_openai::decision::python_json(other),
        };
        let expected = case["rows"].as_array().unwrap();
        assert_eq!(questions.len(), expected.len(), "{name}");

        for ((id, q), want) in questions.iter().zip(expected) {
            assert_eq!(id, want["id"].as_str().unwrap(), "{name}: question order");
            let options = q.option_texts().unwrap();
            let want_options: Vec<String> =
                serde_json::from_value(want["options"].clone()).unwrap();
            assert_eq!(options, want_options, "{name}/{id}: option rendering");

            let built = format
                .build(&bundle, q.qtype, &q.instruction_text(), &options, &state)
                .unwrap();
            let want_tokens: Vec<u32> = serde_json::from_value(want["tokens"].clone()).unwrap();
            let want_markers: Vec<u32> = serde_json::from_value(want["markers"].clone()).unwrap();
            assert_eq!(built.markers, want_markers, "{name}/{id}: marker positions");
            assert_eq!(
                built.tokens.len(),
                want_tokens.len(),
                "{name}/{id}: sequence length"
            );
            if built.tokens != want_tokens {
                let at = built
                    .tokens
                    .iter()
                    .zip(&want_tokens)
                    .position(|(a, b)| a != b)
                    .unwrap();
                let lo = at.saturating_sub(6);
                let hi_got = (at + 6).min(built.tokens.len());
                let hi_want = (at + 6).min(want_tokens.len());
                panic!(
                    "{name}/{id}: diverges at token {at}\n  got  {:?}\n  want {:?}\n  text got  {:?}\n  text want {:?}",
                    &built.tokens[lo..hi_got],
                    &want_tokens[lo..hi_want],
                    bundle.decode(&built.tokens[lo..hi_got], false),
                    bundle.decode(&want_tokens[lo..hi_want], false),
                );
            }
            rows_checked += 1;
        }
    }
    assert!(rows_checked > 0, "{golden}: nothing was checked");
    eprintln!("{golden}: {rows_checked} sequences match");
}

#[test]
fn tiny_sequences_match_the_reference_builder() {
    check("laya-tiny.json");
}

#[test]
fn real_sequences_match_the_reference_builder() {
    check("laya.json");
}

/// The multilingual checkpoint's tokenizer is Metaspace rather than
/// byte-level, and names its structural tokens `<bos>`/`<eos>`/`<mask>`
/// rather than the bracketed spellings. Both are resolved from the
/// tokenizer's own config, and this is what proves it.
#[test]
fn multilingual_sequences_match_the_reference_builder() {
    check("laya-multilingual.json");
}
