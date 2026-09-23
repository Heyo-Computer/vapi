//! Tekken against mistral-common.
//!
//! Round-tripping is not enough on its own: a tokenizer that is internally
//! consistent but disagrees with the one the model was trained on produces
//! fluent output made of the wrong words. These are the reference's own ids.

use std::path::Path;

use vapi_tokenize::Tekken;

fn golden() -> Option<serde_json::Value> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/goldens/voxtral-tekken.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn load() -> Option<(Tekken, serde_json::Value)> {
    let g = golden()?;
    let dir = std::path::PathBuf::from(g["dir"].as_str()?);
    let path = dir.join("tekken.json");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    Some((Tekken::from_file(path).unwrap(), g))
}

#[test]
fn the_vocabulary_is_the_size_the_model_expects() {
    let Some((t, g)) = load() else { return };
    assert_eq!(t.vocab_size(), g["n_words"].as_u64().unwrap() as usize);
    assert_eq!(
        t.num_special(),
        g["num_special_tokens"].as_u64().unwrap() as usize
    );
}

#[test]
fn the_control_tokens_the_speech_path_needs_are_where_the_reference_says() {
    let Some((t, g)) = load() else { return };
    for (name, id) in g["specials"].as_object().unwrap() {
        assert_eq!(
            t.special_id(name),
            Some(id.as_u64().unwrap() as u32),
            "{name}"
        );
    }
    // The ones the decode loop actually reads.
    assert!(t.special_id("[BEGIN_AUDIO]").is_some());
    assert!(t.special_id("[STREAMING_PAD]").is_some());
}

#[test]
fn encoding_matches_the_reference_id_for_id() {
    let Some((t, g)) = load() else { return };
    let mut checked = 0;
    for case in g["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let got = t.encode(text);
        assert_eq!(
            got,
            want,
            "{:?}\n  got  {:?}\n  want {:?}",
            text,
            &got[..got.len().min(16)],
            &want[..want.len().min(16)]
        );
        checked += 1;
    }
    assert!(checked > 0);
    eprintln!("tekken: {checked} strings encode identically");
}

#[test]
fn decoding_matches_the_reference() {
    let Some((t, g)) = load() else { return };
    for case in g["cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let want = case["decoded"].as_str().unwrap();
        assert_eq!(t.decode(&ids, false), want, "{:?}", case["text"]);
    }
}
