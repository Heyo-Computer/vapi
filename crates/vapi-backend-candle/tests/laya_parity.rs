//! The decision model against the reference implementation.
//!
//! Probabilities, not just argmax: calibration is what this model is for, so
//! an answer that picks the right option with the wrong confidence has lost
//! the thing worth having. Skipped when the checkpoint is not present.

#![cfg(feature = "candle")]

use std::path::{Path, PathBuf};

use vapi_backend_candle::{CandleEncoder, EncoderLoadOptions};
use vapi_core::config::{DType, DeviceKind};
use vapi_core::{QuestionType, calibrated_probabilities};
use vapi_engine::{EncoderBackend, EncoderBatch, EncoderRow};

/// f32 on the CPU against torch's f32, through 28 encoder layers and two head
/// layers. Reductions in a different order move the last few bits.
const LOGIT_TOLERANCE: f32 = 2e-3;
/// Probabilities are what a caller acts on, and they are bounded, so they get
/// a tighter absolute bound than the logits they came from.
const PROBABILITY_TOLERANCE: f64 = 1e-3;

struct Golden {
    json: serde_json::Value,
    dir: PathBuf,
}

fn load(name: &str) -> Option<Golden> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens")
        .join(name);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: {} not generated", path.display());
        return None;
    };
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let dir = PathBuf::from(json["dir"].as_str()?);
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} not present", dir.display());
        return None;
    }
    Some(Golden { json, dir })
}

fn cpu(dir: &Path) -> CandleEncoder {
    CandleEncoder::load(
        dir,
        &EncoderLoadOptions {
            // The CPU path in f32 is the numerical reference, as everywhere
            // else in this crate.
            dtype: DType::F32,
            device: DeviceKind::Cpu,
        },
    )
    .expect("load decision model")
}

fn qtype(name: &str) -> QuestionType {
    match name {
        "choice" => QuestionType::Choice,
        "score" => QuestionType::Score,
        "noul" => QuestionType::Noul,
        other => panic!("unknown question type {other}"),
    }
}

fn rows_of(case: &serde_json::Value) -> Vec<EncoderRow> {
    case["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| EncoderRow {
            tokens: serde_json::from_value(r["tokens"].clone()).unwrap(),
            markers: serde_json::from_value(r["markers"].clone()).unwrap(),
            qtype: qtype(r["qtype"].as_str().unwrap()),
        })
        .collect()
}

fn check_encoder(g: &Golden) {
    let probe = &g.json["encoder_probe"];
    let tokens: Vec<u32> = serde_json::from_value(probe["tokens"].clone()).unwrap();
    let want: Vec<Vec<f32>> = serde_json::from_value(probe["hidden"].clone()).unwrap();
    let got = cpu(&g.dir).encoder_hidden(&tokens).unwrap();
    assert_eq!(got.len(), want.len(), "sequence length");

    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.len(), b.len(), "hidden size at position {i}");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            assert!(
                d < 5e-3,
                "encoder drifts at position {i} dim {j}: {x} vs {y}"
            );
            worst = worst.max(d);
        }
    }
    eprintln!("encoder: worst hidden-state drift {worst:.2e}");
}

fn check_answers(g: &Golden) {
    let mut encoder = cpu(&g.dir);
    let cal = vapi_core::DecisionConfig::from_json(&g.json["config"].to_string())
        .unwrap()
        .calibration;

    let mut worst_logit = 0f32;
    let mut worst_prob = 0f64;
    let mut checked = 0;
    for case in g.json["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = rows_of(case);
        let out = encoder.forward(&EncoderBatch::new(rows.clone())).unwrap();
        assert_eq!(out.rows.len(), rows.len(), "{name}");

        for ((row, want), scores) in rows
            .iter()
            .zip(case["rows"].as_array().unwrap())
            .zip(&out.rows)
        {
            let id = want["id"].as_str().unwrap();
            let want_logits: Vec<f32> = serde_json::from_value(want["logits"].clone()).unwrap();
            assert_eq!(scores.logits.len(), want_logits.len(), "{name}/{id}");
            for (i, (a, b)) in scores.logits.iter().zip(&want_logits).enumerate() {
                let d = (a - b).abs();
                assert!(
                    d < LOGIT_TOLERANCE,
                    "{name}/{id}: option {i} logit {a} vs {b}"
                );
                worst_logit = worst_logit.max(d);
            }

            // And the calibrated answer, which is what a caller sees.
            let t = cal.temperature(row.qtype, row.markers.len());
            let p = calibrated_probabilities(&scores.logits, t);
            let answer = &want["answer"];
            let want_p: Vec<f64> = answer["probabilities"]
                .as_object()
                .map(|m| m.values().filter_map(serde_json::Value::as_f64).collect())
                .unwrap_or_else(|| {
                    // A noul reports only p(true), and it is p[1].
                    let t = answer["noul"].as_f64().unwrap();
                    vec![1.0 - t, t]
                });
            assert_eq!(p.len(), want_p.len(), "{name}/{id}: option count");
            for (i, (a, b)) in p.iter().zip(&want_p).enumerate() {
                let d = (a - b).abs();
                assert!(
                    d < PROBABILITY_TOLERANCE,
                    "{name}/{id}: option {i} probability {a} vs {b}"
                );
                worst_prob = worst_prob.max(d);
            }

            // The act head reads the answer distribution, so it drifts last
            // and catches what the logits alone would not.
            let want_act = want["act_probability"].as_f64().unwrap() as f32;
            assert!(
                (scores.act - want_act).abs() < 2e-3,
                "{name}/{id}: act {} vs {want_act}",
                scores.act
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
    eprintln!(
        "{checked} answers match: worst logit {worst_logit:.2e}, worst probability {worst_prob:.2e}"
    );
}

#[test]
fn tiny_encoder_matches_transformers() {
    let Some(g) = load("laya-tiny.json") else {
        return;
    };
    check_encoder(&g);
}

#[test]
fn tiny_answers_match_the_reference() {
    let Some(g) = load("laya-tiny.json") else {
        return;
    };
    check_answers(&g);
}

#[test]
fn real_encoder_matches_transformers() {
    let Some(g) = load("laya.json") else {
        return;
    };
    check_encoder(&g);
}

#[test]
fn multilingual_encoder_matches_transformers() {
    let Some(g) = load("laya-multilingual.json") else {
        return;
    };
    check_encoder(&g);
}

#[test]
fn multilingual_answers_match_the_reference() {
    let Some(g) = load("laya-multilingual.json") else {
        return;
    };
    check_answers(&g);
}

#[test]
fn real_answers_match_the_reference() {
    let Some(g) = load("laya.json") else {
        return;
    };
    check_answers(&g);
}

/// What the deployment actually runs: the device, in bf16.
///
/// A much looser bound than the CPU path on purpose. bf16 carries eight bits
/// of mantissa, and these logits come through 28 encoder layers and two head
/// layers, so the question is not whether they drift but whether the drift
/// changes an answer. Both are checked: the argmax exactly, the probabilities
/// within a bound a caller could not act on differently.
#[cfg(feature = "cuda")]
fn check_cuda(golden: &str) {
    let Some(g) = load(golden) else {
        return;
    };
    let mut gpu = match CandleEncoder::load(
        &g.dir,
        &EncoderLoadOptions {
            dtype: DType::Bf16,
            device: DeviceKind::Cuda,
        },
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: no usable CUDA device ({e})");
            return;
        }
    };
    // The windowed layers are the one place the kernel and the reference
    // could disagree structurally rather than numerically: FlashAttention
    // takes the window as two integers, the reference builds an explicit
    // |i - j| <= 64 mask. Comparing them at the *same* dtype is what
    // separates a wrong window from half-precision drift — and half-precision
    // drift here is large, because ModernBERT carries activation outliers of
    // magnitude ~25 that the scorer's own LayerNorm removes before they reach
    // an answer.
    let longest = g.json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["rows"].as_array().unwrap())
        .max_by_key(|r| r["tokens"].as_array().unwrap().len())
        .unwrap();
    let row = EncoderRow {
        tokens: serde_json::from_value(longest["tokens"].clone()).unwrap(),
        markers: serde_json::from_value(longest["markers"].clone()).unwrap(),
        qtype: qtype(longest["qtype"].as_str().unwrap()),
    };
    assert!(
        row.tokens.len() > 128,
        "no golden row is longer than the window, so this proves nothing"
    );
    let batch = EncoderBatch::new(vec![row]);
    let with_kernel = gpu.forward(&batch).unwrap();
    gpu.use_reference_attention(true);
    let with_reference = gpu.forward(&batch).unwrap();
    gpu.use_reference_attention(false);
    for (a, b) in with_kernel.rows[0]
        .logits
        .iter()
        .zip(&with_reference.rows[0].logits)
    {
        assert!(
            (a - b).abs() < 0.05,
            "windowed attention disagrees with the reference: {a} vs {b}"
        );
    }
    eprintln!(
        "cuda: kernel and reference agree over {} tokens",
        batch.rows[0].tokens.len()
    );

    let cal = vapi_core::DecisionConfig::from_json(&g.json["config"].to_string())
        .unwrap()
        .calibration;

    let mut worst = 0f64;
    let mut checked = 0;
    for case in g.json["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = rows_of(case);
        let out = gpu.forward(&EncoderBatch::new(rows.clone())).unwrap();
        for ((row, want), scores) in rows
            .iter()
            .zip(case["rows"].as_array().unwrap())
            .zip(&out.rows)
        {
            let id = want["id"].as_str().unwrap();
            let want_logits: Vec<f32> = serde_json::from_value(want["logits"].clone()).unwrap();
            let t = cal.temperature(row.qtype, row.markers.len());
            let got = calibrated_probabilities(&scores.logits, t);
            let reference = calibrated_probabilities(&want_logits, t);
            let top = |p: &[f64]| {
                p.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            };
            assert_eq!(
                top(&got),
                top(&reference),
                "{name}/{id}: bf16 changed the answer"
            );
            for (a, b) in got.iter().zip(&reference) {
                let d = (a - b).abs();
                assert!(d < 0.02, "{name}/{id}: probability {a} vs {b}");
                worst = worst.max(d);
            }
            checked += 1;
        }
    }
    eprintln!("cuda: {checked} answers agree, worst probability drift {worst:.2e}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_answers_agree_with_the_cpu_reference() {
    check_cuda("laya.json");
}

/// A second encoder shape through the same loader: mmBERT-base is 22 layers
/// of 768 with a 256k vocabulary and a Metaspace tokenizer, where the English
/// checkpoint is 28 of 1024, byte-level. Passing on one says little about the
/// other.
#[cfg(feature = "cuda")]
#[test]
fn cuda_multilingual_answers_agree_with_the_cpu_reference() {
    check_cuda("laya-multilingual.json");
}
