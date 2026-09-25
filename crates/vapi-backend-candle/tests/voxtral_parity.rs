//! Voxtral against the reference, stage by stage.
//!
//! Checked separately at every stage on purpose. A transcript that is merely
//! *worse* is the hardest kind of wrong to localise, and the stages fail in
//! characteristic ways: a transposed projection shifts the encoder, a
//! mis-grouped projector shifts the embeddings by a frame, a wrong
//! conditioning vector leaves everything plausible and slightly off.

#![cfg(feature = "candle")]

use std::path::{Path, PathBuf};

use vapi_audio::MelSpectrogram;
use vapi_backend_candle::{CandleVoxtral, EncoderLoadOptions};
use vapi_core::config::{DType, DeviceKind};

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

/// One model for every test in this file.
///
/// 4B parameters in f32 is about 16 GB, and cargo runs tests in parallel:
/// loading one per test is how you meet the OOM killer rather than a failing
/// assertion.
static MODEL: std::sync::OnceLock<CandleVoxtral> = std::sync::OnceLock::new();

fn cpu(dir: &Path) -> &'static CandleVoxtral {
    MODEL.get_or_init(|| {
        CandleVoxtral::load(
            dir,
            &EncoderLoadOptions {
                // f32 on the CPU is the numerical reference, as everywhere here.
                dtype: DType::F32,
                device: DeviceKind::Cpu,
            },
        )
        .expect("load voxtral")
    })
}

/// The clip's log-mel frames, padded exactly as the reference processor pads:
/// 32 positions of silence, the audio rounded up to a whole position, then
/// `(delay + 1) + 10` more. Recovering this was most of the work — the golden
/// records what the processor produced, not what it was handed.
fn mel_of(g: &Golden, v: &CandleVoxtral) -> Vec<Vec<f32>> {
    let samples: Vec<f32> = serde_json::from_value(g.json["samples"].clone()).unwrap();
    let cfg = &v.model.config;
    let padded = cfg.pad_audio(&samples, v.mel.hop_length, cfg.default_num_delay_tokens);
    MelSpectrogram::new(v.mel).compute(&padded)
}

/// Rows of a `(n, d)` tensor as `Vec<Vec<f32>>`.
fn rows(t: &candle_core::Tensor, n: usize) -> Vec<Vec<f32>> {
    t.narrow(0, 0, n)
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
}

fn worst(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: width at row {i}");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            assert!(d < 2e-2, "{what}: row {i} dim {j}: {x} vs {y}");
            worst = worst.max(d);
        }
    }
    worst
}

#[test]
fn the_frontend_produces_the_frames_the_reference_saw() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let mel = mel_of(&g, v);
    let want: Vec<usize> = serde_json::from_value(g.json["mel_shape"].clone()).unwrap();
    assert_eq!(mel.len(), want[1], "mel bins");
    assert_eq!(mel[0].len(), want[2], "mel frames");
}

#[test]
fn the_time_conditioning_is_parameter_free_and_matches() {
    // There is no `time_embedding` tensor in the checkpoint: it is a plain
    // sinusoid of the delay, and getting its form wrong changes every layer.
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let delay = g.json["num_delay_tokens"].as_u64().unwrap() as usize;
    let got = v.model.time_embedding(delay).unwrap();
    let want: Vec<f32> = serde_json::from_value(g.json["t_cond"].clone()).unwrap();
    assert_eq!(got.len(), v.model.config.text.hidden_size);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-5, "t_cond[{i}]: {a} vs {b}");
    }
}

#[test]
fn the_encoder_and_projector_match_the_reference() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let mel = mel_of(&g, v);
    // The reference recorded these from the prompt's slice of the audio, not
    // the whole clip.
    let prompt_frames = g.json["prompt_mel_shape"].as_array().unwrap()[2]
        .as_u64()
        .unwrap() as usize;
    let cut: Vec<Vec<f32>> = mel.iter().map(|r| r[..prompt_frames].to_vec()).collect();

    let hidden = v.model.encoder_hidden(&cut).unwrap();
    let want_shape: Vec<usize> = serde_json::from_value(g.json["encoder_shape"].clone()).unwrap();
    assert_eq!(
        hidden.dims(),
        [want_shape[1], want_shape[2]],
        "encoder shape"
    );
    let want: Vec<Vec<f32>> = serde_json::from_value(g.json["encoder_hidden"].clone()).unwrap();
    let e = worst(&rows(&hidden, want.len()), &want, "encoder");

    let embeds = v.model.audio_embeds(&cut).unwrap();
    let want_shape: Vec<usize> =
        serde_json::from_value(g.json["audio_embeds_shape"].clone()).unwrap();
    assert_eq!(
        embeds.dims(),
        [want_shape[1], want_shape[2]],
        "embeds shape"
    );
    let want: Vec<Vec<f32>> = serde_json::from_value(g.json["audio_embeds"].clone()).unwrap();
    let p = worst(&rows(&embeds, want.len()), &want, "projector");
    eprintln!("encoder worst {e:.2e}, projector worst {p:.2e}");
}

#[test]
fn the_decoder_picks_the_same_tokens_as_the_reference() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let mel = mel_of(&g, v);
    let prompt: Vec<u32> = serde_json::from_value(g.json["input_ids"].clone()).unwrap();
    let prompt_frames = g.json["prompt_mel_shape"].as_array().unwrap()[2]
        .as_u64()
        .unwrap() as usize;
    let cut: Vec<Vec<f32>> = mel.iter().map(|r| r[..prompt_frames].to_vec()).collect();

    let audio = v.model.audio_embeds(&cut).unwrap();
    let cond = v
        .model
        .conditioning(g.json["num_delay_tokens"].as_u64().unwrap() as usize)
        .unwrap();
    let mut session = v.model.session();
    let logits = v.model.step(&mut session, &prompt, &audio, &cond).unwrap();

    let mut checked = 0;
    for step in g.json["steps"].as_array().unwrap() {
        let at = step["position"].as_u64().unwrap() as usize;
        let want_argmax = step["argmax"].as_u64().unwrap() as u32;
        let row = logits.i(at).unwrap().to_vec1::<f32>().unwrap();
        let got = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        assert_eq!(got, want_argmax, "position {at}: different token");

        // And the margin, not only the winner: a decoder that agrees on the
        // argmax while the distribution has drifted is still broken.
        let want_top: Vec<u32> = serde_json::from_value(step["top_ids"].clone()).unwrap();
        let want_logits: Vec<f32> = serde_json::from_value(step["top_logits"].clone()).unwrap();
        for (id, want) in want_top.iter().zip(&want_logits).take(3) {
            let got = row[*id as usize];
            assert!(
                (got - want).abs() < 0.2,
                "position {at} token {id}: logit {got} vs {want}"
            );
        }
        checked += 1;
    }
    assert!(checked > 0);
    eprintln!("{checked} decoder positions agree");
}

#[test]
fn the_prompt_is_built_the_way_the_processor_builds_it() {
    // An opening token and one pad per left-pad and delay position. A prompt
    // one token short shifts every audio position against its token.
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let want: Vec<u32> = serde_json::from_value(g.json["input_ids"].clone()).unwrap();
    let cfg = &v.model.config;
    let got = cfg.prompt(want[0], want[1], cfg.default_num_delay_tokens);
    assert_eq!(got, want);
}

#[test]
fn the_transcript_matches() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let mel = mel_of(&g, v);
    let prompt: Vec<u32> = serde_json::from_value(g.json["input_ids"].clone()).unwrap();
    let delay = g.json["num_delay_tokens"].as_u64().unwrap() as usize;

    let got = v.model.transcribe(&mel, &prompt, delay).unwrap();
    let want: Vec<u32> = serde_json::from_value(g.json["generated_ids"].clone()).unwrap();
    assert_eq!(got.len(), want.len(), "token count");
    assert_eq!(got, want, "token sequence");
    eprintln!("transcript matches: {} tokens", got.len());
}

/// Raw samples in, text out, with nothing from the golden but the audio and
/// the answer. This is the one test that would catch a mistake in the glue
/// between the pieces the other tests check separately.
#[test]
fn the_whole_path_transcribes_the_clip() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let v = cpu(&g.dir);
    let tokenizer = match vapi_tokenize::Tokenization::from_dir(&g.dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let tekken = tokenizer.tekken().expect("voxtral ships tekken.json");
    let bos = tekken.special_id("<s>").unwrap();
    let pad = tekken.special_id("[STREAMING_PAD]").unwrap();

    let samples: Vec<f32> = serde_json::from_value(g.json["samples"].clone()).unwrap();
    let tokens = v.transcribe(&samples, bos, pad).unwrap();
    // The prompt is not transcript, and the pads within it are not words.
    let text = tekken.decode(&tokens[v.prompt_len()..], true);

    let want = g.json["text"].as_str().unwrap();
    assert_eq!(text.trim(), want.trim(), "transcript");
    assert!(
        text.to_lowercase().contains("front"),
        "the clip says \"front, center\": {text:?}"
    );
    eprintln!("transcribed {:?}", text);
}

/// What the deployment actually runs: the device, in bf16.
///
/// The transcript is asserted exactly rather than approximately. Half
/// precision moves the logits, but a token is a discrete choice: if bf16
/// changes which word comes out, the drift is no longer cosmetic and the
/// right thing is to know.
#[cfg(feature = "cuda")]
#[test]
fn cuda_transcribes_the_same_words() {
    let Some(g) = load("voxtral-model.json") else {
        return;
    };
    let gpu = match CandleVoxtral::load(
        &g.dir,
        &EncoderLoadOptions {
            dtype: DType::Bf16,
            device: DeviceKind::Cuda,
        },
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("skipping: no usable CUDA device ({e})");
            return;
        }
    };
    let tokenizer = vapi_tokenize::Tokenization::from_dir(&g.dir).unwrap();
    let tekken = tokenizer.tekken().unwrap();
    let samples: Vec<f32> = serde_json::from_value(g.json["samples"].clone()).unwrap();
    let tokens = gpu
        .transcribe(
            &samples,
            tekken.special_id("<s>").unwrap(),
            tekken.special_id("[STREAMING_PAD]").unwrap(),
        )
        .unwrap();
    let text = tekken.decode(&tokens[gpu.prompt_len()..], true);
    assert_eq!(text.trim(), g.json["text"].as_str().unwrap().trim());
    eprintln!("cuda transcribed {text:?}");
}

use candle_core::IndexOp;
