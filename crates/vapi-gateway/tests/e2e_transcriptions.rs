//! End to end: a speech worker, over NATS and HTTP.
//!
//! The audio comes from the golden rather than from a file on the machine, so
//! the test is self-contained and the expected transcript is the reference's
//! own — the same clip the parity tests use, through the whole stack instead
//! of through a library call.

mod common;

use common::{Stack, binaries, golden, model_dir, multipart, nats_url, request, send, wav};

/// Its own model id, so this test's subjects, durable consumer and registry
/// entry cannot collide with a stack someone is running by hand.
const MODEL_ID: &str = "vapi-e2e/voxtral";

const GOLDEN: &str = "voxtral-model.json";

/// Post a WAV to the transcription endpoint.
fn transcribe(port: u16, audio: &[u8], fields: &[(&str, &str)]) -> common::Reply {
    let (boundary, body) = multipart(fields, Some(("clip.wav", audio)));
    send(
        port,
        "POST",
        "/v1/audio/transcriptions",
        &format!("multipart/form-data; boundary={boundary}"),
        &body,
    )
    .expect("request")
}

fn as_json(reply: &common::Reply) -> serde_json::Value {
    serde_json::from_str(&reply.body).unwrap_or_else(|e| {
        panic!(
            "status {} returned something that is not JSON: {e}\n{}",
            reply.status, reply.body
        )
    })
}

/// One stack, many assertions: starting it costs an 8 GB model load, and the
/// things worth checking are all about the same running system.
#[test]
fn a_speech_worker_transcribes_over_the_wire() {
    let (Some(model), Some(nats), Some(g)) = (model_dir(GOLDEN), nats_url(), golden(GOLDEN)) else {
        eprintln!(
            "skipping: needs the voxtral checkpoint, its goldens, and a reachable NATS \
             (VAPI_E2E_NATS)"
        );
        return;
    };
    if binaries().is_none() {
        eprintln!("skipping: build the binaries first (cargo build --release --features cuda)");
        return;
    }

    // The reference's own clip and its own answer.
    let samples: Vec<f32> = serde_json::from_value(g["samples"].clone()).unwrap();
    let expected = g["text"].as_str().unwrap().trim().to_string();
    let seconds = samples.len() as f32 / 16_000.0;
    let clip = wav(&samples, 16_000);

    let stack = Stack::start(MODEL_ID, &model, &nats);

    // --- the transcript ----------------------------------------------------
    let reply = transcribe(stack.port, &clip, &[]);
    assert_eq!(reply.status, 200, "{}", reply.body);
    let json = as_json(&reply);
    assert_eq!(
        json["text"].as_str().unwrap().trim(),
        expected,
        "the stack disagrees with the reference: {}",
        reply.body
    );
    // The default shape is one field, as OpenAI's is.
    assert_eq!(json.as_object().unwrap().len(), 1, "{}", reply.body);

    // --- the other response formats ----------------------------------------
    let bare = transcribe(stack.port, &clip, &[("response_format", "text")]);
    assert_eq!(bare.body.trim(), expected, "{}", bare.body);

    let verbose = as_json(&transcribe(
        stack.port,
        &clip,
        &[("response_format", "verbose_json")],
    ));
    assert_eq!(verbose["task"], "transcribe", "{verbose}");
    assert!(
        (verbose["duration"].as_f64().unwrap() as f32 - seconds).abs() < 0.2,
        "duration should be the audio's, not the request's: {verbose}"
    );
    // The number that decides whether this model is usable live: one position
    // is 80 ms, so below 1.0 a session falls behind the speaker.
    let realtime = verbose["realtime_factor"].as_f64().unwrap();
    assert!(realtime > 0.0, "{verbose}");
    eprintln!("transcribed {seconds:.2}s at {realtime:.2}x realtime: {expected:?}");

    // --- the audio the gateway has to convert ------------------------------
    // 48 kHz stereo is what a caller's recording actually looks like; the
    // gateway resamples and mixes down, and getting either wrong changes the
    // words rather than failing.
    let stereo: Vec<f32> = samples
        .iter()
        .flat_map(|&s| {
            // Upsample by three (16 kHz to 48 kHz) and duplicate to two
            // channels, so the content is unchanged.
            [s, s, s, s, s, s]
        })
        .collect();
    let mut wide = wav(&stereo, 48_000);
    // Patch the header to say two channels: `wav` writes mono.
    wide[22..24].copy_from_slice(&2u16.to_le_bytes());
    wide[32..34].copy_from_slice(&4u16.to_le_bytes());
    wide[28..32].copy_from_slice(&(48_000u32 * 4).to_le_bytes());
    let converted = as_json(&transcribe(stack.port, &wide, &[]));
    assert_eq!(
        converted["text"].as_str().unwrap().trim(),
        expected,
        "48 kHz stereo should transcribe the same as 16 kHz mono: {converted}"
    );

    // --- refusals ----------------------------------------------------------
    let not_audio = as_json(&transcribe(stack.port, b"ID3\x04not an mp3 either", &[]));
    let message = not_audio["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("WAV only"),
        "a caller who sends an MP3 should be told what is accepted: {not_audio}"
    );

    let (boundary, body) = multipart(&[("model", "voxtral")], None);
    let no_file = as_json(
        &send(
            stack.port,
            "POST",
            "/v1/audio/transcriptions",
            &format!("multipart/form-data; boundary={boundary}"),
            &body,
        )
        .unwrap(),
    );
    assert!(
        no_file["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no file field"),
        "{no_file}"
    );

    let bad_format = as_json(&transcribe(
        stack.port,
        &clip,
        &[("response_format", "srt")],
    ));
    assert!(
        bad_format["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("srt"),
        "a subtitle format must be refused, not silently answered as JSON: {bad_format}"
    );

    // Longer than the envelope can carry. Silence is fine: the clip is
    // rejected on its duration, before anything looks at the samples.
    let too_long = as_json(&transcribe(
        stack.port,
        &wav(&vec![0f32; 16_000 * 160], 16_000),
        &[],
    ));
    let message = too_long["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("160s") && message.contains("150s"),
        "the limit should name both the clip and the cap: {too_long}"
    );

    // A chat request against a speech deployment is refused by the gateway
    // before anything is published: the checkpoint ships no chat template.
    let wrong = as_json(
        &request(
            stack.port,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#),
        )
        .unwrap(),
    );
    assert!(
        wrong["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no chat template"),
        "{wrong}"
    );

    // --- the dashboard saw it ----------------------------------------------
    let stats = request(stack.port, "GET", "/dashboard/stats", None).unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats.body).unwrap();
    assert!(
        stats["workers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w["model"] == MODEL_ID),
        "{stats}"
    );
    assert!(
        stats["recent"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["kind"] == "transcription"),
        "{stats}"
    );
}
