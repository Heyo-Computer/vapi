//! `POST /v1/audio/transcriptions`.

use serde::{Deserialize, Serialize};

/// The default response shape: a bare transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// The verbose shape, which callers use to decide whether the answer is worth
/// trusting and what it cost.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerboseTranscriptionResponse {
    pub task: &'static str,
    pub duration: f32,
    pub text: String,
    /// How long the transcription itself took, and how that compares to the
    /// audio. Not in OpenAI's shape, and worth having: below 1.0 a live
    /// session would fall behind the speaker.
    pub processing_seconds: f32,
    pub realtime_factor: f32,
}

/// What `response_format` the caller asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TranscriptionFormat {
    #[default]
    Json,
    Text,
    VerboseJson,
}

impl TranscriptionFormat {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim() {
            "" | "json" => Some(Self::Json),
            "text" => Some(Self::Text),
            "verbose_json" => Some(Self::VerboseJson),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_formats_openai_clients_send_are_understood() {
        assert_eq!(
            TranscriptionFormat::parse("json"),
            Some(TranscriptionFormat::Json)
        );
        assert_eq!(
            TranscriptionFormat::parse(""),
            Some(TranscriptionFormat::Json)
        );
        assert_eq!(
            TranscriptionFormat::parse("text"),
            Some(TranscriptionFormat::Text)
        );
        assert_eq!(
            TranscriptionFormat::parse("verbose_json"),
            Some(TranscriptionFormat::VerboseJson)
        );
        // The subtitle formats are real in OpenAI's API and not supported
        // here, so they have to be refused rather than silently answered as
        // JSON.
        assert_eq!(TranscriptionFormat::parse("srt"), None);
        assert_eq!(TranscriptionFormat::parse("vtt"), None);
    }

    #[test]
    fn the_plain_shape_is_one_field() {
        let v = serde_json::to_value(TranscriptionResponse {
            text: "hello".into(),
        })
        .unwrap();
        assert_eq!(v["text"], "hello");
        assert_eq!(v.as_object().unwrap().len(), 1);
    }
}
