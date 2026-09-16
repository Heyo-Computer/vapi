use vapi_core::Result;
use vapi_openai::ChatMessage;

use crate::template::{ChatTemplate, TemplateSource};

/// Token id reserved for end-of-sequence; byte values occupy 0..=255.
pub const BYTE_EOS: u32 = 256;

const CHATML: &str = "{% for message in messages %}\
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}\
{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// A tokenizer that maps each UTF-8 byte to its own id.
///
/// Not useful for a real model, but it makes the entire pipeline — templating,
/// queueing, batching, paged caching, streaming — runnable end to end with no
/// weights and no download. That matters more than it sounds: it means the
/// transport and scheduler can be exercised on a laptop before the GPU box
/// ever enters the picture.
pub struct ByteTokenizer {
    template: ChatTemplate,
}

impl ByteTokenizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            template: ChatTemplate::new(CHATML, TemplateSource::Explicit)?,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| b as u32).collect()
    }

    /// Decodes lossily on purpose: a prefix of a multi-byte character decodes
    /// to the replacement character, which is exactly the signal the
    /// incremental detokenizer uses to hold a partial character back.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&i| i < 256).map(|&i| i as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        Ok(self.encode(&self.template.render(messages, true)?))
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        vec![BYTE_EOS]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips() {
        let t = ByteTokenizer::new().unwrap();
        let ids = t.encode("hello");
        assert_eq!(ids, vec![104, 101, 108, 108, 111]);
        assert_eq!(t.decode(&ids), "hello");
    }

    #[test]
    fn multibyte_text_round_trips_whole() {
        let t = ByteTokenizer::new().unwrap();
        let ids = t.encode("café 😀");
        assert_eq!(t.decode(&ids), "café 😀");
    }

    #[test]
    fn a_partial_character_decodes_to_the_replacement_marker() {
        // This is what tells the incremental detokenizer to wait.
        let t = ByteTokenizer::new().unwrap();
        let ids = t.encode("é");
        assert!(t.decode(&ids[..1]).ends_with('\u{FFFD}'));
        assert_eq!(t.decode(&ids), "é");
    }

    #[test]
    fn chat_rendering_produces_tokens() {
        let t = ByteTokenizer::new().unwrap();
        let ids = t.encode_chat(&[ChatMessage::user("hi")]).unwrap();
        let text = t.decode(&ids);
        assert!(text.contains("<|im_start|>user"));
        assert!(text.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn eos_is_outside_the_byte_range() {
        let t = ByteTokenizer::new().unwrap();
        assert_eq!(t.eos_token_ids(), vec![256]);
        assert_eq!(t.decode(&[BYTE_EOS]), "", "EOS must not render as text");
    }
}
