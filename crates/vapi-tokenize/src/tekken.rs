//! Mistral's Tekken tokenizer.
//!
//! A tiktoken-style byte-level BPE: a regex that splits text into pieces, a
//! table of byte strings ranked by merge priority, and a block of special
//! tokens in front of the vocabulary. It is not a `tokenizers` JSON and cannot
//! be loaded as one — Voxtral ships `tekken.json` and no `tokenizer.json` —
//! so this reads the format directly.
//!
//! Two details of the format decide the id arithmetic, and getting either
//! wrong shifts every token by a constant, which reads as fluent nonsense
//! rather than as an error:
//!
//! - **Special tokens occupy the first `num_special_tokens` ids**, and a
//!   vocabulary entry of rank `r` is therefore id `r + num_special_tokens`.
//! - **The vocabulary is truncated.** The file carries 150,000 entries and the
//!   model's `vocab_size` is 131,072, so only the first
//!   `vocab_size - num_special_tokens` of them exist as far as the model is
//!   concerned. Ranks past that must never be emitted.

use std::collections::HashMap;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;
use vapi_core::{Error, Result};

#[derive(Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<VocabEntry>,
    special_tokens: Vec<SpecialEntry>,
}

#[derive(Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Deserialize)]
struct VocabEntry {
    token_bytes: String,
}

#[derive(Deserialize)]
struct SpecialEntry {
    rank: usize,
    token_str: String,
    #[serde(default)]
    is_control: bool,
}

/// One special token.
#[derive(Clone, Debug)]
pub struct Special {
    pub text: String,
    /// Control tokens are structural and never appear in decoded output.
    pub is_control: bool,
}

pub struct Tekken {
    /// Byte string for every ordinary token, indexed by `id - num_special`.
    vocab: Vec<Vec<u8>>,
    /// The inverse, for encoding.
    ranks: HashMap<Vec<u8>, u32>,
    /// Indexed by id; `None` for an unused slot in the special block.
    specials: Vec<Option<Special>>,
    by_name: HashMap<String, u32>,
    num_special: usize,
    pattern: fancy_regex::Regex,
}

impl Tekken {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let file: TekkenFile = serde_json::from_str(text)
            .map_err(|e| Error::Tokenizer(format!("tekken.json: {e}")))?;
        let num_special = file.config.default_num_special_tokens;
        let keep = file.config.default_vocab_size.saturating_sub(num_special);

        let b64 = base64::engine::general_purpose::STANDARD;
        let mut vocab = Vec::with_capacity(keep.min(file.vocab.len()));
        let mut ranks = HashMap::with_capacity(vocab.capacity());
        for (rank, entry) in file.vocab.iter().take(keep).enumerate() {
            let bytes = b64
                .decode(&entry.token_bytes)
                .map_err(|e| Error::Tokenizer(format!("tekken vocab rank {rank}: {e}")))?;
            ranks.insert(bytes.clone(), rank as u32);
            vocab.push(bytes);
        }

        let mut specials = vec![None; num_special];
        let mut by_name = HashMap::new();
        for entry in &file.special_tokens {
            if entry.rank >= num_special {
                continue;
            }
            by_name.insert(entry.token_str.clone(), entry.rank as u32);
            specials[entry.rank] = Some(Special {
                text: entry.token_str.clone(),
                is_control: entry.is_control,
            });
        }

        let pattern = fancy_regex::Regex::new(&file.config.pattern)
            .map_err(|e| Error::Tokenizer(format!("tekken split pattern: {e}")))?;

        Ok(Self {
            vocab,
            ranks,
            specials,
            by_name,
            num_special,
            pattern,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.num_special + self.vocab.len()
    }

    pub fn num_special(&self) -> usize {
        self.num_special
    }

    /// The id of a special token by its text, e.g. `[BEGIN_AUDIO]`.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    pub fn special(&self, id: u32) -> Option<&Special> {
        self.specials.get(id as usize)?.as_ref()
    }

    pub fn is_special(&self, id: u32) -> bool {
        (id as usize) < self.num_special
    }

    /// Raw bytes for one token, or empty for an unused special slot.
    fn bytes_of(&self, id: u32) -> &[u8] {
        if self.is_special(id) {
            return match &self.specials[id as usize] {
                Some(s) => s.text.as_bytes(),
                None => &[],
            };
        }
        match self.vocab.get(id as usize - self.num_special) {
            Some(b) => b,
            None => &[],
        }
    }

    /// Decode to bytes, so a caller can handle a multi-byte character split
    /// across two tokens itself.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if skip_special
                && self
                    .special(id)
                    .is_some_and(|s| s.is_control || self.is_special(id))
            {
                continue;
            }
            out.extend_from_slice(self.bytes_of(id));
        }
        out
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }

    /// Encode text as ordinary tokens.
    ///
    /// Special tokens are never produced, even if their spelling appears in
    /// the text: a caller's transcript prompt must not be able to open an
    /// audio section or end a turn.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = 0usize;
        // `find_iter` yields the pieces the pattern matches; anything between
        // them is not skipped but encoded too, so no byte is ever dropped.
        for m in self.pattern.find_iter(text).flatten() {
            let bytes = text.as_bytes();
            if m.start() > at {
                self.encode_piece(&bytes[at..m.start()], &mut out);
            }
            self.encode_piece(&bytes[m.start()..m.end()], &mut out);
            at = m.end();
        }
        if at < text.len() {
            self.encode_piece(&text.as_bytes()[at..], &mut out);
        }
        out
    }

    /// Byte-pair merge within one piece, lowest rank first.
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        if let Some(&rank) = self.ranks.get(piece) {
            out.push(rank + self.num_special as u32);
            return;
        }
        let mut parts: Vec<&[u8]> = piece.chunks(1).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let mut joined = Vec::with_capacity(parts[i].len() + parts[i + 1].len());
                joined.extend_from_slice(parts[i]);
                joined.extend_from_slice(parts[i + 1]);
                if let Some(&rank) = self.ranks.get(&joined)
                    && best.is_none_or(|(_, b)| rank < b)
                {
                    best = Some((i, rank));
                }
            }
            let Some((i, _)) = best else { break };
            // Merging means replacing two adjacent parts with the slice that
            // spans both; they are always contiguous in `piece`.
            let start = parts[..i].iter().map(|p| p.len()).sum::<usize>();
            let len = parts[i].len() + parts[i + 1].len();
            parts[i] = &piece[start..start + len];
            parts.remove(i + 1);
        }
        for part in parts {
            match self.ranks.get(part) {
                Some(&rank) => out.push(rank + self.num_special as u32),
                // A byte with no entry cannot happen in a complete byte-level
                // vocabulary, but truncation could in principle remove one.
                None => out.push(self.by_name.get("<unk>").copied().unwrap_or(0)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-token vocabulary over `a`, `b`, `ab` and `abab`, with two
    /// specials in front, so the id arithmetic is checkable by hand.
    fn tiny() -> Tekken {
        let b64 = base64::engine::general_purpose::STANDARD;
        let entry = |s: &str| format!(r#"{{"rank":0,"token_bytes":"{}"}}"#, b64.encode(s));
        let vocab = ["a", "b", "ab", "abab"]
            .iter()
            .map(|s| entry(s))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{"config":{{"pattern":"\\s+|\\S+","default_vocab_size":6,
                 "default_num_special_tokens":2}},
                 "vocab":[{vocab}],
                 "special_tokens":[{{"rank":0,"token_str":"<unk>","is_control":true}},
                                   {{"rank":1,"token_str":"[AUDIO]","is_control":true}}]}}"#
        );
        Tekken::from_json(&json).unwrap()
    }

    #[test]
    fn ids_are_offset_by_the_special_block() {
        // Rank 0 is id 2, because ids 0 and 1 belong to the specials. Getting
        // this wrong shifts every token by a constant and reads as fluent
        // nonsense rather than as an error.
        let t = tiny();
        assert_eq!(t.num_special(), 2);
        assert_eq!(t.vocab_size(), 6);
        assert_eq!(t.decode(&[2], false), "a");
        assert_eq!(t.decode(&[3], false), "b");
        assert_eq!(t.decode(&[4], false), "ab");
        assert_eq!(t.special_id("[AUDIO]"), Some(1));
    }

    #[test]
    fn the_vocabulary_is_truncated_to_the_models_size() {
        // The file carries more entries than the model has ids for; a rank
        // past the end must not be reachable.
        let t = tiny();
        assert_eq!(t.vocab.len(), 4, "vocab_size 6 minus 2 specials");
        assert_eq!(
            t.decode(&[99], false),
            "",
            "past the end decodes to nothing"
        );
    }

    #[test]
    fn merging_prefers_the_lowest_rank() {
        // "abab" is one token at rank 3; the greedy pair merge has to find it
        // rather than stopping at "ab" + "ab".
        let t = tiny();
        assert_eq!(t.encode("abab"), vec![5]);
        assert_eq!(t.encode("aba"), vec![4, 2]);
        assert_eq!(t.encode("ba"), vec![3, 2]);
    }

    #[test]
    fn encoding_round_trips_through_decoding() {
        let t = tiny();
        for text in ["a", "ab", "abab", "ababab", "bbaa"] {
            assert_eq!(t.decode(&t.encode(text), false), text, "{text}");
        }
    }

    #[test]
    fn control_tokens_are_dropped_when_asked() {
        let t = tiny();
        assert_eq!(t.decode(&[1, 2, 4], true), "aab");
        assert_eq!(t.decode(&[1, 2, 4], false), "[AUDIO]aab");
    }

    #[test]
    fn a_special_spelling_in_the_text_stays_text() {
        // Otherwise a transcript prompt could open an audio section.
        let t = tiny();
        let ids = t.encode("[AUDIO]");
        assert!(!ids.contains(&1), "{ids:?}");
    }

    #[test]
    fn decoding_bytes_leaves_a_split_character_to_the_caller() {
        // A multi-byte character can straddle two tokens; the incremental
        // detokenizer needs the bytes, not a lossy string per token.
        let b64 = base64::engine::general_purpose::STANDARD;
        let json = format!(
            r#"{{"config":{{"pattern":"\\s+|\\S+","default_vocab_size":3,
                 "default_num_special_tokens":1}},
                 "vocab":[{{"rank":0,"token_bytes":"{}"}},{{"rank":1,"token_bytes":"{}"}}],
                 "special_tokens":[{{"rank":0,"token_str":"<unk>","is_control":true}}]}}"#,
            b64.encode([0xE2, 0x82]),
            b64.encode([0xAC])
        );
        let t = Tekken::from_json(&json).unwrap();
        assert_eq!(t.decode_bytes(&[1, 2], false), "€".as_bytes());
        assert_eq!(t.decode(&[1, 2], false), "€");
    }
}
