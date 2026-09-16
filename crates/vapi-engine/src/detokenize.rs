/// Turns a growing token sequence into a stream of text deltas.
///
/// Decoding tokens one at a time does not work. A single UTF-8 character can
/// be split across two tokens, and in most BPE tokenizers a token's text
/// depends on what preceded it (leading-space handling especially). Decoding
/// the *whole* sequence every step would be correct but quadratic.
///
/// So this keeps two offsets into the token list and decodes only the tail
/// between them: `[prefix_offset..read_offset]` is text already emitted, and
/// `[prefix_offset..]` is that plus whatever is new. The difference is the
/// delta, and decoding both from the same starting point means the boundary
/// effects cancel out. When the new text ends in the Unicode replacement
/// character the token sequence is mid-character, so nothing is emitted until
/// the next token completes it.
pub struct IncrementalDetokenizer {
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
    emitted: String,
}

impl Default for IncrementalDetokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            emitted: String::new(),
        }
    }

    /// Seed with prompt tokens, which are decoded but never emitted.
    pub fn with_prompt(prompt: &[u32]) -> Self {
        Self {
            tokens: prompt.to_vec(),
            prefix_offset: prompt.len(),
            read_offset: prompt.len(),
            emitted: String::new(),
        }
    }

    /// All text emitted so far.
    pub fn text(&self) -> &str {
        &self.emitted
    }

    /// Feed one token and get the text delta. Empty when the token only
    /// completes part of a character — the bytes are not lost, they arrive
    /// with the next token.
    pub fn push<F>(&mut self, token: u32, decode: F) -> String
    where
        F: Fn(&[u32]) -> String,
    {
        self.tokens.push(token);

        let prefix_text = decode(&self.tokens[self.prefix_offset..self.read_offset]);
        let new_text = decode(&self.tokens[self.prefix_offset..]);

        // A trailing replacement character means the tokenizer could not form
        // a complete character from these bytes yet.
        if new_text.len() <= prefix_text.len() || new_text.ends_with('\u{FFFD}') {
            return String::new();
        }

        let delta = new_text[prefix_text.len()..].to_string();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        self.emitted.push_str(&delta);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Toy tokenizer: each id maps to a byte string, and decoding is
    /// concatenation followed by a lossy UTF-8 conversion — which is exactly
    /// how a real tokenizer behaves when a character straddles two tokens.
    fn decoder(table: HashMap<u32, &'static [u8]>) -> impl Fn(&[u32]) -> String {
        move |ids: &[u32]| {
            let mut bytes = Vec::new();
            for id in ids {
                bytes.extend_from_slice(table.get(id).copied().unwrap_or(b""));
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    fn ascii_decoder() -> impl Fn(&[u32]) -> String {
        decoder(HashMap::from([
            (1, &b"Hel"[..]),
            (2, &b"lo"[..]),
            (3, &b" wor"[..]),
            (4, &b"ld"[..]),
        ]))
    }

    #[test]
    fn plain_ascii_streams_token_by_token() {
        let d = ascii_decoder();
        let mut it = IncrementalDetokenizer::new();
        let deltas: Vec<String> = [1, 2, 3, 4].iter().map(|&t| it.push(t, &d)).collect();
        assert_eq!(deltas, vec!["Hel", "lo", " wor", "ld"]);
        assert_eq!(it.text(), "Hello world");
    }

    #[test]
    fn concatenated_deltas_equal_the_full_decode() {
        let d = ascii_decoder();
        let mut it = IncrementalDetokenizer::new();
        let mut acc = String::new();
        for t in [1, 2, 3, 4] {
            acc.push_str(&it.push(t, &d));
        }
        assert_eq!(
            acc,
            d(&[1, 2, 3, 4]),
            "streaming must reconstruct the whole text"
        );
        assert_eq!(it.text(), acc);
    }

    #[test]
    fn a_character_split_across_two_tokens_is_not_corrupted() {
        // "é" is 0xC3 0xA9. Split it across two tokens: naive per-token
        // decoding would emit U+FFFD twice and lose the character.
        let d = decoder(HashMap::from([
            (1, &b"caf"[..]),
            (2, &b"\xc3"[..]),
            (3, &b"\xa9"[..]),
            (4, &b"!"[..]),
        ]));
        let mut it = IncrementalDetokenizer::new();

        assert_eq!(it.push(1, &d), "caf");
        assert_eq!(it.push(2, &d), "", "half a character must not be emitted");
        assert_eq!(
            it.push(3, &d),
            "é",
            "the character arrives once it is complete"
        );
        assert_eq!(it.push(4, &d), "!");
        assert_eq!(it.text(), "café!");
        assert!(!it.text().contains('\u{FFFD}'));
    }

    #[test]
    fn a_four_byte_emoji_split_three_ways_survives() {
        // U+1F600 is F0 9F 98 80.
        let d = decoder(HashMap::from([
            (1, &b"hi "[..]),
            (2, &b"\xf0\x9f"[..]),
            (3, &b"\x98"[..]),
            (4, &b"\x80"[..]),
        ]));
        let mut it = IncrementalDetokenizer::new();
        let mut acc = String::new();
        for t in [1, 2, 3, 4] {
            acc.push_str(&it.push(t, &d));
        }
        assert_eq!(acc, "hi 😀");
    }

    #[test]
    fn prompt_tokens_are_context_but_never_emitted() {
        let d = ascii_decoder();
        let mut it = IncrementalDetokenizer::with_prompt(&[1, 2]);
        let delta = it.push(3, &d);
        assert_eq!(delta, " wor", "only generated tokens reach the client");
        assert_eq!(it.text(), " wor");
    }

    #[test]
    fn leading_space_context_is_preserved() {
        // Decoding token 3 alone gives " wor"; the point is that the emitted
        // stream matches a full decode rather than per-token decodes.
        let d = ascii_decoder();
        let mut it = IncrementalDetokenizer::new();
        let mut acc = String::new();
        for t in [1, 2, 3, 4] {
            acc.push_str(&it.push(t, &d));
        }
        assert_eq!(acc, d(&[1, 2, 3, 4]));
    }

    #[test]
    fn an_empty_token_yields_an_empty_delta_without_desyncing() {
        let d = decoder(HashMap::from([
            (1, &b"a"[..]),
            (2, &b""[..]),
            (3, &b"b"[..]),
        ]));
        let mut it = IncrementalDetokenizer::new();
        assert_eq!(it.push(1, &d), "a");
        assert_eq!(it.push(2, &d), "");
        assert_eq!(it.push(3, &d), "b");
        assert_eq!(it.text(), "ab");
    }
}
