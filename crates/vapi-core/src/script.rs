//! Which writing system a piece of text is in.
//!
//! This exists because of a specific, documented failure. A decision model
//! trained on an English encoder does not degrade gracefully on text it
//! cannot read — it stays confident. The published figure for the English
//! checkpoint on Khmer is **0.000 accuracy at 0.952 confidence**, which means
//! no confidence threshold downstream can catch it. The only thing that can
//! is looking at the bytes before the forward pass, which costs microseconds.
//!
//! Deliberately coarse: enough to tell "this checkpoint cannot read this" from
//! "it can", not a language identifier. Latin covers the languages the English
//! encoder was trained on plus many it handles adequately; everything else is
//! named so an operator can see *why* a request was flagged.

use std::fmt;

/// A writing system, at the granularity that decides which checkpoint can
/// read a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
    Thai,
    Lao,
    Tibetan,
    Myanmar,
    Georgian,
    Armenian,
    Ethiopic,
    Khmer,
    Han,
    Kana,
    Hangul,
    /// Digits, punctuation, emoji: anything that carries no script.
    Neutral,
}

impl Script {
    /// The script of one character.
    pub fn of(c: char) -> Self {
        let u = c as u32;
        match u {
            0x0041..=0x005A | 0x0061..=0x007A => Self::Latin,
            0x00C0..=0x024F | 0x1E00..=0x1EFF | 0x2C60..=0x2C7F => Self::Latin,
            0x0370..=0x03FF | 0x1F00..=0x1FFF => Self::Greek,
            0x0400..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Self::Cyrillic,
            0x0530..=0x058F | 0xFB13..=0xFB17 => Self::Armenian,
            0x0590..=0x05FF | 0xFB1D..=0xFB4F => Self::Hebrew,
            0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF => Self::Arabic,
            0x0900..=0x097F | 0xA8E0..=0xA8FF => Self::Devanagari,
            0x0980..=0x09FF => Self::Bengali,
            0x0A00..=0x0A7F => Self::Gurmukhi,
            0x0A80..=0x0AFF => Self::Gujarati,
            0x0B80..=0x0BFF => Self::Tamil,
            0x0C00..=0x0C7F => Self::Telugu,
            0x0C80..=0x0CFF => Self::Kannada,
            0x0D00..=0x0D7F => Self::Malayalam,
            0x0D80..=0x0DFF => Self::Sinhala,
            0x0E00..=0x0E7F => Self::Thai,
            0x0E80..=0x0EFF => Self::Lao,
            0x0F00..=0x0FFF => Self::Tibetan,
            0x1000..=0x109F | 0xAA60..=0xAA7F => Self::Myanmar,
            0x10A0..=0x10FF | 0x2D00..=0x2D2F => Self::Georgian,
            0x1200..=0x137F | 0x2D80..=0x2DDF => Self::Ethiopic,
            0x1780..=0x17FF | 0x19E0..=0x19FF => Self::Khmer,
            0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF => Self::Kana,
            0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => Self::Hangul,
            0x2E80..=0x2FDF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Self::Han,
            0x20000..=0x323AF => Self::Han,
            _ => Self::Neutral,
        }
    }

    /// Whether a checkpoint trained on Latin text can be expected to read it.
    ///
    /// Latin only. Cyrillic and Greek share nothing with a byte-level Latin
    /// vocabulary beyond punctuation, and the measured collapse is the same.
    pub fn is_latin(self) -> bool {
        matches!(self, Self::Latin | Self::Neutral)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Latin => "latin",
            Self::Cyrillic => "cyrillic",
            Self::Greek => "greek",
            Self::Arabic => "arabic",
            Self::Hebrew => "hebrew",
            Self::Devanagari => "devanagari",
            Self::Bengali => "bengali",
            Self::Gurmukhi => "gurmukhi",
            Self::Gujarati => "gujarati",
            Self::Tamil => "tamil",
            Self::Telugu => "telugu",
            Self::Kannada => "kannada",
            Self::Malayalam => "malayalam",
            Self::Sinhala => "sinhala",
            Self::Thai => "thai",
            Self::Lao => "lao",
            Self::Tibetan => "tibetan",
            Self::Myanmar => "myanmar",
            Self::Georgian => "georgian",
            Self::Armenian => "armenian",
            Self::Ethiopic => "ethiopic",
            Self::Khmer => "khmer",
            Self::Han => "han",
            Self::Kana => "kana",
            Self::Hangul => "hangul",
            Self::Neutral => "neutral",
        }
    }
}

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What script a text is mostly written in, and how much of it that is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub script: Script,
    /// Share of the script-carrying characters, 0.0..=1.0. A document that is
    /// half English and half Hindi reads as Devanagari at 0.5, and the share
    /// is what decides whether that matters.
    pub share: f32,
    /// Characters that carried a script at all.
    pub letters: usize,
}

impl Reading {
    /// Whether an English-only checkpoint should be trusted with this.
    ///
    /// A handful of characters is not evidence: a Latin document quoting one
    /// Chinese name is still a Latin document. The threshold is on the share,
    /// not on the presence.
    pub fn readable_by_latin_model(&self) -> bool {
        self.script.is_latin() || self.letters < 8 || self.share < 0.25
    }
}

/// Read a text's dominant script.
///
/// Counts characters, ignoring anything script-neutral, and stops after
/// `LIMIT` of them — a long document's script is settled well before its end,
/// and this runs in front of every request.
pub fn read(text: &str) -> Reading {
    const LIMIT: usize = 4096;
    let mut counts: Vec<(Script, usize)> = Vec::new();
    let mut letters = 0usize;
    for c in text.chars() {
        let s = Script::of(c);
        if s == Script::Neutral {
            continue;
        }
        letters += 1;
        match counts.iter_mut().find(|(k, _)| *k == s) {
            Some((_, n)) => *n += 1,
            None => counts.push((s, 1)),
        }
        if letters >= LIMIT {
            break;
        }
    }
    match counts.iter().max_by_key(|(_, n)| *n) {
        Some(&(script, n)) => Reading {
            script,
            share: n as f32 / letters as f32,
            letters,
        },
        None => Reading {
            script: Script::Neutral,
            share: 1.0,
            letters: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scripts_that_matter_are_told_apart() {
        for (text, want) in [
            ("Please refund the duplicate charge", Script::Latin),
            ("मुझसे दो बार शुल्क लिया गया", Script::Devanagari),
            ("Пожалуйста, верните деньги", Script::Cyrillic),
            ("الرجاء إعادة المبلغ", Script::Arabic),
            ("សូមសងប្រាក់មកវិញ", Script::Khmer),
            ("請退還重複的費用", Script::Han),
            ("重複した請求を返金してください", Script::Kana),
            ("중복 청구를 환불해 주세요", Script::Hangul),
            ("กรุณาคืนเงิน", Script::Thai),
            ("Παρακαλώ επιστρέψτε", Script::Greek),
            ("נא להחזיר את הכסף", Script::Hebrew),
        ] {
            assert_eq!(read(text).script, want, "{text}");
        }
    }

    #[test]
    fn a_latin_document_survives_a_foreign_name() {
        // The failure this guards against is the opposite of the obvious one:
        // refusing a perfectly readable English ticket because it quotes a
        // product name in Chinese.
        let r = read("Our customer 田中 reported a duplicate charge on invoice 4411");
        assert_eq!(r.script, Script::Latin);
        assert!(r.readable_by_latin_model());
    }

    #[test]
    fn a_document_that_is_mostly_not_latin_is_flagged() {
        let r = read("Ticket #4411: मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।");
        assert_eq!(r.script, Script::Devanagari);
        assert!(!r.readable_by_latin_model(), "{r:?}");
    }

    #[test]
    fn numbers_and_punctuation_carry_no_script() {
        let r = read("#4411 — 2026-09-23 (100%) !!!");
        assert_eq!(r.script, Script::Neutral);
        assert_eq!(r.letters, 0);
        assert!(r.readable_by_latin_model());
    }

    #[test]
    fn a_few_characters_are_not_evidence() {
        // Not enough to route on, and guessing from three letters would flip
        // requests between checkpoints at random.
        assert!(read("日本").readable_by_latin_model());
        assert!(!read("日本語のチケットです、返金をお願いします").readable_by_latin_model());
    }

    #[test]
    fn a_long_document_is_read_from_its_beginning_only() {
        // Bounded work per request: the script is settled long before the end
        // of a 200 KB ticket thread.
        let text = "मुझसे दो बार शुल्क लिया गया ".repeat(2000);
        let r = read(&text);
        assert_eq!(r.script, Script::Devanagari);
        assert!(r.letters <= 4096);
    }

    #[test]
    fn the_share_reflects_a_mixed_document() {
        let r = read("refund please: कृपया पैसे वापस करें");
        assert!(r.share > 0.3 && r.share < 0.75, "{r:?}");
    }
}
