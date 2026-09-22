//! Parsing a model's structured output at the gateway.
//!
//! LFM2.5 answers inside a fixed frame: the prompt opens `<think>`, the
//! model writes its reasoning, closes with `</think>`, then answers; a tool
//! call is written as `<|tool_call_start|>[name(arg='value', ...)]
//! <|tool_call_end|>`. The markers are ordinary tokens, so they arrive in
//! the token stream as text. This module turns that stream into what an
//! OpenAI client expects: `reasoning_content`, `content`, and `tool_calls`
//! with JSON arguments. The markers are the format's; the format is used
//! only when the tokenizer's vocabulary has them.

use vapi_openai::ToolCall;
use vapi_tokenize::Tokenization;

/// How a model writes a tool call between its markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolSyntax {
    /// LFM2: `[name(arg='value', other=1)]`, Python call syntax.
    PythonCall,
    /// Qwen: `{"name": "f", "arguments": {...}}`, one object per call.
    Json,
}

/// The marker strings of a model's output format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputFormat {
    /// Opens a reasoning block, when the model writes the opener itself.
    /// LFM2's prompt ends with `<think>`, so there is nothing to open;
    /// Qwen writes `<think>` as its first token.
    pub think_open: Option<&'static str>,
    /// Closes the reasoning block.
    pub think_close: Option<&'static str>,
    pub tool_start: &'static str,
    pub tool_end: &'static str,
    pub syntax: ToolSyntax,
}

impl OutputFormat {
    /// The format of whichever model is loaded, read off its vocabulary.
    pub fn detect(tokenizer: &Tokenization) -> Option<Self> {
        let has = |t: &str| tokenizer.has_token(t);
        let thinks = has("<think>") && has("</think>");
        if has("<|tool_call_start|>") && has("<|tool_call_end|>") {
            // LFM2: the prompt opens the reasoning block for the model.
            return Some(Self {
                think_open: None,
                think_close: thinks.then_some("</think>"),
                tool_start: "<|tool_call_start|>",
                tool_end: "<|tool_call_end|>",
                syntax: ToolSyntax::PythonCall,
            });
        }
        if has("<tool_call>") && has("</tool_call>") {
            // Qwen: the model writes both markers itself.
            return Some(Self {
                think_open: thinks.then_some("<think>"),
                think_close: thinks.then_some("</think>"),
                tool_start: "<tool_call>",
                tool_end: "</tool_call>",
                syntax: ToolSyntax::Json,
            });
        }
        None
    }
}

/// What a piece of model output turned out to be.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
    ToolCalls(Vec<ToolCall>),
}

/// Incremental parser over token texts. Each marker is a single token, so
/// it arrives whole inside one delta; a delta may still carry text on
/// either side of it.
#[derive(Clone)]
pub struct OutputParser {
    fmt: OutputFormat,
    in_think: bool,
    /// Just left the reasoning block: the blank line the model puts between
    /// its thinking and its answer is not content, and it may arrive in a
    /// later delta than the closing marker.
    trim_leading: bool,
    tool_buf: Option<String>,
    calls_made: usize,
    call_id_seed: String,
}

impl OutputParser {
    pub fn new(fmt: OutputFormat, call_id_seed: &str) -> Self {
        Self {
            // A model that writes its own opener starts outside the block.
            in_think: fmt.think_close.is_some() && fmt.think_open.is_none(),
            trim_leading: false,
            fmt,
            tool_buf: None,
            calls_made: 0,
            call_id_seed: call_id_seed.to_string(),
        }
    }

    pub fn saw_tool_calls(&self) -> bool {
        self.calls_made > 0
    }

    pub fn push(&mut self, delta: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        let mut rest = delta;
        while !rest.is_empty() {
            if self.in_think {
                let close = self.fmt.think_close.expect("in_think implies a marker");
                match rest.find(close) {
                    Some(i) => {
                        if i > 0 {
                            out.push(Piece::Reasoning(rest[..i].to_string()));
                        }
                        self.in_think = false;
                        self.trim_leading = true;
                        rest = &rest[i + close.len()..];
                    }
                    None => {
                        out.push(Piece::Reasoning(rest.to_string()));
                        rest = "";
                    }
                }
            } else if let Some(buf) = &mut self.tool_buf {
                match rest.find(self.fmt.tool_end) {
                    Some(i) => {
                        buf.push_str(&rest[..i]);
                        let text = self.tool_buf.take().expect("buffering");
                        out.push(self.finish_tool_text(&text));
                        rest = &rest[i + self.fmt.tool_end.len()..];
                    }
                    None => {
                        buf.push_str(rest);
                        rest = "";
                    }
                }
            } else {
                // Whichever comes first: a tool call, or the model opening
                // a reasoning block it writes itself.
                let tool_at = rest.find(self.fmt.tool_start);
                let think_at = self.fmt.think_open.and_then(|m| rest.find(m));
                match (tool_at, think_at) {
                    (Some(t), think) if think.is_none_or(|k| t < k) => {
                        if let Some(c) = self.content(&rest[..t]) {
                            out.push(c);
                        }
                        self.tool_buf = Some(String::new());
                        rest = &rest[t + self.fmt.tool_start.len()..];
                    }
                    (_, Some(k)) => {
                        if let Some(c) = self.content(&rest[..k]) {
                            out.push(c);
                        }
                        self.in_think = true;
                        rest = &rest[k + self.fmt.think_open.expect("matched").len()..];
                    }
                    _ => {
                        if let Some(c) = self.content(rest) {
                            out.push(c);
                        }
                        rest = "";
                    }
                }
            }
        }
        out
    }

    /// End of stream: an unterminated tool call is handed back as text.
    pub fn finish(&mut self) -> Vec<Piece> {
        match self.tool_buf.take() {
            Some(text) if !text.is_empty() => {
                vec![Piece::Content(format!("{}{text}", self.fmt.tool_start))]
            }
            _ => Vec::new(),
        }
    }

    /// Content, with the blank line after the reasoning block dropped.
    /// Returns `None` when nothing is left.
    fn content(&mut self, text: &str) -> Option<Piece> {
        let text = if self.trim_leading {
            let t = text.trim_start_matches(['\n', '\r']);
            if !t.is_empty() {
                self.trim_leading = false;
            }
            t
        } else {
            text
        };
        (!text.is_empty()).then(|| Piece::Content(text.to_string()))
    }

    fn finish_tool_text(&mut self, text: &str) -> Piece {
        let parsed = match self.fmt.syntax {
            ToolSyntax::PythonCall => parse_calls(text),
            ToolSyntax::Json => parse_json_call(text),
        };
        match parsed {
            Ok(calls) if !calls.is_empty() => {
                let calls = calls
                    .into_iter()
                    .map(|(name, args)| {
                        let idx = self.calls_made;
                        self.calls_made += 1;
                        ToolCall::function(self.call_id(idx), name, args.to_string())
                    })
                    .collect();
                Piece::ToolCalls(calls)
            }
            _ => {
                // Not something we can hand to a client as a call; keep the
                // model's text so nothing is silently dropped.
                Piece::Content(format!(
                    "{}{text}{}",
                    self.fmt.tool_start, self.fmt.tool_end
                ))
            }
        }
    }

    fn call_id(&self, idx: usize) -> String {
        let h = blake3::hash(format!("{}:{idx}", self.call_id_seed).as_bytes());
        format!("call_{}", &h.to_hex()[..16])
    }
}

/// Parse `{"name": "f", "arguments": {...}}` as Qwen writes it, one call
/// per pair of markers.
pub fn parse_json_call(text: &str) -> Result<Vec<(String, serde_json::Value)>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text.trim()).map_err(|e| format!("tool call is not JSON: {e}"))?;
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| "tool call has no name".to_string())?;
    let args = match v.get("arguments") {
        None => serde_json::Value::Object(Default::default()),
        // Some models write the arguments as a JSON string rather than an
        // object; take either.
        Some(serde_json::Value::String(s)) => {
            serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.clone()))
        }
        Some(other) => other.clone(),
    };
    Ok(vec![(name.to_string(), args)])
}

/// Parse `[name(arg=value, ...), ...]` as LFM2 writes it: Python call
/// syntax with string, number, `True`/`False`/`None` and JSON-literal
/// argument values.
pub fn parse_calls(text: &str) -> Result<Vec<(String, serde_json::Value)>, String> {
    let mut p = Parser {
        s: text.trim(),
        i: 0,
    };
    p.ws();
    let bracketed = p.eat('[');
    let mut calls = Vec::new();
    loop {
        p.ws();
        if bracketed && p.eat(']') {
            break;
        }
        if p.at_end() {
            if bracketed {
                return Err("unterminated call list".into());
            }
            break;
        }
        let name = p.ident()?;
        p.ws();
        if !p.eat('(') {
            return Err(format!("expected '(' after {name}"));
        }
        let mut args = serde_json::Map::new();
        loop {
            p.ws();
            if p.eat(')') {
                break;
            }
            let key = p.ident()?;
            p.ws();
            if !p.eat('=') {
                return Err(format!("expected '=' after argument {key}"));
            }
            p.ws();
            let value = p.value()?;
            args.insert(key, value);
            p.ws();
            if p.eat(',') {
                continue;
            }
            if !p.eat(')') {
                return Err("expected ',' or ')' in arguments".into());
            }
            break;
        }
        calls.push((name, serde_json::Value::Object(args)));
        p.ws();
        if p.eat(',') {
            continue;
        }
        if bracketed {
            p.ws();
            if p.eat(']') {
                break;
            }
            return Err("expected ',' or ']' after a call".into());
        }
    }
    Ok(calls)
}

struct Parser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Parser<'a> {
    fn at_end(&self) -> bool {
        self.i >= self.s.len()
    }

    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn ws(&mut self) {
        while let Some(c) = self.rest().chars().next()
            && c.is_whitespace()
        {
            self.i += c.len_utf8();
        }
    }

    fn eat(&mut self, c: char) -> bool {
        if self.rest().starts_with(c) {
            self.i += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        let start = self.i;
        while let Some(c) = self.rest().chars().next()
            && (c.is_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            self.i += c.len_utf8();
        }
        if start == self.i {
            return Err(format!(
                "expected a name at {:?}",
                self.rest().chars().take(10).collect::<String>()
            ));
        }
        Ok(self.s[start..self.i].to_string())
    }

    fn value(&mut self) -> Result<serde_json::Value, String> {
        let rest = self.rest();
        let Some(c) = rest.chars().next() else {
            return Err("expected a value".into());
        };
        match c {
            '\'' | '"' => self.string(c),
            '{' | '[' => {
                // A JSON literal, as the template's `tojson` writes them.
                let mut it =
                    serde_json::Deserializer::from_str(rest).into_iter::<serde_json::Value>();
                match it.next() {
                    Some(Ok(v)) => {
                        self.i += it.byte_offset();
                        Ok(v)
                    }
                    _ => Err("bad JSON literal".into()),
                }
            }
            _ => {
                let start = self.i;
                while let Some(c) = self.rest().chars().next()
                    && !matches!(c, ',' | ')' | ']')
                    && !c.is_whitespace()
                {
                    self.i += c.len_utf8();
                }
                let word = &self.s[start..self.i];
                Ok(match word {
                    "True" | "true" => serde_json::Value::Bool(true),
                    "False" | "false" => serde_json::Value::Bool(false),
                    "None" | "null" => serde_json::Value::Null,
                    _ => {
                        if let Ok(n) = word.parse::<i64>() {
                            serde_json::Value::from(n)
                        } else if let Ok(f) = word.parse::<f64>() {
                            serde_json::Value::from(f)
                        } else {
                            serde_json::Value::String(word.to_string())
                        }
                    }
                })
            }
        }
    }

    fn string(&mut self, quote: char) -> Result<serde_json::Value, String> {
        self.i += 1;
        let mut out = String::new();
        let mut chars = self.rest().char_indices();
        while let Some((off, c)) = chars.next() {
            match c {
                '\\' => {
                    let Some((_, e)) = chars.next() else { break };
                    out.push(match e {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        other => other,
                    });
                }
                c if c == quote => {
                    self.i += off + 1;
                    return Ok(serde_json::Value::String(out));
                }
                c => out.push(c),
            }
        }
        Err("unterminated string".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lfm() -> OutputFormat {
        OutputFormat {
            think_open: None,
            think_close: Some("</think>"),
            tool_start: "<|tool_call_start|>",
            tool_end: "<|tool_call_end|>",
            syntax: ToolSyntax::PythonCall,
        }
    }

    fn qwen() -> OutputFormat {
        OutputFormat {
            think_open: Some("<think>"),
            think_close: Some("</think>"),
            tool_start: "<tool_call>",
            tool_end: "</tool_call>",
            syntax: ToolSyntax::Json,
        }
    }

    #[test]
    fn parses_python_style_calls() {
        let calls = parse_calls(r#"[get_weather(city='Paris', unit="celsius")]"#).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1["city"], "Paris");
        assert_eq!(calls[0].1["unit"], "celsius");

        let calls = parse_calls(
            r#"[a(n=3, f=2.5, ok=True, nothing=None, obj={"k": [1, 2]}, s='it\'s\nfine'), b()]"#,
        )
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].1["n"], 3);
        assert_eq!(calls[0].1["f"], 2.5);
        assert_eq!(calls[0].1["ok"], true);
        assert!(calls[0].1["nothing"].is_null());
        assert_eq!(calls[0].1["obj"]["k"][1], 2);
        assert_eq!(calls[0].1["s"], "it's\nfine");
        assert_eq!(calls[1].0, "b");
        assert!(calls[1].1.as_object().unwrap().is_empty());

        assert!(parse_calls("[get_weather(city=").is_err());
        assert!(parse_calls("just text").is_err());
    }

    #[test]
    fn splits_reasoning_content_and_tool_calls() {
        let mut p = OutputParser::new(lfm(), "req");
        let mut pieces = Vec::new();
        for d in [
            "The user",
            " wants weather.",
            "</think>",
            "\n\n",
            "Sure.",
            "<|tool_call_start|>",
            "[get_weather(",
            "city='Paris')]",
            "<|tool_call_end|>",
            " done",
        ] {
            pieces.extend(p.push(d));
        }
        pieces.extend(p.finish());
        assert_eq!(pieces[0], Piece::Reasoning("The user".into()));
        assert_eq!(pieces[1], Piece::Reasoning(" wants weather.".into()));
        assert_eq!(pieces[2], Piece::Content("Sure.".into()));
        match &pieces[3] {
            Piece::ToolCalls(c) => {
                assert_eq!(c.len(), 1);
                assert_eq!(c[0].function.name, "get_weather");
                assert_eq!(c[0].function.arguments, r#"{"city":"Paris"}"#);
                assert!(c[0].id.starts_with("call_"));
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
        assert_eq!(pieces[4], Piece::Content(" done".into()));
        assert!(p.saw_tool_calls());
    }

    #[test]
    fn a_model_that_opens_its_own_reasoning_block_is_split_the_same_way() {
        let mut p = OutputParser::new(qwen(), "req");
        let mut pieces = Vec::new();
        for d in [
            "<think>",
            "Let me",
            " check.",
            "</think>",
            "\n\n",
            "Sure. ",
            "<tool_call>",
            "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n",
            "</tool_call>",
        ] {
            pieces.extend(p.push(d));
        }
        pieces.extend(p.finish());
        assert_eq!(pieces[0], Piece::Reasoning("Let me".into()));
        assert_eq!(pieces[1], Piece::Reasoning(" check.".into()));
        assert_eq!(pieces[2], Piece::Content("Sure. ".into()));
        match &pieces[3] {
            Piece::ToolCalls(c) => {
                assert_eq!(c[0].function.name, "get_weather");
                assert_eq!(c[0].function.arguments, r#"{"city":"Oslo"}"#);
            }
            other => panic!("expected a tool call, got {other:?}"),
        }
        assert!(p.saw_tool_calls());
    }

    #[test]
    fn json_tool_calls_take_arguments_as_an_object_or_a_string() {
        let calls = parse_json_call(r#"{"name": "f", "arguments": {"a": 1}}"#).unwrap();
        assert_eq!(calls[0].1["a"], 1);
        let calls = parse_json_call(r#"{"name": "f", "arguments": "{\"a\": 2}"}"#).unwrap();
        assert_eq!(calls[0].1["a"], 2);
        let calls = parse_json_call(r#"{"name": "f"}"#).unwrap();
        assert!(calls[0].1.as_object().unwrap().is_empty());
        assert!(parse_json_call("not json").is_err());
        assert!(parse_json_call(r#"{"arguments": {}}"#).is_err());
    }

    #[test]
    fn a_broken_tool_call_is_kept_as_text() {
        let mut p = OutputParser::new(lfm(), "req");
        let mut pieces = p.push("</think>\nx<|tool_call_start|>oops(<|tool_call_end|>");
        assert_eq!(pieces.remove(0), Piece::Content("x".into()));
        assert_eq!(
            pieces.remove(0),
            Piece::Content("<|tool_call_start|>oops(<|tool_call_end|>".into())
        );
        assert!(!p.saw_tool_calls());
        let mut p = OutputParser::new(lfm(), "req");
        let _ = p.push("</think><|tool_call_start|>[a(");
        assert_eq!(
            p.finish(),
            vec![Piece::Content("<|tool_call_start|>[a(".into())]
        );
    }

    #[test]
    fn call_ids_are_stable_per_request_and_position() {
        let a = OutputParser::new(lfm(), "r1").call_id(0);
        let b = OutputParser::new(lfm(), "r1").call_id(0);
        let c = OutputParser::new(lfm(), "r2").call_id(0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
