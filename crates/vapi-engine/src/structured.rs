//! Constrained decoding: the answer has to parse.
//!
//! A `response_format` turns generation into a walk over a grammar. This
//! module holds the grammar (a JSON-schema subset, flattened into an arena)
//! and a machine that consumes the answer one character at a time and says
//! whether it is still on a path to a valid document. The sampler asks the
//! machine which tokens are still possible and masks the rest, so an
//! invalid character is never sampled rather than being caught afterwards.
//!
//! The machine is cheap to clone, because deciding whether a token is
//! allowed means feeding its text to a copy and seeing whether the copy
//! survives. State is a small stack of frames holding arena indices, so a
//! clone is a short `Vec` copy and no schema is ever duplicated.
//!
//! What the subset covers: `object` with `properties`, `required` and
//! `additionalProperties`, `array` with `items`, `string`, `number`,
//! `integer`, `boolean`, `null`, `enum` and `const`, and a schema with no
//! `type` at all (any JSON value). Anything else is rejected when the
//! schema is compiled, so a client gets a 400 rather than an unconstrained
//! generation it believes was constrained.

use std::collections::HashMap;

/// A compiled schema: nodes in an arena, root at index 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    nodes: Vec<Node>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    /// Any JSON value.
    Any,
    Object {
        /// Property name to node, in declaration order.
        properties: Vec<(String, u32)>,
        /// Indices into `properties` that must appear.
        required: Vec<usize>,
        /// Whether names outside `properties` are allowed.
        additional: bool,
    },
    Array {
        items: u32,
    },
    Str,
    Number {
        integer: bool,
    },
    Bool,
    Null,
    /// `enum` or `const`: the value must be one of these, written exactly
    /// as canonical JSON.
    Choice {
        options: Vec<String>,
    },
}

impl Schema {
    /// The schema that accepts any JSON document.
    pub fn any() -> Self {
        Self {
            nodes: vec![Node::Any],
        }
    }

    /// Compile a JSON Schema document. `Err` names the unsupported part.
    pub fn compile(schema: &serde_json::Value) -> Result<Self, String> {
        let mut nodes = Vec::new();
        let root = compile_node(schema, &mut nodes, 0)?;
        // `compile_node` appends; make sure the root is index 0 for clarity.
        if root != 0 {
            nodes.swap(0, root as usize);
            remap(&mut nodes, 0, root);
        }
        Ok(Self { nodes })
    }
}

/// Swap two node indices everywhere they are referenced.
fn remap(nodes: &mut [Node], a: u32, b: u32) {
    let fix = |i: &mut u32| {
        if *i == a {
            *i = b;
        } else if *i == b {
            *i = a;
        }
    };
    for n in nodes.iter_mut() {
        match n {
            Node::Object { properties, .. } => {
                for (_, i) in properties.iter_mut() {
                    fix(i);
                }
            }
            Node::Array { items } => fix(items),
            _ => {}
        }
    }
}

const MAX_DEPTH: usize = 32;

fn compile_node(
    schema: &serde_json::Value,
    nodes: &mut Vec<Node>,
    depth: usize,
) -> Result<u32, String> {
    if depth > MAX_DEPTH {
        return Err(format!("schema nests deeper than {MAX_DEPTH} levels"));
    }
    let obj = match schema {
        // `true` is "anything", which JSON Schema allows in place of {}.
        serde_json::Value::Bool(true) => {
            nodes.push(Node::Any);
            return Ok(nodes.len() as u32 - 1);
        }
        serde_json::Value::Object(o) => o,
        other => return Err(format!("a schema must be an object, got {other}")),
    };

    for key in obj.keys() {
        const SUPPORTED: &[&str] = &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "const",
            "title",
            "description",
            "$schema",
            "default",
            "examples",
        ];
        if !SUPPORTED.contains(&key.as_str()) {
            return Err(format!(
                "schema keyword {key:?} is not supported; supported: {}",
                SUPPORTED[..7].join(", ")
            ));
        }
    }

    // `const` and `enum` pin the value outright, whatever its type says.
    if let Some(c) = obj.get("const") {
        nodes.push(Node::Choice {
            options: vec![canonical(c)],
        });
        return Ok(nodes.len() as u32 - 1);
    }
    if let Some(e) = obj.get("enum") {
        let options = e
            .as_array()
            .ok_or_else(|| "enum must be an array".to_string())?
            .iter()
            .map(canonical)
            .collect::<Vec<_>>();
        if options.is_empty() {
            return Err("enum must have at least one value".into());
        }
        nodes.push(Node::Choice { options });
        return Ok(nodes.len() as u32 - 1);
    }

    let ty = match obj.get("type") {
        None => {
            nodes.push(Node::Any);
            return Ok(nodes.len() as u32 - 1);
        }
        Some(serde_json::Value::String(s)) => s.as_str(),
        Some(serde_json::Value::Array(_)) => {
            return Err("a list of types is not supported; use one type".into());
        }
        Some(other) => return Err(format!("type must be a string, got {other}")),
    };

    let node = match ty {
        "string" => Node::Str,
        "integer" => Node::Number { integer: true },
        "number" => Node::Number { integer: false },
        "boolean" => Node::Bool,
        "null" => Node::Null,
        "array" => {
            let items = match obj.get("items") {
                None => {
                    nodes.push(Node::Any);
                    nodes.len() as u32 - 1
                }
                Some(s) => compile_node(s, nodes, depth + 1)?,
            };
            Node::Array { items }
        }
        "object" => {
            let mut properties = Vec::new();
            if let Some(props) = obj.get("properties") {
                let props = props
                    .as_object()
                    .ok_or_else(|| "properties must be an object".to_string())?;
                for (name, sub) in props {
                    let id = compile_node(sub, nodes, depth + 1)?;
                    properties.push((name.clone(), id));
                }
            }
            let required: Vec<usize> = match obj.get("required") {
                None => Vec::new(),
                Some(r) => {
                    let names = r
                        .as_array()
                        .ok_or_else(|| "required must be an array".to_string())?;
                    let mut out = Vec::new();
                    for n in names {
                        let n = n
                            .as_str()
                            .ok_or_else(|| "required entries must be strings".to_string())?;
                        let i = properties
                            .iter()
                            .position(|(p, _)| p == n)
                            .ok_or_else(|| format!("required property {n:?} has no schema"))?;
                        out.push(i);
                    }
                    out
                }
            };
            // Defaulting to false is the opposite of JSON Schema, on
            // purpose: a constrained generation with free-form extra keys
            // is barely constrained, and a caller who wants them can say so.
            let additional = match obj.get("additionalProperties") {
                None => properties.is_empty(),
                Some(serde_json::Value::Bool(b)) => *b,
                Some(_) => return Err("additionalProperties must be true or false".into()),
            };
            if properties.is_empty() && !additional {
                return Err(
                    "an object with no properties and no additionalProperties can only be {}"
                        .into(),
                );
            }
            Node::Object {
                properties,
                required,
                additional,
            }
        }
        other => return Err(format!("unsupported type {other:?}")),
    };
    nodes.push(node);
    Ok(nodes.len() as u32 - 1)
}

/// A JSON value written the one way the machine will accept it.
fn canonical(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

// --------------------------------------------------------------- machine

/// Where the machine is inside one value.
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    /// About to read a value of this node.
    Value(u32),
    /// Inside a string of this node; `escape` counts remaining escape
    /// characters (1 after a backslash, 4 more for `\u`).
    Str { escape: u8, len: u32 },
    /// Inside a number. The flags track what may still come.
    Number {
        integer: bool,
        seen_digit: bool,
        seen_dot: bool,
        seen_exp: bool,
        exp_digit: bool,
        after_sign: bool,
        /// Characters written so far, bounded by [`MAX_NUMBER_CHARS`].
        len: u16,
    },
    /// Matching a fixed word: `true`, `false`, `null`, or one option of a
    /// `Choice`. `alts` are the still-possible options and `at` how much of
    /// them has matched.
    Word { alts: Vec<String>, at: usize },
    /// Inside an object.
    Object {
        node: u32,
        state: ObjState,
        /// Property indices already seen.
        seen: Vec<usize>,
        /// The key being read, when one is.
        key: String,
        /// The key's string is escaped-in-progress.
        escape: u8,
    },
    /// Inside an array.
    Array { items: u32, state: ArrState },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjState {
    /// After `{`: expecting a key or `}`.
    Start,
    /// Reading a key's characters.
    Key,
    /// Key closed, expecting `:`.
    Colon,
    /// After `:`: the value frame is pushed on top of this one.
    Value,
    /// Value done: expecting `,` or `}`.
    Next,
    /// After `,`: expecting a key.
    KeyExpected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrState {
    /// After `[`: expecting a value or `]`.
    Start,
    /// A value frame is on top.
    Value,
    /// Value done: expecting `,` or `]`.
    Next,
    /// After `,`: expecting a value.
    ValueExpected,
}

/// Incremental validator for one sequence.
///
/// Feed it the text of each sampled token with [`push`](Self::push); ask
/// whether a candidate would still be valid with [`allows`](Self::allows).
#[derive(Clone, Debug)]
pub struct Machine {
    schema: std::sync::Arc<Schema>,
    stack: Vec<Frame>,
    /// A complete document has been read; only whitespace may follow.
    done: bool,
    /// Whether any non-whitespace character has been read. The document
    /// starts at the first character, so leading whitespace is refused
    /// rather than letting a model fill its budget with spaces.
    started: bool,
    /// Length of the current whitespace run, capped so indentation stays
    /// legal but a run of it cannot go on forever.
    ws_run: u8,
}

/// Longest run of whitespace the machine will accept.
const MAX_WS_RUN: u8 = 16;

/// Longest number the machine will accept, in characters.
///
/// Every digit is legal JSON, so a model that falls into a repetition loop
/// can spend an entire budget on one integer and never close the document.
/// This is far longer than any number a caller wants and short enough that
/// the loop ends.
const MAX_NUMBER_CHARS: u16 = 32;

/// Longest string the machine will accept, in characters. Generous: it is
/// there to stop a loop, not to constrain content.
const MAX_STRING_CHARS: u32 = 8192;

/// Whitespace as JSON defines it.
///
/// Not `char::is_ascii_whitespace`, which also accepts a form feed: the
/// model emitted one between two values, this machine let it through, and
/// the answer then failed to parse in the caller's JSON library. A
/// constraint that accepts more than the format does is worse than none,
/// because it is trusted.
fn is_json_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl Machine {
    pub fn new(schema: std::sync::Arc<Schema>) -> Self {
        Self {
            stack: vec![Frame::Value(0)],
            schema,
            done: false,
            started: false,
            ws_run: 0,
        }
    }

    /// Whether a complete, valid document has been read.
    ///
    /// A bare number is the one value that ends only because something
    /// else begins, so a top-level number with the digits it needs counts
    /// as complete even though nothing has closed it.
    pub fn complete(&self) -> bool {
        if self.done {
            return true;
        }
        match self.stack.as_slice() {
            [
                Frame::Number {
                    seen_digit,
                    seen_exp,
                    exp_digit,
                    ..
                },
            ] => *seen_digit && (!*seen_exp || *exp_digit),
            _ => false,
        }
    }

    /// Feed one character. `false` means the document can no longer be
    /// valid, and the machine is left in an unspecified state.
    pub fn push_char(&mut self, c: char) -> bool {
        // Whitespace is structural nowhere: bound it so a constrained
        // generation cannot spend its budget on it, and refuse it before
        // the document has started at all.
        if is_json_ws(c) && !self.in_string_body() {
            if !self.started || self.ws_run >= MAX_WS_RUN {
                return false;
            }
            self.ws_run += 1;
        } else {
            self.ws_run = 0;
            self.started = true;
        }
        if self.done {
            return is_json_ws(c);
        }
        let Some(frame) = self.stack.last().cloned() else {
            return false;
        };
        match frame {
            Frame::Value(node) => self.start_value(node, c),
            Frame::Str { escape, .. } => self.in_string(escape, c),
            Frame::Number { .. } => self.in_number(c),
            Frame::Word { .. } => self.in_word(c),
            Frame::Object { .. } => self.in_object(c),
            Frame::Array { .. } => self.in_array(c),
        }
    }

    /// Feed a whole string, stopping at the first character that cannot be
    /// valid.
    pub fn push(&mut self, text: &str) -> bool {
        for c in text.chars() {
            if !self.push_char(c) {
                return false;
            }
        }
        true
    }

    /// Whether `text` could follow what has been read, and still lead
    /// somewhere. Does not advance.
    pub fn allows(&self, text: &str) -> bool {
        let mut probe = self.clone();
        probe.push(text) && probe.viable()
    }

    /// How many containers are still open.
    ///
    /// Scalars are not counted: a number ends when the next character
    /// arrives, so including it would make a comma look like progress
    /// towards closing the document when it opens another element.
    pub fn depth(&self) -> usize {
        self.stack
            .iter()
            .filter(|f| matches!(f, Frame::Object { .. } | Frame::Array { .. }))
            .count()
    }

    /// Whether the document would be complete after `text`.
    pub fn completes_with(&self, text: &str) -> bool {
        let mut probe = self.clone();
        probe.push(text) && probe.complete()
    }

    /// Inside a string's contents, where whitespace is just a character.
    fn in_string_body(&self) -> bool {
        matches!(
            self.stack.last(),
            Some(Frame::Str { .. })
                | Some(Frame::Object {
                    state: ObjState::Key,
                    ..
                })
        )
    }

    /// Can a valid document still be reached from here?
    ///
    /// A prefix can be valid and yet doomed: a comma inside an object whose
    /// every property has already appeared leaves nothing that could follow.
    /// The sampler asks this as well as validity, so the model is never
    /// offered a token that paints it into a corner.
    pub fn viable(&self) -> bool {
        if self.done {
            return true;
        }
        self.stack.iter().all(|f| self.frame_viable(f))
    }

    fn frame_viable(&self, frame: &Frame) -> bool {
        match frame {
            Frame::Word { alts, .. } => !alts.is_empty(),
            Frame::Object {
                node,
                state,
                seen,
                key,
                ..
            } => {
                if *node == u32::MAX {
                    return true;
                }
                let Node::Object {
                    properties,
                    required,
                    additional,
                } = self.node(*node)
                else {
                    return false;
                };
                let more_possible =
                    *additional || (0..properties.len()).any(|i| !seen.contains(&i));
                let can_close = required.iter().all(|r| seen.contains(r));
                match state {
                    ObjState::Start | ObjState::Next => can_close || more_possible,
                    ObjState::KeyExpected => more_possible,
                    // A key in progress has to be one that can still be
                    // completed. Checking it as each character arrives is
                    // not enough: a single token can be `,"`, which opens a
                    // key in one step, and an object with every property
                    // used has nowhere to go from there.
                    ObjState::Key => {
                        *additional
                            || properties.iter().enumerate().any(|(i, (name, _))| {
                                !seen.contains(&i) && name.starts_with(key.as_str())
                            })
                    }
                    ObjState::Colon | ObjState::Value => true,
                }
            }
            // Strings, numbers and arrays can always be finished.
            _ => true,
        }
    }

    fn node(&self, id: u32) -> &Node {
        &self.schema.nodes[id as usize]
    }

    /// One value finished: pop it and tell the frame underneath.
    fn finish_value(&mut self) -> bool {
        self.stack.pop();
        match self.stack.last_mut() {
            None => {
                self.done = true;
                true
            }
            Some(Frame::Object { state, .. }) => {
                *state = ObjState::Next;
                true
            }
            Some(Frame::Array { state, .. }) => {
                *state = ArrState::Next;
                true
            }
            // A value frame under a value frame cannot happen: containers
            // are the only things that push.
            Some(_) => false,
        }
    }

    fn start_value(&mut self, node: u32, c: char) -> bool {
        if is_json_ws(c) {
            return true;
        }
        let allowed = |n: &Node| -> Option<Frame> {
            match (n, c) {
                (Node::Any | Node::Str, '"') => Some(Frame::Str { escape: 0, len: 0 }),
                (Node::Any, '{') => Some(Frame::Object {
                    node: u32::MAX,
                    state: ObjState::Start,
                    seen: Vec::new(),
                    key: String::new(),
                    escape: 0,
                }),
                (Node::Any, '[') => Some(Frame::Array {
                    items: u32::MAX,
                    state: ArrState::Start,
                }),
                (Node::Any | Node::Number { .. }, '-' | '0'..='9') => Some(Frame::Number {
                    integer: matches!(n, Node::Number { integer: true }),
                    seen_digit: c != '-',
                    seen_dot: false,
                    seen_exp: false,
                    exp_digit: false,
                    after_sign: c == '-',
                    len: 1,
                }),
                (Node::Any | Node::Bool, 't' | 'f') => Some(Frame::Word {
                    alts: vec![if c == 't' {
                        "true".into()
                    } else {
                        "false".into()
                    }],
                    at: 1,
                }),
                (Node::Any | Node::Null, 'n') => Some(Frame::Word {
                    alts: vec!["null".into()],
                    at: 1,
                }),
                _ => None,
            }
        };

        let frame = match self.node(node).clone() {
            Node::Object {
                properties,
                additional,
                ..
            } if c == '{' => {
                let _ = (properties, additional);
                Frame::Object {
                    node,
                    state: ObjState::Start,
                    seen: Vec::new(),
                    key: String::new(),
                    escape: 0,
                }
            }
            Node::Array { items } if c == '[' => Frame::Array {
                items,
                state: ArrState::Start,
            },
            Node::Choice { options } => {
                let alts: Vec<String> = options.into_iter().filter(|o| o.starts_with(c)).collect();
                if alts.is_empty() {
                    return false;
                }
                Frame::Word { alts, at: 1 }
            }
            other => match allowed(&other) {
                Some(f) => f,
                None => return false,
            },
        };
        // A one-character word (there are none in JSON) would finish here.
        *self.stack.last_mut().expect("value frame") = frame;
        self.settle_word()
    }

    /// A `Word` frame that has matched a whole option is a finished value.
    fn settle_word(&mut self) -> bool {
        let finished = match self.stack.last() {
            Some(Frame::Word { alts, at }) => alts.iter().any(|a| a.len() == *at),
            _ => false,
        };
        if finished {
            // Ambiguity between one option and a longer one starting with
            // it cannot arise in JSON literals or canonical values, so
            // finishing at the first complete match is safe.
            return self.finish_value();
        }
        true
    }

    fn in_string(&mut self, escape: u8, c: char) -> bool {
        let Some(Frame::Str { escape: e, len }) = self.stack.last_mut() else {
            return false;
        };
        // Past the cap the only character left is the one that ends it.
        if *len >= MAX_STRING_CHARS && c != '"' {
            return false;
        }
        *len += 1;
        if escape > 0 {
            // Inside an escape: `\uXXXX` needs hex, everything else is one
            // character from a fixed set.
            let ok = if escape == 5 {
                matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')
            } else {
                c.is_ascii_hexdigit()
            };
            if !ok {
                return false;
            }
            *e = if escape == 5 && c == 'u' {
                4
            } else {
                escape - 1
            };
            if escape == 5 && c != 'u' {
                *e = 0;
            }
            return true;
        }
        match c {
            '"' => self.finish_value(),
            '\\' => {
                *e = 5;
                true
            }
            // Control characters must be escaped in JSON.
            c if (c as u32) < 0x20 => false,
            _ => true,
        }
    }

    fn in_number(&mut self, c: char) -> bool {
        let Some(Frame::Number {
            integer,
            seen_digit,
            seen_dot,
            seen_exp,
            exp_digit,
            after_sign,
            len,
        }) = self.stack.last_mut()
        else {
            return false;
        };
        // Past the cap only the characters that end the number are left.
        let at_cap = *len >= MAX_NUMBER_CHARS;
        *len += 1;
        if at_cap && matches!(c, '0'..='9' | '.' | 'e' | 'E' | '+' | '-') {
            return false;
        }
        match c {
            '0'..='9' => {
                if *seen_exp {
                    *exp_digit = true;
                } else {
                    *seen_digit = true;
                }
                *after_sign = false;
                true
            }
            '.' if !*integer && !*seen_dot && !*seen_exp && *seen_digit => {
                *seen_dot = true;
                *seen_digit = false;
                true
            }
            'e' | 'E' if !*integer && !*seen_exp && *seen_digit => {
                *seen_exp = true;
                *after_sign = false;
                true
            }
            '+' | '-' if *seen_exp && !*exp_digit && !*after_sign => {
                *after_sign = true;
                true
            }
            // Anything else ends the number, if it is a complete one.
            _ => {
                let complete = *seen_digit && (!*seen_exp || *exp_digit);
                if !complete {
                    return false;
                }
                if !self.finish_value() {
                    return false;
                }
                // The character belongs to whatever comes next.
                self.push_char(c)
            }
        }
    }

    fn in_word(&mut self, c: char) -> bool {
        let Some(Frame::Word { alts, at }) = self.stack.last_mut() else {
            return false;
        };
        let pos = *at;
        alts.retain(|a| a.as_bytes().get(pos).map(|b| *b as char) == Some(c));
        if alts.is_empty() {
            return false;
        }
        *at += 1;
        self.settle_word()
    }

    fn in_object(&mut self, c: char) -> bool {
        // Read the node id first: deciding what a key may contain needs the
        // schema, which cannot be borrowed while the frame is.
        let node = match self.stack.last() {
            Some(Frame::Object { node, .. }) => *node,
            _ => return false,
        };
        let keys_are_free = self.keys_are_free(node);
        let Some(Frame::Object {
            state,
            seen,
            key,
            escape,
            ..
        }) = self.stack.last_mut()
        else {
            return false;
        };
        match *state {
            ObjState::Start | ObjState::KeyExpected => {
                if is_json_ws(c) {
                    return true;
                }
                if c == '}' && *state == ObjState::Start {
                    // Only if nothing is still required.
                    let seen = seen.clone();
                    return self.close_object(node, &seen);
                }
                if c != '"' {
                    return false;
                }
                key.clear();
                *escape = 0;
                *state = ObjState::Key;
                true
            }
            ObjState::Key => {
                if *escape > 0 {
                    let ok = if *escape == 5 {
                        matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')
                    } else {
                        c.is_ascii_hexdigit()
                    };
                    if !ok {
                        return false;
                    }
                    let next = if *escape == 5 && c == 'u' {
                        4
                    } else if *escape == 5 {
                        0
                    } else {
                        *escape - 1
                    };
                    *escape = next;
                    key.push(c);
                    return true;
                }
                match c {
                    '"' => {
                        let done = key.clone();
                        *state = ObjState::Colon;
                        self.check_key(node, &done)
                    }
                    // An escape in a key is only useful when any key is
                    // allowed. With a fixed property set it is a way to
                    // write characters that no property name contains, so
                    // it is refused rather than left to run on.
                    '\\' if keys_are_free => {
                        *escape = 5;
                        key.push(c);
                        true
                    }
                    '\\' => false,
                    c if (c as u32) < 0x20 => false,
                    _ => {
                        key.push(c);
                        // Reject as soon as no property can match, so the
                        // sampler cannot wander down a dead key.
                        let partial = key.clone();
                        self.key_possible(node, &partial)
                    }
                }
            }
            ObjState::Colon => {
                if is_json_ws(c) {
                    return true;
                }
                if c != ':' {
                    return false;
                }
                let key = key.clone();
                let value_node = self.value_node_for(node, &key);
                let Some(Frame::Object { state, .. }) = self.stack.last_mut() else {
                    return false;
                };
                *state = ObjState::Value;
                match value_node {
                    Some(n) => {
                        self.stack.push(Frame::Value(n));
                        true
                    }
                    None => false,
                }
            }
            ObjState::Value => false, // the value frame is on top
            ObjState::Next => {
                if is_json_ws(c) {
                    return true;
                }
                match c {
                    ',' => {
                        *state = ObjState::KeyExpected;
                        true
                    }
                    '}' => {
                        let seen = seen.clone();
                        self.close_object(node, &seen)
                    }
                    _ => false,
                }
            }
        }
    }

    /// `}` is only allowed when every required property has appeared.
    fn close_object(&mut self, node: u32, seen: &[usize]) -> bool {
        if node != u32::MAX
            && let Node::Object { required, .. } = self.node(node)
            && !required.iter().all(|r| seen.contains(r))
        {
            return false;
        }
        self.finish_value()
    }

    /// Whether this object accepts property names outside its schema.
    fn keys_are_free(&self, node: u32) -> bool {
        if node == u32::MAX {
            return true;
        }
        matches!(
            self.node(node),
            Node::Object {
                additional: true,
                ..
            }
        )
    }

    /// Could any allowed property name still start with `partial`?
    fn key_possible(&mut self, node: u32, partial: &str) -> bool {
        if node == u32::MAX {
            return true;
        }
        let Node::Object {
            properties,
            additional,
            ..
        } = self.node(node)
        else {
            return false;
        };
        if *additional {
            return true;
        }
        let seen: Vec<usize> = match self.stack.last() {
            Some(Frame::Object { seen, .. }) => seen.clone(),
            _ => Vec::new(),
        };
        properties
            .iter()
            .enumerate()
            .any(|(i, (name, _))| !seen.contains(&i) && name.starts_with(partial))
    }

    /// A key just closed: record it, and reject an unknown or repeated one.
    fn check_key(&mut self, node: u32, key: &str) -> bool {
        if node == u32::MAX {
            return true;
        }
        let Node::Object {
            properties,
            additional,
            ..
        } = self.node(node).clone()
        else {
            return false;
        };
        match properties.iter().position(|(name, _)| name == key) {
            Some(i) => {
                let Some(Frame::Object { seen, .. }) = self.stack.last_mut() else {
                    return false;
                };
                if seen.contains(&i) {
                    return false;
                }
                seen.push(i);
                true
            }
            None => additional,
        }
    }

    /// The node a property's value must satisfy.
    fn value_node_for(&mut self, node: u32, key: &str) -> Option<u32> {
        if node == u32::MAX {
            return Some(self.any_node());
        }
        let Node::Object {
            properties,
            additional,
            ..
        } = self.node(node).clone()
        else {
            return None;
        };
        match properties.iter().find(|(name, _)| name == key) {
            Some((_, id)) => Some(*id),
            None if additional => Some(self.any_node()),
            None => None,
        }
    }

    /// Index of an `Any` node, appending one if the schema has none. The
    /// schema is shared, so this returns an index into the existing arena;
    /// every compiled schema that can reach `Any` already contains one.
    fn any_node(&self) -> u32 {
        self.schema
            .nodes
            .iter()
            .position(|n| matches!(n, Node::Any))
            .map(|i| i as u32)
            .unwrap_or(u32::MAX)
    }

    fn in_array(&mut self, c: char) -> bool {
        let Some(Frame::Array { items, state }) = self.stack.last_mut() else {
            return false;
        };
        let items = *items;
        match *state {
            ArrState::Start | ArrState::ValueExpected => {
                if is_json_ws(c) {
                    return true;
                }
                if c == ']' && *state == ArrState::Start {
                    return self.finish_value();
                }
                *state = ArrState::Value;
                let node = if items == u32::MAX {
                    self.any_node()
                } else {
                    items
                };
                if node == u32::MAX {
                    return false;
                }
                self.stack.push(Frame::Value(node));
                self.push_char(c)
            }
            ArrState::Value => false,
            ArrState::Next => {
                if is_json_ws(c) {
                    return true;
                }
                match c {
                    ',' => {
                        *state = ArrState::ValueExpected;
                        true
                    }
                    ']' => self.finish_value(),
                    _ => false,
                }
            }
        }
    }
}

/// An `Any` node must exist for free-form values to be reachable.
///
/// A schema compiled from `{"type": "object", "additionalProperties": true}`
/// has no `Any` of its own, so one is appended when the schema is built.
pub fn with_any_node(schema: Schema) -> Schema {
    let mut nodes = schema.nodes;
    if !nodes.iter().any(|n| matches!(n, Node::Any)) {
        nodes.push(Node::Any);
    }
    Schema { nodes }
}

/// Which token ids a machine still allows, computed by trying them.
///
/// The vocabulary is bucketed by a token's first byte once, so a step only
/// tries the tokens that could possibly start the next character rather
/// than all 128K of them.
pub struct TokenMasker {
    /// Token id to its text; empty for tokens with no printable form.
    texts: Vec<String>,
    /// First byte to the token ids starting with it.
    by_first: HashMap<u8, Vec<u32>>,
}

impl TokenMasker {
    /// `texts` is the vocabulary, indexed by token id.
    pub fn new(texts: Vec<String>) -> Self {
        let mut by_first: HashMap<u8, Vec<u32>> = HashMap::new();
        for (id, t) in texts.iter().enumerate() {
            if let Some(b) = t.as_bytes().first() {
                by_first.entry(*b).or_default().push(id as u32);
            }
        }
        Self { texts, by_first }
    }

    pub fn vocab_size(&self) -> usize {
        self.texts.len()
    }

    /// Set every logit the machine forbids to negative infinity.
    ///
    /// `eos` is allowed exactly when the document is already complete,
    /// which is how generation is made to stop at the closing brace rather
    /// than rambling after it.
    pub fn mask(&self, machine: &Machine, eos: &[u32], logits: &mut [f32]) {
        let complete = machine.complete();
        // Read the EOS logits before masking, since the mask below would
        // otherwise overwrite them and leave nothing to restore.
        let eos_logits: Vec<(usize, f32)> = eos
            .iter()
            .filter_map(|&e| logits.get(e as usize).map(|l| (e as usize, *l)))
            .collect();
        if complete {
            // The document is closed: the only thing left to say is
            // nothing. Trailing whitespace would just be noise in the
            // answer the caller parses.
            for l in logits.iter_mut() {
                *l = f32::NEG_INFINITY;
            }
            for (i, original) in eos_logits {
                logits[i] = original;
            }
            return;
        }
        let mut allowed = vec![false; logits.len()];
        // Only tokens whose first byte can follow are worth trying.
        for (byte, ids) in &self.by_first {
            let c = *byte as char;
            // A multi-byte character's first byte is not a character on its
            // own; try those tokens rather than ruling them out.
            let worth_trying = !byte.is_ascii() || machine.allows(&c.to_string());
            if !worth_trying {
                continue;
            }
            for &id in ids {
                let i = id as usize;
                if i < allowed.len() && machine.allows(&self.texts[i]) {
                    allowed[i] = true;
                }
            }
        }
        if !allowed.iter().any(|a| *a) {
            // Nothing in the vocabulary continues this document. Close what
            // is open instead, and failing that end the turn: leaving the
            // row unmasked hands the model a free hand, which is how an
            // unschema'd key ended up in an answer that was supposed to be
            // constrained.
            metrics::counter!("vapi_constraint_dead_ends_total").increment(1);
            self.mask_closing(machine, logits, &mut allowed);
            if !allowed.iter().any(|a| *a) {
                for l in logits.iter_mut() {
                    *l = f32::NEG_INFINITY;
                }
                for (i, original) in eos_logits {
                    logits[i] = original;
                }
                return;
            }
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !allowed[i] {
                *l = f32::NEG_INFINITY;
            }
        }
        for (i, _) in eos_logits {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

impl TokenMasker {
    /// Fill `allowed` with the tokens that close a container or complete
    /// the document.
    fn mask_closing(&self, machine: &Machine, logits: &[f32], allowed: &mut [bool]) {
        let depth = machine.depth();
        for ids in self.by_first.values() {
            for &id in ids {
                let i = id as usize;
                if i >= logits.len() {
                    continue;
                }
                let mut probe = machine.clone();
                if probe.push(&self.texts[i])
                    && probe.viable()
                    && (probe.depth() < depth || probe.complete())
                {
                    allowed[i] = true;
                }
            }
        }
    }

    /// Like [`mask`](Self::mask), but only tokens that bring the document
    /// to a close are left.
    ///
    /// Used when a request is about to run out of budget: a truncated
    /// document is useless to a caller who asked for a format, so the last
    /// few tokens are spent closing what is open. If no token closes
    /// anything, this falls back to the ordinary mask.
    pub fn mask_finishing(&self, machine: &Machine, eos: &[u32], logits: &mut [f32]) {
        if machine.complete() {
            return self.mask(machine, eos, logits);
        }
        let mut allowed = vec![false; logits.len()];
        self.mask_closing(machine, logits, &mut allowed);
        if !allowed.iter().any(|a| *a) {
            return self.mask(machine, eos, logits);
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !allowed[i] {
                *l = f32::NEG_INFINITY;
            }
        }
        for &e in eos {
            if let Some(l) = logits.get_mut(e as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn machine(schema: serde_json::Value) -> Machine {
        Machine::new(Arc::new(with_any_node(Schema::compile(&schema).unwrap())))
    }

    fn accepts(schema: serde_json::Value, text: &str) -> bool {
        let mut m = machine(schema);
        m.push(text) && m.complete()
    }

    #[test]
    fn free_form_json_accepts_documents_and_rejects_rubbish() {
        let any = || serde_json::json!({});
        for good in [
            "{}",
            "[]",
            "null",
            "true",
            "-12.5e+3",
            r#""hi""#,
            r#"{"a": [1, 2, {"b": null}], "c": "x"}"#,
            "{ \"a\" : 1 }  ",
        ] {
            assert!(accepts(any(), good), "should accept {good}");
        }
        for bad in [
            "{",
            "}",
            "{\"a\"}",
            "{'a': 1}",
            "[1,]",
            "tru",
            "01x",
            "\"unterminated",
            "{\"a\": }",
            "nul",
        ] {
            assert!(!accepts(any(), bad), "should reject {bad}");
        }
    }

    #[test]
    fn a_typed_schema_constrains_shape_and_types() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["city", "temp"],
        });
        assert!(accepts(
            schema.clone(),
            r#"{"city": "Paris", "temp": 18, "tags": ["a", "b"]}"#
        ));
        assert!(accepts(schema.clone(), r#"{"city":"Paris","temp":-3}"#));
        // Missing a required property.
        assert!(!accepts(schema.clone(), r#"{"city": "Paris"}"#));
        // Wrong type.
        assert!(!accepts(schema.clone(), r#"{"city": 1, "temp": 2}"#));
        assert!(!accepts(schema.clone(), r#"{"city": "P", "temp": 1.5}"#));
        // Unknown property, with additionalProperties defaulting to false.
        assert!(!accepts(schema.clone(), r#"{"city":"P","temp":1,"x":1}"#));
        // Repeated property.
        assert!(!accepts(
            schema.clone(),
            r#"{"city":"P","temp":1,"city":"Q"}"#
        ));
        // Array item of the wrong type.
        assert!(!accepts(schema, r#"{"city":"P","temp":1,"tags":[1]}"#));
    }

    #[test]
    fn a_dead_key_is_rejected_at_the_first_impossible_character() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
        });
        let mut m = machine(schema);
        assert!(m.push(r#"{"ci"#));
        // No property starts with "cx", so the character cannot be sampled.
        assert!(!m.allows("x"));
        assert!(m.allows("t"));
    }

    #[test]
    fn enums_and_consts_are_matched_character_by_character() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"unit": {"enum": ["celsius", "fahrenheit"]}},
            "required": ["unit"],
        });
        assert!(accepts(schema.clone(), r#"{"unit": "celsius"}"#));
        assert!(accepts(schema.clone(), r#"{"unit": "fahrenheit"}"#));
        assert!(!accepts(schema.clone(), r#"{"unit": "kelvin"}"#));
        let mut m = machine(schema);
        assert!(m.push(r#"{"unit": "c"#));
        assert!(m.allows("elsius\"}"));
        assert!(!m.allows("x"));

        let c = serde_json::json!({"const": 42});
        assert!(accepts(c.clone(), "42"));
        assert!(!accepts(c, "43"));
    }

    #[test]
    fn completion_is_only_at_a_closed_document() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        });
        let mut m = machine(schema);
        assert!(m.push(r#"{"a": 1"#));
        assert!(!m.complete(), "the object is still open");
        assert!(m.completes_with("}"));
        assert!(m.push("}"));
        assert!(m.complete());
        // Nothing but whitespace may follow.
        assert!(m.allows(" \n"));
        assert!(!m.allows("x"));
    }

    #[test]
    fn unsupported_schemas_are_rejected_with_a_reason() {
        for bad in [
            serde_json::json!({"type": "string", "pattern": "^a"}),
            serde_json::json!({"anyOf": [{"type": "string"}]}),
            serde_json::json!({"type": ["string", "null"]}),
            serde_json::json!({"type": "wat"}),
            serde_json::json!("not an object"),
        ] {
            let err = Schema::compile(&bad).unwrap_err();
            assert!(!err.is_empty(), "{bad} should be rejected");
        }
        // The supported ones compile.
        assert!(Schema::compile(&serde_json::json!({"type": "object", "properties": {}, "additionalProperties": true})).is_ok());
    }

    #[test]
    fn a_key_cannot_be_escaped_into_nonsense() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        });
        let mut m = machine(schema);
        assert!(m.push("{"));
        assert!(m.allows("\""));
        assert!(m.push("\""));
        assert!(!m.allows("\\"), "no escapes where the key set is fixed");
        assert!(!m.allows("z"));
        assert!(m.allows("ok\":true}"));
        // Free-form keys may be escaped.
        let mut m = machine(serde_json::json!({}));
        assert!(m.push("{\""));
        assert!(m.allows("\\u0041"));
    }

    #[test]
    fn a_dead_end_closes_the_document_rather_than_freeing_the_model() {
        // An object whose every property has been used: a comma is valid
        // JSON so far but leads nowhere, so a step that somehow reaches
        // that state must still not let anything through.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        });
        let vocab: Vec<String> = vec![
            "\"b\"".into(), // 0, an unknown key
            "}".into(),     // 1
            " ".into(),     // 2
            "".into(),      // 3, EOS
        ];
        let masker = TokenMasker::new(vocab);
        let mut m = machine(schema);
        assert!(m.push("{\"a\": 1,"));
        assert!(!m.viable(), "nothing can follow that comma");
        let mut logits = vec![0.0f32; 4];
        masker.mask(&m, &[3], &mut logits);
        assert!(logits[0].is_infinite(), "an unknown key stays masked");
        assert!(
            logits[..3].iter().all(|l| l.is_infinite()),
            "nothing can be written: {logits:?}"
        );
        assert_eq!(logits[3], 0.0, "the turn ends instead");
    }

    #[test]
    fn a_finishing_mask_closes_what_is_open() {
        let schema = serde_json::json!({"type": "array", "items": {"type": "integer"}});
        let vocab: Vec<String> = vec![
            "[".into(), // 0
            "1".into(), // 1
            ",".into(), // 2
            "]".into(), // 3
            "".into(),  // 4, EOS
        ];
        let masker = TokenMasker::new(vocab);
        let mut m = machine(schema);
        assert!(m.push("[1, 2"));
        // Normally the array may continue.
        let mut logits = vec![0.0f32; 5];
        masker.mask(&m, &[4], &mut logits);
        assert_eq!(logits[2], 0.0, "a comma is allowed");
        assert_eq!(logits[3], 0.0, "so is the closing bracket");
        // Out of budget: only the bracket.
        let mut logits = vec![0.0f32; 5];
        masker.mask_finishing(&m, &[4], &mut logits);
        assert!(logits[2].is_infinite(), "no more elements");
        assert_eq!(logits[3], 0.0, "close it");
        assert!(logits[4].is_infinite(), "not finished yet");
        // Once closed, the finishing mask is the ordinary one.
        assert!(m.push("]"));
        let mut logits = vec![0.0f32; 5];
        masker.mask_finishing(&m, &[4], &mut logits);
        assert_eq!(logits[4], 0.0, "end the turn");
    }

    #[test]
    fn a_token_that_leads_nowhere_is_not_allowed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        });
        let mut m = machine(schema);
        assert!(m.push(r#"{"a": 1"#));
        // The object can close, and a comma is valid JSON so far, but
        // nothing could follow it: every property has been used.
        assert!(m.allows("}"));
        assert!(!m.allows(","), "a comma here has no legal continuation");
    }

    #[test]
    fn opening_a_key_that_cannot_exist_is_not_allowed() {
        // `,"` arrives as one token in real vocabularies, so the check has
        // to see that the key it opens has nowhere to go.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a"],
        });
        let mut m = machine(schema);
        assert!(m.push("{\"a\": 1"));
        assert!(m.allows(",\""), "b is still free");
        assert!(m.push(",\"b\": 2"));
        assert!(!m.allows(",\""), "nothing is left to name");
        assert!(m.allows("}"));
    }

    #[test]
    fn a_number_cannot_run_forever() {
        // Every digit is legal JSON, so without a bound a model in a
        // repetition loop never closes the document. That is the failure
        // Qwen3 produced: a 200-digit integer and a truncated answer.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        });
        let mut m = machine(schema);
        assert!(m.push("{\"n\": "));
        assert!(m.push(&"1".repeat(MAX_NUMBER_CHARS as usize)));
        assert!(!m.allows("1"), "the number is at its limit");
        assert!(m.allows("}"), "but it can still be closed");
        assert!(m.completes_with("}"));
    }

    #[test]
    fn only_json_whitespace_is_whitespace() {
        // A form feed is ASCII whitespace but not JSON whitespace, and a
        // document containing one does not parse.
        let mut m = machine(serde_json::json!({"type": "array", "items": {"type": "integer"}}));
        assert!(m.push("[1,"));
        assert!(m.allows(" "));
        assert!(m.allows("\t"));
        assert!(m.allows("\r\n"));
        assert!(!m.allows("\u{000c}"), "a form feed is not JSON whitespace");
        assert!(!m.allows("\u{000b}"), "nor is a vertical tab");
    }

    #[test]
    fn whitespace_is_bounded_and_cannot_come_first() {
        let m = machine(serde_json::json!({}));
        assert!(!m.allows(" "), "the document starts at the first character");
        assert!(m.allows("{"));
        let mut m = machine(serde_json::json!({"type": "array", "items": {"type": "integer"}}));
        assert!(m.push("[1,"));
        assert!(m.allows("\n  "), "indentation is fine");
        assert!(!m.allows(&" ".repeat(20)), "a long run is not");
        // Whitespace inside a string is just a character.
        let mut m = machine(serde_json::json!({"type": "string"}));
        assert!(m.push("\"a"));
        assert!(m.allows("   b"));
    }

    #[test]
    fn the_masker_leaves_only_tokens_that_keep_the_document_valid() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        });
        let vocab: Vec<String> = vec![
            "{".into(),     // 0
            "\"a\"".into(), // 1
            "\"b\"".into(), // 2
            ":".into(),     // 3
            "1".into(),     // 4
            "}".into(),     // 5
            "x".into(),     // 6
            "".into(),      // 7, the EOS token has no text
        ];
        let masker = TokenMasker::new(vocab);
        let mut m = machine(schema);
        let mut logits = vec![0.0f32; 8];
        masker.mask(&m, &[7], &mut logits);
        assert_eq!(logits[0], 0.0, "an object may open");
        assert!(logits[1].is_infinite(), "a key needs the brace first");
        assert!(logits[6].is_infinite(), "junk is masked");
        assert!(logits[7].is_infinite(), "the document is not complete");

        assert!(m.push("{"));
        let mut logits = vec![0.0f32; 8];
        masker.mask(&m, &[7], &mut logits);
        assert_eq!(logits[1], 0.0, "the known key is allowed");
        assert!(logits[2].is_infinite(), "an unknown key is not");

        assert!(m.push("\"a\":1}"));
        let mut logits = vec![0.0f32; 8];
        masker.mask(&m, &[7], &mut logits);
        assert_eq!(logits[7], 0.0, "EOS is allowed once complete");
        assert!(logits[0].is_infinite(), "nothing may follow");
        assert!(
            logits
                .iter()
                .enumerate()
                .all(|(i, l)| i == 7 || l.is_infinite()),
            "once complete, only the end of turn is left: {logits:?}"
        );
    }
}
