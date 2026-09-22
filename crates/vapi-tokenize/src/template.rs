use minijinja::{Environment, UndefinedBehavior, Value};
use vapi_core::{Error, Result};
use vapi_openai::{ChatMessage, ChatRole};

/// Where a model's chat template came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateSource {
    /// `tokenizer_config.json`'s `chat_template`, as a plain string.
    TokenizerConfig,
    /// A named entry from a `chat_template` list, which newer repos use to
    /// ship several templates (e.g. `default` alongside `tool_use`).
    Named(String),
    /// `chat_template.jinja` next to the tokenizer, which newer repos use
    /// instead of the config key.
    TemplateFile,
    /// Supplied by us, for tests or models with no template of their own.
    Explicit,
}

/// A Jinja chat template, rendered exactly as Hugging Face renders it.
///
/// The details here are not incidental. A template mismatch does not crash —
/// it produces a subtly malformed prompt, and the model's output degrades in a
/// way that looks like a bad model rather than a bad string. Every quirk
/// handled below corresponds to a construct real templates use.
pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: TemplateSource,
}

const TEMPLATE_NAME: &str = "chat";

impl ChatTemplate {
    pub fn new(source_text: impl Into<String>, source: TemplateSource) -> Result<Self> {
        let mut env = Environment::new();

        // Templates rely on `messages[0]['role']`-style access and on
        // undefined values being falsy rather than an error.
        env.set_undefined_behavior(UndefinedBehavior::Chainable);
        // Byte-exact fidelity with HF's rendering.
        env.set_keep_trailing_newline(true);
        // HF templates are written against Python's Jinja2, and lean on
        // dict/str methods like `message.get("content")`, `.items()`,
        // `.startswith()`. minijinja's Python-compat hook supplies them.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        // Many templates call `raise_exception(...)` to reject unsupported
        // conversations; without it they fail with "unknown function".
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        // `tojson` as transformers defines it: Python's `json.dumps` with
        // `ensure_ascii=False`, i.e. `", "` and `": "` separators and keys in
        // insertion order. minijinja's own filter writes compact JSON, which
        // would change every token of a tool definition in the prompt.
        env.add_filter(
            "tojson",
            |v: Value| -> std::result::Result<Value, minijinja::Error> {
                let json = serde_json::to_value(&v).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })?;
                let mut out = String::new();
                python_json(&json, &mut out);
                Ok(Value::from(out))
            },
        );
        // Newer Llama templates date-stamp the system prompt.
        env.add_function("strftime_now", |_fmt: String| -> Value {
            // Deterministic by design: a wall-clock date in the prompt would
            // change the token ids every day and defeat the prefix cache for
            // every request that carries a system prompt.
            Value::from("01 Jan 2025")
        });

        // `{% generation %}` ... `{% endgeneration %}` is a transformers-only
        // tag that marks assistant spans for training masks; at inference HF
        // renders straight through it. minijinja rejects it, so drop the tags.
        let source_text = strip_generation_tags(&source_text.into());
        env.add_template_owned(TEMPLATE_NAME, source_text)
            .map_err(|e| Error::ChatTemplate(e.to_string()))?;

        Ok(Self { env, source })
    }

    /// Extract the template from a parsed `tokenizer_config.json`.
    ///
    /// `chat_template` is a string in most repos but a list of
    /// `{name, template}` objects in newer ones; both are handled.
    pub fn from_tokenizer_config(config: &serde_json::Value) -> Result<Option<Self>> {
        match config.get("chat_template") {
            Some(serde_json::Value::String(s)) => {
                Ok(Some(Self::new(s.clone(), TemplateSource::TokenizerConfig)?))
            }
            Some(serde_json::Value::Array(entries)) => {
                // Prefer the one named "default", else the first.
                let pick = entries
                    .iter()
                    .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| entries.first());
                let Some(entry) = pick else { return Ok(None) };
                let Some(text) = entry.get("template").and_then(|t| t.as_str()) else {
                    return Ok(None);
                };
                let name = entry
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("default");
                Ok(Some(Self::new(
                    text.to_string(),
                    TemplateSource::Named(name.to_string()),
                )?))
            }
            _ => Ok(None),
        }
    }

    /// Render a conversation with no special-token strings.
    ///
    /// Only for templates that never mention `bos_token`/`eos_token` (ChatML
    /// and friends) and for tests. A real model goes through
    /// [`ChatTemplate::render_with`], because Llama 3.x templates open with
    /// `{{- bos_token }}` and rendering that as `""` silently drops
    /// `<|begin_of_text|>` from every prompt.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with(messages, add_generation_prompt, "", "", None)
    }

    /// Render a conversation.
    ///
    /// `add_generation_prompt` appends the assistant turn header, which is
    /// what makes the model continue rather than predict another user turn.
    /// `bos_token` and `eos_token` are the model's special-token *strings*
    /// from `tokenizer_config.json`; the template decides where they go, and
    /// the tokenizer then encodes them without adding its own.
    /// `tools`, when given, are OpenAI tool definitions handed to the
    /// template's `tools` variable as they are; the template decides how to
    /// present them (LFM2.5 lists them in the system turn).
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| Error::ChatTemplate(e.to_string()))?;

        let msgs: Vec<Value> = messages.iter().map(message_to_value).collect();
        let tools: Option<Vec<Value>> =
            tools.map(|t| t.iter().map(Value::from_serialize).collect());
        tmpl.render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token,
            eos_token => eos_token,
            tools => tools,
        })
        .map_err(|e| Error::ChatTemplate(format!("{e:#}")))
    }
}

/// Remove `{% generation %}` / `{% endgeneration %}` tags, with or without
/// whitespace-control dashes, leaving the wrapped content in place.
fn strip_generation_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{%") {
        let Some(end_rel) = rest[start..].find("%}") else {
            break;
        };
        let end = start + end_rel + 2;
        let inner = rest[start + 2..end - 2].trim_matches(|c| c == '-' || c == ' ' || c == '\t');
        if inner == "generation" || inner == "endgeneration" {
            out.push_str(&rest[..start]);
        } else {
            out.push_str(&rest[..end]);
        }
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// `json.dumps(v, ensure_ascii=False)` byte for byte.
fn python_json(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value as J;
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Number(n) => out.push_str(&n.to_string()),
        J::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        J::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_json(item, out);
            }
            out.push(']');
        }
        J::Object(map) => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                python_json(item, out);
            }
            out.push('}');
        }
    }
}

fn role_str(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        // Templates overwhelmingly branch on "tool"; the deprecated
        // "function" role is treated the same way.
        ChatRole::Tool | ChatRole::Function => "tool",
    }
}

fn message_to_value(m: &ChatMessage) -> Value {
    // `content` must be a plain string before rendering. Handing the template
    // a structured value makes it interpolate a debug representation into the
    // prompt, which is silently wrong.
    let mut map = std::collections::BTreeMap::from([
        ("role", Value::from(role_str(m.role))),
        (
            "content",
            Value::from(m.content.clone().unwrap_or_default()),
        ),
    ]);
    if let Some(name) = &m.name {
        map.insert("name", Value::from(name.clone()));
    }
    if let Some(id) = &m.tool_call_id {
        map.insert("tool_call_id", Value::from(id.clone()));
    }
    if let Some(calls) = &m.tool_calls {
        // Templates take `function.arguments` as a mapping (transformers
        // parses the JSON before rendering; LFM2.5's template raises on a
        // string), so decode it here. Undecodable arguments stay a string
        // and the template reports it.
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let args: serde_json::Value = serde_json::from_str(&c.function.arguments)
                    .unwrap_or(serde_json::Value::String(c.function.arguments.clone()));
                Value::from_serialize(serde_json::json!({
                    "id": c.id,
                    "type": c.kind,
                    "function": {"name": c.function.name, "arguments": args},
                }))
            })
            .collect();
        map.insert("tool_calls", Value::from(calls));
    }
    Value::from(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ChatML-style template, the shape Qwen and many others use.
    const CHATML: &str = "{% for message in messages %}\
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}\
{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    fn chatml() -> ChatTemplate {
        ChatTemplate::new(CHATML, TemplateSource::Explicit).unwrap()
    }

    #[test]
    fn renders_a_conversation_in_chatml() {
        let t = chatml();
        let out = t
            .render(
                &[
                    ChatMessage::system("You are terse."),
                    ChatMessage::user("Hi"),
                ],
                true,
            )
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn the_generation_prompt_is_what_makes_the_model_answer() {
        let t = chatml();
        let with = t.render(&[ChatMessage::user("Hi")], true).unwrap();
        let without = t.render(&[ChatMessage::user("Hi")], false).unwrap();
        assert!(with.ends_with("<|im_start|>assistant\n"));
        assert!(!without.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn a_multi_turn_conversation_round_trips_in_order() {
        let t = chatml();
        let out = t
            .render(
                &[
                    ChatMessage::user("one"),
                    ChatMessage::assistant("two"),
                    ChatMessage::user("three"),
                ],
                true,
            )
            .unwrap();
        let positions: Vec<_> = ["one", "two", "three"]
            .iter()
            .map(|s| out.find(s).unwrap())
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "turns must stay ordered"
        );
    }

    #[test]
    fn a_message_with_no_content_renders_as_empty_not_as_debug_output() {
        let t = chatml();
        let msg = ChatMessage {
            content: None,
            ..ChatMessage::assistant("")
        };
        let out = t.render(&[msg], false).unwrap();
        assert_eq!(out, "<|im_start|>assistant\n<|im_end|>\n");
        assert!(!out.contains("None") && !out.contains("null"));
    }

    #[test]
    fn raise_exception_becomes_an_error_rather_than_an_unknown_function() {
        let t = ChatTemplate::new(
            "{% if true %}{{ raise_exception('system messages unsupported') }}{% endif %}",
            TemplateSource::Explicit,
        )
        .unwrap();
        let err = t.render(&[ChatMessage::user("hi")], false).unwrap_err();
        assert!(
            format!("{err}").contains("system messages unsupported"),
            "the template's own message must survive: {err}"
        );
    }

    #[test]
    fn strftime_now_is_available_and_deterministic() {
        // A live date here would change the prompt's token ids daily and
        // invalidate every cached system-prompt prefix.
        let t =
            ChatTemplate::new("{{ strftime_now('%d %b %Y') }}", TemplateSource::Explicit).unwrap();
        let a = t.render(&[], false).unwrap();
        let b = t.render(&[], false).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn undefined_values_chain_instead_of_erroring() {
        // Templates routinely probe fields that are not always present.
        let t = ChatTemplate::new(
            "{% if messages[0].tool_calls %}tools{% else %}none{% endif %}",
            TemplateSource::Explicit,
        )
        .unwrap();
        assert_eq!(t.render(&[ChatMessage::user("hi")], false).unwrap(), "none");
    }

    #[test]
    fn tools_and_tool_calls_reach_the_template() {
        const T: &str = "{% if tools %}tools:{% for t in tools %}{{ t.function.name }};{% endfor %}\n{% endif %}\
{% for m in messages %}{{ m.role }}:{{ m.content }}\
{% if m.tool_calls %}{% for c in m.tool_calls %}[{{ c.function.name }}({{ c.function.arguments.city }})]{% endfor %}{% endif %}\
{% if m.tool_call_id %}<{{ m.tool_call_id }}>{% endif %}\n{% endfor %}";
        let t = ChatTemplate::new(T, TemplateSource::Explicit).unwrap();
        let tools =
            vec![serde_json::json!({"type": "function", "function": {"name": "get_weather"}})];
        let mut call = ChatMessage::assistant("");
        call.tool_calls = Some(vec![vapi_openai::ToolCall::function(
            "call_1",
            "get_weather",
            r#"{"city":"Paris"}"#.into(),
        )]);
        let mut result = ChatMessage::user("18C");
        result.role = ChatRole::Tool;
        result.tool_call_id = Some("call_1".into());
        let out = t
            .render_with(
                &[ChatMessage::user("weather?"), call, result],
                false,
                "",
                "",
                Some(&tools),
            )
            .unwrap();
        assert_eq!(
            out,
            "tools:get_weather;\nuser:weather?\nassistant:[get_weather(Paris)]\ntool:18C<call_1>\n"
        );
        // Without tools the block is skipped.
        let out = t.render(&[ChatMessage::user("hi")], false).unwrap();
        assert_eq!(out, "user:hi\n");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let t = ChatTemplate::new("{{ v | tojson }}", TemplateSource::Explicit).unwrap();
        let tmpl = t.env.get_template(TEMPLATE_NAME).unwrap();
        let v = serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {"required": ["a"], "n": 1.5, "ok": true, "none": null, "s": "é\"q\n"}}});
        let out = tmpl
            .render(minijinja::context! { v => Value::from_serialize(&v) })
            .unwrap();
        assert_eq!(
            out,
            r#"{"type": "function", "function": {"name": "f", "parameters": {"required": ["a"], "n": 1.5, "ok": true, "none": null, "s": "é\"q\n"}}}"#
        );
    }

    #[test]
    fn tool_and_function_roles_both_render_as_tool() {
        let t = chatml();
        for role in [ChatRole::Tool, ChatRole::Function] {
            let msg = ChatMessage {
                role,
                ..ChatMessage::user("r")
            };
            assert!(
                t.render(&[msg], false)
                    .unwrap()
                    .contains("<|im_start|>tool")
            );
        }
    }

    #[test]
    fn a_string_chat_template_is_picked_up_from_tokenizer_config() {
        let cfg = serde_json::json!({ "chat_template": CHATML });
        let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(t.source, TemplateSource::TokenizerConfig);
    }

    #[test]
    fn a_list_valued_chat_template_prefers_the_default_entry() {
        // Newer repos ship several templates; picking the wrong one silently
        // changes the prompt format.
        let cfg = serde_json::json!({
            "chat_template": [
                { "name": "tool_use", "template": "TOOLS" },
                { "name": "default", "template": CHATML },
            ]
        });
        let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(t.source, TemplateSource::Named("default".into()));
        assert!(
            t.render(&[ChatMessage::user("x")], false)
                .unwrap()
                .contains("<|im_start|>")
        );
    }

    #[test]
    fn a_model_with_no_template_is_not_an_error() {
        let cfg = serde_json::json!({ "model_max_length": 4096 });
        assert!(ChatTemplate::from_tokenizer_config(&cfg).unwrap().is_none());
    }

    /// The opening of the Llama 3.x templates, which emit BOS themselves.
    const LLAMA3_HEAD: &str = "{{- bos_token }}\
{%- for message in messages %}\
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}\
{%- endfor %}\
{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

    #[test]
    fn the_models_bos_string_reaches_a_template_that_emits_it() {
        // F3: Llama 3 opens with `{{- bos_token }}`. Rendering that as ""
        // produced prompts with no <|begin_of_text|>, which HF never does.
        let t = ChatTemplate::new(LLAMA3_HEAD, TemplateSource::Explicit).unwrap();
        let out = t
            .render_with(
                &[ChatMessage::user("Hi")],
                true,
                "<|begin_of_text|>",
                "<|eot_id|>",
                None,
            )
            .unwrap();
        assert_eq!(
            out,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        // And the convenience form really does leave it out, so nothing that
        // loads a real model may use it.
        assert!(
            t.render(&[ChatMessage::user("Hi")], true)
                .unwrap()
                .starts_with("<|start_header_id|>")
        );
    }

    #[test]
    fn python_dict_and_string_methods_work() {
        // LFM2.5's template uses `message.get("content")` and friends.
        let t = ChatTemplate::new(
            "{% for m in messages %}{% if m.get('content') %}{{ m['content'].upper() }}{% endif %}{% endfor %}",
            TemplateSource::Explicit,
        )
        .unwrap();
        assert_eq!(t.render(&[ChatMessage::user("hi")], false).unwrap(), "HI");
    }

    #[test]
    fn generation_tags_are_ignored_like_hugging_face_does() {
        // LFM2.5 wraps assistant turns in transformers' training-mask tag.
        let t = ChatTemplate::new(
            "{% for m in messages %}{{ m['role'] }}:{% generation %}{{ m['content'] }}{% endgeneration %};{% endfor %}",
            TemplateSource::Explicit,
        )
        .unwrap();
        let out = t.render(&[ChatMessage::assistant("hi")], false).unwrap();
        assert_eq!(out, "assistant:hi;");
        assert_eq!(
            strip_generation_tags("a{%- generation -%}b{%- endgeneration -%}c{% if x %}d"),
            "abc{% if x %}d"
        );
    }

    #[test]
    fn trailing_newlines_are_preserved() {
        // HF keeps them; dropping one shifts every subsequent token id.
        let t = ChatTemplate::new("hello\n", TemplateSource::Explicit).unwrap();
        assert_eq!(t.render(&[], false).unwrap(), "hello\n");
    }
}
