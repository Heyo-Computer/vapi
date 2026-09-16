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
        // Newer Llama templates date-stamp the system prompt.
        env.add_function("strftime_now", |_fmt: String| -> Value {
            // Deterministic by design: a wall-clock date in the prompt would
            // change the token ids every day and defeat the prefix cache for
            // every request that carries a system prompt.
            Value::from("01 Jan 2025")
        });

        env.add_template_owned(TEMPLATE_NAME, source_text.into())
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

    /// Render a conversation.
    ///
    /// `add_generation_prompt` appends the assistant turn header, which is
    /// what makes the model continue rather than predict another user turn.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| Error::ChatTemplate(e.to_string()))?;

        let msgs: Vec<Value> = messages.iter().map(message_to_value).collect();
        tmpl.render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            bos_token => "",
            eos_token => "",
        })
        .map_err(|e| Error::ChatTemplate(format!("{e:#}")))
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
    Value::from(std::collections::BTreeMap::from([
        ("role", Value::from(role_str(m.role))),
        (
            "content",
            Value::from(m.content.clone().unwrap_or_default()),
        ),
    ]))
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
            role: ChatRole::Assistant,
            content: None,
            name: None,
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
    fn tool_and_function_roles_both_render_as_tool() {
        let t = chatml();
        for role in [ChatRole::Tool, ChatRole::Function] {
            let msg = ChatMessage {
                role,
                content: Some("r".into()),
                name: None,
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

    #[test]
    fn trailing_newlines_are_preserved() {
        // HF keeps them; dropping one shifts every subsequent token id.
        let t = ChatTemplate::new("hello\n", TemplateSource::Explicit).unwrap();
        assert_eq!(t.render(&[], false).unwrap(), "hello\n");
    }
}
