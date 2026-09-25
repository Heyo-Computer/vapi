//! API keys, and the cache namespace that comes with them.
//!
//! Two features that look separate and are not. A shared prefix cache is a
//! **timing side channel**: time-to-first-token reveals whether someone
//! recently submitted a given prefix, so a gateway that can tell callers
//! apart should also keep their cached prefixes apart. Giving each key its
//! own namespace by default turns that from something an operator has to
//! remember into something they have to opt out of.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One credential, and what it is allowed to share.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    /// For logs and metrics. Never the key itself.
    pub name: String,
    /// The secret the caller presents.
    pub key: String,
    /// Cache namespace. Defaults to the key's name, so two keys share no
    /// cached prefixes unless they are deliberately given the same one —
    /// which is worth doing for several keys belonging to one team, and
    /// never worth doing across tenants.
    #[serde(default)]
    pub namespace: Option<String>,
}

impl ApiKey {
    pub fn namespace(&self) -> &str {
        self.namespace.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    /// Credentials that may call the API. **Empty means no authentication**,
    /// which is the historical behaviour and is fine on loopback; the gateway
    /// says so at startup rather than leaving it to be discovered.
    pub keys: Vec<ApiKey>,
}

impl AuthConfig {
    pub fn enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Find the key a caller presented.
    ///
    /// Compared as digests in constant time: a byte-by-byte comparison that
    /// returns early tells an attacker how much of a guess was right, and
    /// comparing raw strings of different lengths leaks the length too.
    pub fn lookup(&self, presented: &str) -> Option<&ApiKey> {
        let want = blake3::hash(presented.as_bytes());
        let mut found = None;
        for candidate in &self.keys {
            // No early exit: the loop always runs to the end, so the time it
            // takes does not depend on which key matched.
            let have = blake3::hash(candidate.key.as_bytes());
            if constant_time_eq(want.as_bytes(), have.as_bytes()) {
                found = Some(candidate);
            }
        }
        found
    }

    /// Complain about configurations that look like mistakes.
    ///
    /// Returned rather than logged so the caller decides whether a duplicate
    /// is fatal; at startup it is a warning, in a test it is an assertion.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for k in &self.keys {
            if k.key.trim().is_empty() {
                out.push(format!("key {:?} has an empty secret", k.name));
            }
            if let Some(previous) = seen.insert(&k.key, &k.name) {
                out.push(format!(
                    "keys {previous:?} and {:?} share a secret, so they cannot be told apart",
                    k.name
                ));
            }
        }
        out
    }
}

/// Who is making a request.
///
/// Always present, even with authentication off, so nothing downstream has to
/// branch on whether a caller is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    /// The key's name, or `anonymous`.
    pub name: String,
    /// The cache namespace this caller's work belongs to.
    pub namespace: String,
}

impl Principal {
    pub const ANONYMOUS: &'static str = "anonymous";

    /// The caller when authentication is off.
    pub fn anonymous(namespace: impl Into<String>) -> Self {
        Self {
            name: Self::ANONYMOUS.into(),
            namespace: namespace.into(),
        }
    }

    pub fn is_anonymous(&self) -> bool {
        self.name == Self::ANONYMOUS
    }
}

impl From<&ApiKey> for Principal {
    fn from(key: &ApiKey) -> Self {
        Self {
            name: key.name.clone(),
            namespace: key.namespace().to_string(),
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> AuthConfig {
        AuthConfig {
            keys: vec![
                ApiKey {
                    name: "alice".into(),
                    key: "sk-alice".into(),
                    namespace: None,
                },
                ApiKey {
                    name: "bob".into(),
                    key: "sk-bob".into(),
                    namespace: Some("shared".into()),
                },
            ],
        }
    }

    #[test]
    fn no_keys_means_no_authentication() {
        assert!(!AuthConfig::default().enabled());
        assert!(keys().enabled());
    }

    #[test]
    fn a_key_is_found_by_its_secret_and_nothing_else() {
        let c = keys();
        assert_eq!(c.lookup("sk-alice").unwrap().name, "alice");
        assert!(c.lookup("sk-alic").is_none(), "a prefix must not match");
        assert!(c.lookup("sk-alicee").is_none());
        assert!(c.lookup("alice").is_none(), "the name is not the secret");
        assert!(c.lookup("").is_none());
    }

    #[test]
    fn each_key_gets_its_own_namespace_unless_told_otherwise() {
        // The default is isolation. A shared prefix cache is a timing side
        // channel, so two tenants sharing one by accident is the failure
        // this prevents.
        let c = keys();
        assert_eq!(c.lookup("sk-alice").unwrap().namespace(), "alice");
        assert_eq!(c.lookup("sk-bob").unwrap().namespace(), "shared");
    }

    #[test]
    fn a_principal_carries_the_namespace_downstream() {
        let c = keys();
        let p = Principal::from(c.lookup("sk-alice").unwrap());
        assert_eq!(p.name, "alice");
        assert_eq!(p.namespace, "alice");
        assert!(!p.is_anonymous());

        let anon = Principal::anonymous("global");
        assert!(anon.is_anonymous());
        assert_eq!(anon.namespace, "global");
    }

    #[test]
    fn duplicate_and_empty_secrets_are_reported() {
        let c = AuthConfig {
            keys: vec![
                ApiKey {
                    name: "a".into(),
                    key: "same".into(),
                    namespace: None,
                },
                ApiKey {
                    name: "b".into(),
                    key: "same".into(),
                    namespace: None,
                },
                ApiKey {
                    name: "c".into(),
                    key: "  ".into(),
                    namespace: None,
                },
            ],
        };
        let problems = c.problems().join("; ");
        assert!(problems.contains("share a secret"), "{problems}");
        assert!(problems.contains("empty secret"), "{problems}");
        assert!(AuthConfig::default().problems().is_empty());
    }

    #[test]
    fn comparison_does_not_short_circuit_on_the_first_byte() {
        assert!(constant_time_eq(b"abcd", b"abcd"));
        assert!(!constant_time_eq(b"abcd", b"abce"));
        assert!(!constant_time_eq(b"abcd", b"zbcd"));
        assert!(!constant_time_eq(b"abcd", b"abc"));
    }
}
