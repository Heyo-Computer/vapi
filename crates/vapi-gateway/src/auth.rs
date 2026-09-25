//! API-key authentication.
//!
//! The layer runs on every protected route whether or not keys are
//! configured, and inserts a [`Principal`] either way. Nothing downstream has
//! to know whether authentication is on — it reads the principal's namespace
//! and gets isolation when there are keys and the shared default when there
//! are not.
//!
//! # Where the key may come from
//!
//! `Authorization: Bearer <key>` is what an OpenAI client sends and what the
//! API path uses. The other two exist for the dashboard, which is a browser
//! and cannot set a header: opening `/dashboard?key=<key>` once exchanges the
//! key for a cookie. Without that, turning authentication on would leave the
//! dashboard either broken or — worse — an unauthenticated way to run
//! inference through its "Try it" box.

use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use vapi_core::Principal;

use crate::state::SharedState;

/// The cookie the dashboard uses, set by `?key=` and never readable by
/// scripts on the page.
pub const COOKIE: &str = "vapi_key";

/// Pull a presented key out of a request, from any of the three places.
pub fn presented_key(headers: &header::HeaderMap, query: Option<&str>) -> Option<String> {
    if let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = value.trim();
        // `Bearer` is case-insensitive per RFC 6750, and clients differ.
        if let Some(rest) = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
        {
            return Some(rest.trim().to_string());
        }
    }
    // Anthropic-style clients, and curl users who find it easier.
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(value.trim().to_string());
    }
    if let Some(cookie) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
        && let Some(key) = cookie_value(cookie, COOKIE)
    {
        return Some(key);
    }
    query.and_then(|q| query_value(q, "key"))
}

/// One cookie out of a `Cookie:` header.
fn cookie_value(header: &str, name: &str) -> Option<String> {
    header.split(';').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k.trim() == name).then(|| v.trim().to_string())
    })
}

/// One parameter out of a query string, percent-decoding `%XX` and `+`.
pub fn query_value(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then(|| percent_decode(v))
    })
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(b) => {
                        out.push(b);
                        i += 3;
                    }
                    None => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Reject a request that has no valid key, and name the principal otherwise.
pub async fn require_key(State(st): State<SharedState>, mut req: Request, next: Next) -> Response {
    let principal = match st.cfg.auth.enabled() {
        false => Principal::anonymous(st.cfg.cache.default_namespace.clone()),
        true => {
            let query = req.uri().query().map(str::to_string);
            let presented = presented_key(req.headers(), query.as_deref());
            match presented.as_deref().and_then(|k| st.cfg.auth.lookup(k)) {
                Some(key) => Principal::from(key),
                None => {
                    metrics::counter!("vapi_unauthorized_total").increment(1);
                    let browser = req.uri().path().starts_with("/dashboard");
                    return unauthorized(presented.is_some(), browser);
                }
            }
        }
    };
    req.extensions_mut().insert(principal);
    next.run(req).await
}

fn unauthorized(presented: bool, browser: bool) -> Response {
    let message = match (presented, browser) {
        (true, _) => "invalid api key".to_string(),
        // A browser cannot set a header, so point it at the one thing it can
        // do rather than at instructions it cannot follow.
        (false, true) => "missing api key; open /dashboard?key=<key> once".to_string(),
        (false, false) => "missing api key; send `Authorization: Bearer <key>`".to_string(),
    };
    (
        StatusCode::UNAUTHORIZED,
        // The header a spec-abiding client looks for before retrying.
        [(header::WWW_AUTHENTICATE, "Bearer")],
        axum::Json(serde_json::json!({
            "error": { "message": message, "type": "invalid_request_error", "code": "invalid_api_key" }
        })),
    )
        .into_response()
}

/// `Set-Cookie` for a dashboard session.
///
/// `HttpOnly` so a script on the page cannot read it, `SameSite=Strict` so
/// another site cannot ride it. Not `Secure`, because the dashboard is served
/// over plain HTTP on loopback and marking it secure would stop the cookie
/// being set at all.
pub fn session_cookie(key: &str) -> String {
    format!("{COOKIE}={key}; Path=/; HttpOnly; SameSite=Strict")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        h
    }

    #[test]
    fn a_bearer_header_is_read_in_either_case() {
        for value in ["Bearer sk-1", "bearer sk-1", "  Bearer   sk-1  "] {
            assert_eq!(
                presented_key(&headers(&[("authorization", value)]), None).as_deref(),
                Some("sk-1"),
                "{value:?}"
            );
        }
    }

    #[test]
    fn other_authorization_schemes_are_not_mistaken_for_keys() {
        // `Basic dXNlcjpwYXNz` is not a bearer token, and treating it as one
        // would put a base64 blob in front of the key comparison.
        assert!(presented_key(&headers(&[("authorization", "Basic abc")]), None).is_none());
    }

    #[test]
    fn the_header_wins_over_the_cookie_and_the_query() {
        let h = headers(&[
            ("authorization", "Bearer from-header"),
            ("cookie", "vapi_key=from-cookie"),
        ]);
        assert_eq!(
            presented_key(&h, Some("key=from-query")).as_deref(),
            Some("from-header")
        );
    }

    #[test]
    fn the_dashboard_can_present_a_cookie_or_a_query_parameter() {
        let h = headers(&[("cookie", "other=1; vapi_key=sk-2; another=3")]);
        assert_eq!(presented_key(&h, None).as_deref(), Some("sk-2"));
        assert_eq!(
            presented_key(&HeaderMap::new(), Some("a=1&key=sk-3&b=2")).as_deref(),
            Some("sk-3")
        );
    }

    #[test]
    fn a_key_with_url_unsafe_characters_survives_the_query() {
        assert_eq!(
            presented_key(&HeaderMap::new(), Some("key=sk%2Fa%2Bb%3Dc")).as_deref(),
            Some("sk/a+b=c")
        );
    }

    #[test]
    fn nothing_presented_is_nothing_found() {
        assert!(presented_key(&HeaderMap::new(), None).is_none());
        assert!(presented_key(&HeaderMap::new(), Some("model=x")).is_none());
        assert!(presented_key(&headers(&[("cookie", "unrelated=1")]), None).is_none());
    }

    #[test]
    fn the_session_cookie_cannot_be_read_by_a_script_or_ridden_from_another_site() {
        let c = session_cookie("sk-1");
        assert!(c.contains("HttpOnly"), "{c}");
        assert!(c.contains("SameSite=Strict"), "{c}");
    }
}
