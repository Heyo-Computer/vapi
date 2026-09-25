//! End to end: a decision worker, over NATS and HTTP.

mod common;

use common::{Stack, binaries, json, model_dir, nats_url, request};

/// Its own model id, so this test's subjects, durable consumer and registry
/// entry cannot collide with a stack someone is running by hand.
const MODEL_ID: &str = "vapi-e2e/laya";

// ------------------------------------------------------------------- tests

/// One stack, many assertions: starting it costs a model load, and the
/// things worth checking are all about the same running system.
#[test]
fn a_decision_worker_answers_over_the_wire() {
    let (Some(model), Some(nats)) = (model_dir("laya.json"), nats_url()) else {
        eprintln!("skipping: needs ~/models/laya and a reachable NATS (VAPI_E2E_NATS)");
        return;
    };
    if binaries().is_none() {
        eprintln!("skipping: build the binaries first (cargo build --release --features cuda)");
        return;
    }
    let stack = Stack::start(MODEL_ID, &model, &nats);

    // --- the answer itself -------------------------------------------------
    let reply = json(
        stack.port,
        "/v1/decisions",
        r#"{"state": {"from": "user@acme.com",
                     "subject": "Duplicate charge on invoice #4411",
                     "body": "We were billed twice for March. Refund it today or we cancel."},
            "questions": {
              "department": {"type": "choice",
                             "instructions": "Which department should handle this request?",
                             "criteria": {"billing": "invoices, payments, refunds",
                                          "technical": "bugs, outages, system errors",
                                          "other": "everything else"}},
              "urgency": {"type": "score", "instructions": "How urgent is this request?",
                          "criteria": ["not urgent", "soon", "critical"]},
              "churn_risk": {"type": "noul",
                             "instructions": "Does the user threaten to cancel or leave?"}}}"#,
    );
    let answers = &reply["answers"];
    assert_eq!(reply["object"], "decision", "{reply}");

    // The model's judgement, not just its plumbing. A billing complaint that
    // threatens to cancel is the fixture the goldens use, and if the answer
    // moved, something upstream of the wire is wrong.
    assert_eq!(answers["department"]["choice"], "billing", "{reply}");
    let billing = answers["department"]["probabilities"]["billing"]
        .as_f64()
        .unwrap();
    assert!(billing > 0.5, "billing at {billing}: {reply}");
    assert!(
        answers["churn_risk"]["noul"].as_f64().unwrap() > 0.5,
        "a cancellation threat should read as churn risk: {reply}"
    );

    // --- shapes, which only the wire can get wrong -------------------------
    // Question order is preserved: the options of a choice go into the prompt
    // in this order, so a reordering changes the answer as well as the keys.
    let keys: Vec<&str> = answers
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["department", "urgency", "churn_risk"], "{reply}");

    // A score reports an expectation over its levels, not an argmax.
    let score = answers["urgency"]["score"].as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score), "{reply}");
    assert!(answers["urgency"]["legend"]["2"] == "critical", "{reply}");
    // A noul reports one probability and no distribution.
    assert!(answers["churn_risk"]["probabilities"].is_null(), "{reply}");
    assert!(
        reply["usage"]["prompt_tokens"].as_u64().unwrap() > 0,
        "{reply}"
    );

    // --- refusals ----------------------------------------------------------
    let bad = json(
        stack.port,
        "/v1/decisions",
        r#"{"state":"x","questions":{"q":{"type":"choice","instructions":"?",
            "criteria":{"only":null}}}}"#,
    );
    assert!(
        bad["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("two options"),
        "a one-option choice has no answer to give: {bad}"
    );

    // A generation request against a decision deployment is refused by the
    // gateway before anything is published — a decision checkpoint ships no
    // chat template, so there is nothing to render. Failing here rather than
    // at the worker is the better of the two: no queue work, and the error
    // names the actual reason instead of "the wrong kind of worker answered".
    let wrong = json(
        stack.port,
        "/v1/chat/completions",
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    );
    assert!(
        wrong["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no chat template"),
        "{wrong}"
    );

    // --- the dashboard saw it ----------------------------------------------
    let stats = request(stack.port, "GET", "/dashboard/stats", None).unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats.body).unwrap();
    assert!(
        stats["workers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w["model"] == MODEL_ID),
        "{stats}"
    );
    assert!(
        stats["recent"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["kind"] == "decision"),
        "{stats}"
    );
}
