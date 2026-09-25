//! Shared harness for the end-to-end tests.
//!
//! The parity and unit tests prove the model and the request shaping. What
//! none of them can catch is the class of bug that only appears once the
//! pieces are separate processes: a gateway and a worker that disagree about
//! `model.id` and so publish to and consume from different subjects, a worker
//! that loads the wrong engine for a checkpoint, an answer that survives the
//! wire but loses its question order.
//!
//! Every test here skips, loudly, unless its three prerequisites are present:
//! built binaries, the checkpoint, and a reachable NATS. A test that silently
//! passes because it did nothing is worse than no test.

#![allow(dead_code)] // Each test file uses a different part of this.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// How long to wait for a worker to load a 842 MB checkpoint and register.
const READY_TIMEOUT: Duration = Duration::from_secs(120);

// ------------------------------------------------------------ prerequisites

pub fn binaries() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for profile in ["release", "debug"] {
        let gateway = root.join("target").join(profile).join("vapi-gateway");
        let worker = root.join("target").join(profile).join("vapi-worker");
        if gateway.exists() && worker.exists() {
            return Some((gateway, worker));
        }
    }
    None
}

/// A golden file, if it has been generated.
pub fn golden(name: &str) -> Option<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// The checkpoint a golden was generated from, so an end-to-end test and the
/// parity tests cannot drift onto different weights.
pub fn model_dir(golden_name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(golden(golden_name)?["dir"].as_str()?);
    dir.join("model.safetensors").exists().then_some(dir)
}

/// A `multipart/form-data` body, for the endpoints that take an upload.
pub fn multipart(fields: &[(&str, &str)], file: Option<(&str, &[u8])>) -> (String, Vec<u8>) {
    let boundary = format!("vapi{}", std::process::id());
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    if let Some((filename, bytes)) = file {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (boundary, body)
}

/// 16-bit mono WAV, so a test can build its own audio rather than depending
/// on a file the machine may not have.
pub fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data: Vec<u8> = samples
        .iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
        .collect();
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// `VAPI_E2E_NATS`, or the port `docker-compose.yml` publishes.
pub fn nats_url() -> Option<String> {
    let url = std::env::var("VAPI_E2E_NATS").unwrap_or_else(|_| "nats://127.0.0.1:14222".into());
    let address = url.trim_start_matches("nats://");
    TcpStream::connect_timeout(&address.parse().ok()?, Duration::from_millis(500))
        .ok()
        .map(|_| url)
}

pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

// ------------------------------------------------------------- tiny client

pub struct Reply {
    pub status: u16,
    pub body: String,
}

/// Just enough HTTP to talk to the gateway.
///
/// Hand-written rather than pulling in a client: this crate has no HTTP
/// dependency and one test is a poor reason to give it one.
pub fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> std::io::Result<Reply> {
    send(
        port,
        method,
        path,
        "application/json",
        body.unwrap_or("").as_bytes(),
    )
}

/// The same, with the body's content type spelled out — an upload is not
/// JSON, and axum's multipart extractor reads the boundary from this header.
pub fn send(
    port: u16,
    method: &str,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<Reply> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(300)))?;
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // Skip headers; `Connection: close` means the body runs to EOF.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
    }
    let mut body = String::new();
    reader.read_to_string(&mut body)?;
    Ok(Reply { status, body })
}

pub fn json(port: u16, path: &str, body: &str) -> serde_json::Value {
    let reply = request(port, "POST", path, Some(body)).expect("request");
    serde_json::from_str(&reply.body).unwrap_or_else(|e| {
        panic!(
            "{path} returned {} which is not JSON: {e}\n{}",
            reply.status, reply.body
        )
    })
}

// ------------------------------------------------------------------- stack

/// A gateway and a worker, killed when the test ends however it ends.
pub struct Stack {
    gateway: Child,
    worker: Child,
    pub port: u16,
    config: PathBuf,
    model_id: String,
}

impl Drop for Stack {
    fn drop(&mut self) {
        // Leaking either would hold a port and a GPU allocation for the rest
        // of the run.
        let _ = self.worker.kill();
        let _ = self.gateway.kill();
        let _ = self.worker.wait();
        let _ = self.gateway.wait();
        let _ = std::fs::remove_file(&self.config);
    }
}

impl Stack {
    pub fn start(model_id: &str, model: &Path, nats: &str) -> Stack {
        let (gateway_bin, worker_bin) = binaries().expect("checked by the caller");
        let port = free_port();
        let config =
            std::env::temp_dir().join(format!("vapi-e2e-laya-{}.toml", std::process::id()));
        std::fs::write(
            &config,
            format!(
                r#"
[nats]
url = "{nats}"
[gateway]
bind = "127.0.0.1:{port}"
[worker]
max_concurrent_seqs = 8
max_batched_tokens = 4096
metrics_bind = "127.0.0.1:{}"
[model]
id = "{model_id}"
path = "{}"
device = "auto"
dtype = "auto"
max_context = 512
num_blocks = 16
[cache]
prefix_cache = false
response_cache = false
"#,
                free_port(),
                model.display()
            ),
        )
        .expect("write the config");

        let spawn = |bin: &Path| {
            Command::new(bin)
                .arg("--config")
                .arg(&config)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap_or_else(|e| panic!("spawn {}: {e}", bin.display()))
        };
        // The gateway creates the stream the worker waits for.
        let gateway = spawn(&gateway_bin);
        let worker = spawn(&worker_bin);
        let mut stack = Stack {
            gateway,
            worker,
            port,
            config,
            model_id: model_id.to_string(),
        };
        stack.wait_until_ready();
        stack
    }

    /// Ready means the gateway answers *and* a worker for **this model** has
    /// registered.
    ///
    /// The model id matters: the registry bucket is shared across every
    /// worker on the NATS server, so waiting for "any worker" would return
    /// immediately on someone else's stack and then time out on the first
    /// request, which is a confusing way to fail.
    fn wait_until_ready(&mut self) {
        let deadline = Instant::now() + READY_TIMEOUT;
        while Instant::now() < deadline {
            // A process that has already exited will never become ready, and
            // waiting out the full timeout to say so buries the real error
            // under two minutes of nothing.
            for (what, child) in [("gateway", &mut self.gateway), ("worker", &mut self.worker)] {
                if let Ok(Some(status)) = child.try_wait() {
                    panic!(
                        "the {what} exited with {status} before becoming ready; \
                         run it by hand with the same config to see why"
                    );
                }
            }
            if let Ok(stats) = request(self.port, "GET", "/dashboard/stats", None)
                && stats.status == 200
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(&stats.body)
                && v["workers"]
                    .as_array()
                    .is_some_and(|w| w.iter().any(|x| x["model"] == self.model_id))
            {
                return;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        panic!(
            "no worker for {} registered within {READY_TIMEOUT:?}",
            self.model_id
        );
    }
}
