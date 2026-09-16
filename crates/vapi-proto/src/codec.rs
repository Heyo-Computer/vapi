use bytes::Bytes;
use serde::{Serialize, de::DeserializeOwned};
use vapi_core::{Error, Result};

/// JSON on the wire.
///
/// A compact binary format would shave bytes off each delta, but these
/// messages are small and dominated by network round-trips, and being able to
/// read a live stream with `nats sub` is worth more during development than
/// the bytes saved. The codec is isolated here so swapping it later touches
/// one file.
pub fn encode<T: Serialize>(value: &T) -> Result<Bytes> {
    serde_json::to_vec(value)
        .map(Bytes::from)
        .map_err(|e| Error::Transport(format!("encode: {e}")))
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| Error::Transport(format!("decode: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_error_is_transport_not_a_panic() {
        let r: Result<crate::Job> = decode(b"{not json");
        assert!(matches!(r, Err(Error::Transport(_))));
    }
}
