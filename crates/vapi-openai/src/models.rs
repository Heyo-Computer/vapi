use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<Model>,
}

impl ModelList {
    pub fn new(data: Vec<Model>) -> Self {
        Self {
            object: "list",
            data,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model",
            created: super::unix_now(),
            owned_by: "vapi".into(),
        }
    }
}
