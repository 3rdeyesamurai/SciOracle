use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OracleState {
    pub current_conjecture: Option<String>,
    pub validation_status: String,
    pub validation_errors: Vec<String>,
    pub iteration_count: u32,
    pub discovery_visualized: bool,
    pub ebm_energy: Option<f32>,
    pub critic_signature: Option<String>,
    pub proof_status: String,
    
    // Using a catch-all for extra dynamics
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Default for OracleState {
    fn default() -> Self {
        Self {
            current_conjecture: None,
            validation_status: "pending".to_string(),
            validation_errors: vec![],
            iteration_count: 0,
            discovery_visualized: false,
            ebm_energy: None,
            critic_signature: None,
            proof_status: "unverified".to_string(),
            extra: HashMap::new(),
        }
    }
}

/// Thread-safe, immutable state transitions utilizing Rust's ownership boundaries.
#[pyclass]
pub struct SciOracleStateManager {
    file_path: String,
    state: Arc<RwLock<OracleState>>,
}

#[pymethods]
impl SciOracleStateManager {
    #[new]
    pub fn new(path: String) -> Self {
        let initial_state = if let Ok(data) = fs::read_to_string(&path) {
            serde_json::from_str(&data).unwrap_or_else(|_| OracleState::default())
        } else {
            OracleState::default()
        };

        let manager = Self {
            file_path: path.clone(),
            state: Arc::new(RwLock::new(initial_state.clone())),
        };
        manager.persist(&initial_state);
        manager
    }

    pub fn read_state(&self) -> PyResult<String> {
        let state_guard = self.state.read().unwrap();
        let json = serde_json::to_string(&*state_guard)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
        Ok(json)
    }

    pub fn update_state(&self, json_updates: String) -> PyResult<()> {
        let updates: HashMap<String, Value> = serde_json::from_str(&json_updates)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;
            
        let mut next_state = {
            let guard = self.state.read().unwrap();
            guard.clone() // Immutability transition wrapper, explicitly copying the struct
        };

        // State Transition modifications
        if let Some(c) = updates.get("current_conjecture") {
            next_state.current_conjecture = if c.is_null() { None } else { Some(c.as_str().unwrap_or("").to_string()) };
        }
        if let Some(v) = updates.get("validation_status") {
            next_state.validation_status = v.as_str().unwrap_or("pending").to_string();
        }
        if let Some(errs) = updates.get("validation_errors") {
            if let Some(array) = errs.as_array() {
                next_state.validation_errors = array.iter().map(|e| e.as_str().unwrap_or("").to_string()).collect();
            }
        }
        if let Some(i) = updates.get("iteration_count") {
            next_state.iteration_count = i.as_u64().unwrap_or(0) as u32;
        }
        if let Some(s) = updates.get("critic_signature") {
             next_state.critic_signature = if s.is_null() { None } else { Some(s.as_str().unwrap_or("").to_string()) };
        }
        
        // Push remaining updates into explicit extra blob
        for (k, v) in updates.into_iter() {
            if !["current_conjecture", "validation_status", "validation_errors", "iteration_count", "critic_signature"].contains(&k.as_str()) {
                next_state.extra.insert(k, v);
            }
        }

        {
            let mut guard = self.state.write().unwrap();
            *guard = next_state.clone();
        }
        
        self.persist(&next_state);
        Ok(())
    }
}

impl SciOracleStateManager {
    fn persist(&self, state: &OracleState) {
        if let Ok(json_data) = serde_json::to_string_pretty(state) {
            let tmp_path = format!("{}.tmp", self.file_path);
            let _ = fs::write(&tmp_path, &json_data);
            let _ = fs::rename(&tmp_path, &self.file_path);
        }
    }
}
