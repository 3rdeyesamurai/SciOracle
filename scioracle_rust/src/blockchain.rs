use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

/// Function for Immutable Discovery Logging on Blockchain.
/// Creates a cryptographic hash of Sound discoveries and formats them into a Blockchain payload.
#[pyfunction]
pub fn hash_discovery(signature: String, solver_address: String, energy_reward: f32) -> PyResult<String> {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let payload = json!({
        "network": "Solana_Ethereum",
        "protocol": "Proof_of_Discovery",
        "conjecture_signature": signature,
        "solver": solver_address,
        "energy_gap": energy_reward,
        "timestamp": ts
    });

    let payload_str = payload.to_string();
    
    let mut hasher = Sha256::new();
    hasher.update(payload_str.as_bytes());
    let result = hasher.finalize();
    
    let hex_hash = hex::encode(result);
    
    let tx_payload = json!({
        "payload": payload,
        "tx_hash": hex_hash
    });

    Ok(tx_payload.to_string())
}
