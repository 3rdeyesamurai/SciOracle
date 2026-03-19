#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod blockchain;
mod state_manager;
mod tokenizer;
mod validator;

use state_manager::SciOracleStateManager;
use tokenizer::ASTGraphTokenizer;
use validator::SymbolicValidator;

#[pymodule]
fn scioracle_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ASTGraphTokenizer>()?;
    m.add_class::<SymbolicValidator>()?;
    m.add_class::<SciOracleStateManager>()?;
    m.add_function(wrap_pyfunction!(blockchain::hash_discovery, m)?)?;
    Ok(())
}
