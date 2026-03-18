use pyo3::prelude::*;

mod tokenizer;
mod validator;
mod state_manager;
mod blockchain;

use tokenizer::ASTGraphTokenizer;
use validator::SymbolicValidator;
use state_manager::SciOracleStateManager;

#[pymodule]
fn scioracle_rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ASTGraphTokenizer>()?;
    m.add_class::<SymbolicValidator>()?;
    m.add_class::<SciOracleStateManager>()?;
    m.add_function(wrap_pyfunction!(blockchain::hash_discovery, m)?)?;
    Ok(())
}
