use pyo3::prelude::*;
use egg::{rewrite as rw, *};

define_language! {
    enum Math {
        Num(i32),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        Symbol(Symbol),
    }
}

pub fn make_rules() -> Vec<Rewrite<Math, ()>> {
    vec![
        rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rw!("add-0"; "(+ ?a 0)" => "?a"),
        rw!("mul-0"; "(* ?a 0)" => "0"),
        rw!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

#[pyclass]
pub struct SymbolicValidator {
    rules: Vec<Rewrite<Math, ()>>,
}

#[pymethods]
impl SymbolicValidator {
    #[new]
    pub fn new() -> Self {
        Self {
            rules: make_rules(),
        }
    }

    /// Evaluates using the `egg` e-graphs crate the equivalence of lhs and rhs.
    /// This removes the massive SymPy/Z3 Python overhead.
    pub fn check_equivalence(&self, lhs: String, rhs: String) -> PyResult<bool> {
        // Ex: "(+ x 1)" and "(+ 1 x)"
        let parse_lhs = match lhs.parse::<RecExpr<Math>>() {
            Ok(expr) => expr,
            Err(_) => return Ok(false),
        };
        let parse_rhs = match rhs.parse::<RecExpr<Math>>() {
            Ok(expr) => expr,
            Err(_) => return Ok(false),
        };

        let mut runner = Runner::default().with_expr(&parse_lhs).run(&self.rules);
        let id_lhs = runner.roots[0];
        let id_rhs = runner.egraph.add_expr(&parse_rhs);

        // Run equality check over the equivalence classes without any Sympy execution overhead!
        Ok(runner.egraph.find(id_lhs) == runner.egraph.find(id_rhs))
    }
}
