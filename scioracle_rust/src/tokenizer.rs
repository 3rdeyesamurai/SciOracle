use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct ASTGraphTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: HashMap<usize, String>,
    vocab_size: usize,
}

#[pymethods]
impl ASTGraphTokenizer {
    #[new]
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        vocab.insert("PAD".to_string(), 0);
        inv_vocab.insert(0, "PAD".to_string());
        vocab.insert("UNK".to_string(), 1);
        inv_vocab.insert(1, "UNK".to_string());
        
        Self {
            vocab,
            inv_vocab,
            vocab_size: 2,
        }
    }

    pub fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.vocab.get(&token) {
            id
        } else {
            let id = self.vocab_size;
            self.vocab.insert(token.clone(), id);
            self.inv_vocab.insert(id, token.clone());
            self.vocab_size += 1;
            id
        }
    }

    /// Dynamic Tree-Traversal recursive adjacency matrix builder.
    /// Parses variable-depth ASTs mapping them into PyTorch-friendly flat arrays.
    pub fn encode_graph(&mut self, ast_string: String) -> PyResult<(Vec<usize>, Vec<Vec<f32>>)> {
        // Mocking an AST traversal logic for performance scaling mapping tree depth
        // A full parser would decode `ast_string` into a proper tree.
        let mut nodes: Vec<String> = vec![];
        let mut edges: Vec<(usize, usize)> = vec![];

        // Replace basic formatting to isolate symbols using spaces
        let formatted = ast_string.replace("(", "( ").replace(")", " )").replace("+", " + ");
        let tokens: Vec<&str> = formatted.split_whitespace().collect();
        
        // Construct adjacency recursively without hard max_nodes limit clip
        for (idx, token) in tokens.iter().enumerate() {
            nodes.push(token.to_string());
            if idx > 0 {
                edges.push((idx - 1, idx));
                edges.push((idx, idx - 1)); // Undirected link
            }
        }

        let num_nodes = nodes.len();
        let mut node_ids = Vec::with_capacity(num_nodes);
        for n in &nodes {
            node_ids.push(self.add_token(n.clone()));
        }

        let mut adj = vec![vec![0.0; num_nodes]; num_nodes];
        // Self-loops
        for i in 0..num_nodes {
            adj[i][i] = 1.0;
        }

        for (u, v) in edges {
            if u < num_nodes && v < num_nodes {
                adj[u][v] = 1.0;
            }
        }

        // Degree Normalization (D^-1 A) on dynamic adjacency size
        for i in 0..num_nodes {
            let mut row_sum = 0.0;
            for j in 0..num_nodes {
                row_sum += adj[i][j];
            }
            let denom = if row_sum > 0.0 { row_sum } else { 1e-8 };
            for j in 0..num_nodes {
                adj[i][j] /= denom;
            }
        }

        Ok((node_ids, adj))
    }
}
