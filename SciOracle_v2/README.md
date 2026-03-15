# SciOracle v2: Autonomous Mathematical Discovery Engine

SciOracle v2 is a next-generation automated machine learning research system capable of discovering mathematical identities, verifying symbolic equations, generating proofs, exploring mathematical spaces autonomously, and managing its own theorem knowledge graph.

It tightly integrates **Energy-Based Models (EBMs)** mapped over **Graph Neural Networks (GNNs)** combined with **Monte Carlo Tree Search (MCTS)** and **Reinforcement Learning** policies. Every identity is scrutinized heavily using a dual verification system employing **SymPy** and the rigorous **Z3 Theorem Prover**. 

## Repository Architecture

```text
SciOracle_v2/
├── README.md                 # Project architecture overview & documentation
├── config/
│   └── config.yaml           # Model hyperparameters, DB info, System details
├── core/
│   ├── energy_model.py       # Mathematical EBM mapping equations to energy levels 
│   └── graph_encoder.py      # GNN extracting embeddings from SymPy ASTs
├── symbolic/
│   ├── ast_parser.py         # Converts SymPy into PyTorch Geometric
│   ├── mutator.py            # Executes transformations (expand, integrate, etc.)
│   └── rewrite_rules.py      # Standard supplementary logic relations
├── search/
│   ├── mcts.py               # Tree Search guided dynamically by neural energy scores
│   └── rl_policy.py          # Value and Policy logic heads mapping state vectors 
├── verification/
│   ├── sympy_validator.py    # Algebraic equivalence simplifications
│   └── z3_validator.py       # Converts Python equations to UNSAT proofs
├── proof/
│   └── proof_generator.py    # Generates exportable derivation objects
├── memory/
│   ├── database.py           # SQLite storage backends for proven theorems
│   └── theorem_graph.py      # NetworkX structures tracking path similarities
├── training/
│   ├── dataset_generator.py  # Generates vast sets of equations computationally
│   └── trainer.py            # Curriculum curriculum Contrastive Loss loop
├── agents/
│   └── research_agent.py     # Main high-level loop orchestrating all stages
└── cli/
    └── interface.py          # Unified operator command access
```

## System Overview & Training Pipeline

1. **Symbolic Extraction & GNN Encoding**: Sympy expressions are parsed via the AST syntax trees directly into Node types (`Add`, `Mul`, `Function`, `Variable`, etc) and parsed through message passing into embedding representations.
2. **Curriculum EBM Training**: A robust dual contrastive framework evaluates `Energy = || f(lhs) - f(rhs) ||^2`. Low energy implies structural semantic equality. The Dataset synthesizer continuously climbs stages across math areas (arithmetic -> linear -> geometry -> calculus) feeding this layout.
3. **MCTS Expansion Process**: Using the mutated Sympy combinations, tree search traverses nodes via our RL evaluation matrix `π(path_vector)`.
4. **Z3 Verifications**: At terminal nodes, mathematical forms undergo stringent analysis guaranteeing validity via checking satisfiability.
5. **Knowledge Subgraph**: The results are wrapped cleanly via `ProofObjects` maintaining transparency over what specific algorithms caused the breakthrough.

## Installation

Ensure you have a recent consumer GPU holding at least 6–8GB VRAM (or use a CPU array). Built using Python 3.10+.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric sympy networkx z3-solver pyyaml matplotlib
```

## CLI Usage

Enter your environment and pass direct Python command links.
*(Run these commands from `SciOracle_v2` folder.)*

```bash
# 1. Train the neural core 
python cli/interface.py train

# 2. Explore single formulas 
python cli/interface.py explore "(x + 2)*(x + 3)"

# 3. Read accumulated database
python cli/interface.py query

# 4. Trigger the full AI agent pipeline (Generates, MCTS, Verify, Database)
python cli/interface.py auto --iters 15
```

## Example Discovery Output

```text
Generated problem state: (9*x - 3)*(2*x + 1)
Discovered identity trajectory end: 18*x**2 + 3*x - 3
SUCCESS: Identity Verified using SymPy + Z3 
Proof Details:

Original Form: (9*x - 3)*(2*x + 1)
-------------------------
Step  1: expand => 18*x**2 + 3*x - 3
-------------------------

Final Identity Verified: 18*x**2 + 3*x - 3
```

This output structure reflects true formalized autonomous scientific search loops handling symbolic manipulation iteratively!
