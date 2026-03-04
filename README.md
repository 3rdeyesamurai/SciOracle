# Mathematical Energy-Based Model (Math-EBM)

An AI architecture designed for symbolic mathematical discovery using Graph Neural Networks (GNNs), Langevin Dynamics, and Contrastive Divergence, fully optimized for consumer-grade GPU constraints.

## Features
- **Graph Neural Network (GNN)** processing of mathematical Abstract Syntax Trees (ASTs).
- **Langevin Dynamics MCMC** continuous optimization over discrete symbolic node representations.
- **6GB VRAM Optimization** achieved via `torch.utils.checkpoint` memory accumulation.
- **Arithmetic Cold Start** phase (500 epochs) before algebraic complexity is introduced.
- **SQLite Database Integration** seamlessly serializes Natural Language to Math representations.
- **Offline JSON Generation** generates 10,000 algorithmic algebraic identities.

---

## Prerequisites

To run the training loop and CLI, you will strictly need Python 3.9+ along with PyTorch and SymPy installed.

### Setup Instructions
Run the following in your terminal to install dependencies:
```bash
pip install torch sympy
```

*(Note: PyTorch will automatically install either the CUDA or CPU version depending on your system's hardware configuration.)*

---

## 🚀 How to Run

### 1. Training the Model (Database & Serialization Pipeline)
Before you can interact with the Mathematical CLI, you must populate the weights and database logic using the training flag. 

You can execute the entire training pipeline natively in your terminal:
```bash
python ebm_math_discovery.py --train
```

#### Optional: CPU Mode
Because algebraic graph nodes can easily exceed 6GB VRAM limits during massive tensor parallelization, you can add `--cpu` to forcefully bind the training tensor allocations to your system's 32GB RAM module instead.
```bash
python ebm_math_discovery.py --train --cpu
```

### 2. Interactive Natural Language CLI
Once `math_ebm.pt` weight checkpoints are serialized to disk, entering the script normally drops you into the **Interactive Discovery Shell** where you can type algebraic identities in **Plain English**.

```bash
python ebm_math_discovery.py
```

**Example Use Case**:
```text
=== Mathematical EBM Interface ===
Loading model weights...
Model ready!

Problem (Natural Language): x squared plus 5 times x plus 6
Proposed Solution (NLP): (x plus 2) times (x plus 3)

Parsed Math (Problem):      x**2 + 5*x + 6
Parsed Math (Sol):          (x + 2)*(x + 3)
==> Predicted Energy: 0.1245
    (Mathematically sound logically: TRUE)
    (Saved to knowledge database)
```
