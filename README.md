# Mathematical Energy-Based Model (Math-EBM)

An AI architecture designed for symbolic mathematical discovery using Graph Neural Networks (GNNs), Langevin Dynamics, and Contrastive Divergence, fully optimized for consumer-grade GPU constraints.

## Features
- **Dual-Encoder Architecture:** Separate tokenizers for Natural Language inputs (`LLMSeqTokenizer` backed by HuggingFace) and mathematical verification outputs (`ASTGraphTokenizer`).
- **Self-Improvement Cross-Attention:** A neural mechanism where arbitrary LLM sequences actively query internal mathematical AST structures to dynamically align abstract constraints.
- **Algorithmic Experience Replay Discovery:** During Langevin optimization steps, successful identity discoveries mapped natively over SymPy evaluation are pushed into an experience verification buffer, teaching the network positive mathematical truths.
- **Langevin Dynamics MCMC** continuous optimization over discrete symbolic node representations.
- **6GB VRAM Optimization** achieved via `torch.utils.checkpoint` memory accumulation.
- **Arithmetic Cold Start** phase (500 epochs) before algebraic complexity is introduced.
- **SQLite Database Integration** seamlessly serializes Natural Language to Math representations.
- **Offline JSON Generation** generates 10,000 algorithmic algebraic identities.

---

## Prerequisites

To run the training loop and CLI, you will strictly need Python 3.9+ along with PyTorch, SymPy, and Transformers installed.

### Setup Instructions
Run the following in your terminal to install dependencies:
```bash
pip install torch sympy transformers
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

## Deployment and Orchestration

### 1. Containing the Application

To ensure a reproducible and isolated environment, you can containerize the SciOracle Agent Suite using either Conda or Docker.

#### Option A: Conda Environment (Recommended for Local GPU Access)
Conda provides an excellent way to manage Python dependencies while retaining native access to your local Windows environment (RTX 2060 GPU and system RAM).

1. **Install Miniconda or Anaconda** if you haven't already.
2. **Create a new Conda environment:**
   ```bash
   conda create -n scioracle python=3.10
   ```
3. **Activate the environment:**
   ```bash
   conda activate scioracle
   ```
4. **Install the required dependencies:**
   ```bash
   pip install torch sympy transformers z3-solver pyyaml
   ```
5. **Run the Math-EBM or OpenClaw Master Loop natively:**
   ```bash
   python core_agent.py
   ```

#### Option B: Docker Container (Recommended for Isolated Deployment)
Using Docker encapsulates the entire environment, making it easy to deploy across different hardware setups without dependency conflicts.

1. **Create a `Dockerfile` in your project root:**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   # Install essential system dependencies (for z3 and scientific libraries)
   RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       && rm -rf /var/lib/apt/lists/*
       
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   # Set the entry point to the OpenClaw orchestrator
   CMD ["python", "core_agent.py"]
   ```
2. **Create a `requirements.txt` file:**
   ```text
   torch
   sympy
   transformers
   z3-solver
   pyyaml
   ```
3. **Build the Docker Image:**
   ```bash
   docker build -t scioracle-agent .
   ```
4. **Run the Container:**
  *(Note: GPU support in Docker on Windows requires WSL 2 and the NVIDIA Container Toolkit to be configured properly).*
   ```bash
   docker run -it --rm --gpus all -v ${PWD}/discoveries:/app/discoveries scioracle-agent
   ```
   *This command runs the container, enables GPU access via `--gpus all`, and mounts the local `discoveries/` folder so generated math logs and visual proofs are securely mirrored to your host machine.*

### 2. Securely Operating with OpenClaw

The integration of SciOracle with the OpenClaw framework introduces autonomous code execution and logical evaluation. Security must be prioritized.

1. **State-First Protocol Isolation:**
   The `state_manager.py` writes all agent contexts to local disk (`state.json`) rather than persisting them directly in OpenClaw memory structures. Ensure that cross-origin scripting or unauthorized external queries do not interact with your local `state.json`. If deploying via Docker, only mount necessary specific paths.
2. **CPU/GPU Compute Isolation (VRAM Gate):**
   Within `config.yaml`, the system asserts `vram_gate: cap_gb: 6` and explicitly directs OpenClaw's `Oracle_Coder` module to RAM execution. Never mix the active CPU execution thread with OpenClaw's GPU module to inherently protect hardware limits.
3. **Sandboxed Skill Execution:**
   The LLM (`qwen2.5-coder:32b`) generates Python code via the `Oracle_Coder` agent. Because LLM-generated code can be unpredictable:
   - Run the full OpenClaw application exclusively inside the aforementioned **Docker Container** or **Conda Environment**. 
   - Never execute `core_agent.py` inside a production web server environment bearing sensitive keys without properly restricting the evaluation limits inside `skills/symbolic_log.py`.
4. **Local LLM Backend Privacy:**
   Ensure `backend: "ollama"` is properly running in your environment before you execute OpenClaw. Because Ollama handles model inference locally on your hardware, no intellectual property (generated problem representations or Math-EBM data) ever leaves your machine, providing a secure, air-gapped mathematical discovery environment.
