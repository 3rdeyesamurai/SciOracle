# SciOracle: 2026 Vibe Coding Architecture
**Mathematical Energy-Based Model (Math-EBM)**

An AI architecture designed for symbolic mathematical discovery using Graph Neural Networks (GNNs), Langevin Dynamics, and Contrastive Divergence. Meticulously optimized for local execution on consumer hardware (e.g., RTX 2060 with 32GB RAM) with a definitive **2026 hybrid-cloud deployment** intent.

## The Key Innovation: Vibe Coding
SciOracle pioneers the **"Vibe Coding" architecture** — translating raw, natural language scientific intent into mathematically validated execution code. This process relies strictly on resilient, persistent state files (`state.json`) rather than ephemeral conversational memory. You describe the "vibe" of the formula, and SciOracle's dual encoders systematically graph, solve, and formally prove it.

## Core Features
- **Vibe-to-Math Translation Pipeline:** Arbitrary LLM natural language sequences actively query internal mathematical AST structures to dynamically align abstract constraints.
- **Dual-Encoder Architecture:** Separate tokenizers for Natural Language inputs (`LLMSeqTokenizer`) and mathematical verification outputs (`ASTGraphTokenizer`).
- **Algorithmic Experience Replay Discovery:** During Langevin optimization steps, successful identity discoveries mapped natively over SymPy evaluation are pushed into an experience verification buffer.
- **Langevin Dynamics MCMC** continuous optimization over discrete symbolic node representations.
- **6GB VRAM Strict Optimization** achieved via `torch.utils.checkpoint` and explicit RAM offloading on RTX 2060 architectures.
- **Arithmetic Cold Start** phase (500 epochs) before algebraic complexity is introduced.
- **SQLite Database Integration** seamlessly serializes Natural Language to Math representations.
- **Offline JSON Generation** generates 10,000 algorithmic algebraic identities.
- **Physics Attribution Layer** tags conjectures with likely applied-physics domains.
- **Discovery Declaration Pipeline** promotes low-energy, symbolically-sound conjectures.
- **Hardware-Scaled Runtime Profiles** explicitly bridge local constraints with 2026 hybrid-cloud scale-out nodes.
- **Conversational Context Memory** stores conversational states for continuous vibe context extraction.

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

### Beginner Path Checklist (First Successful Run)

Use this list if you are new to SciOracle and want a safe first success.

1. **Create environment + install deps**
   - `python 3.10+`
   - `pip install torch sympy transformers z3-solver pyyaml fastapi uvicorn pymupdf`
2. **Confirm baseline config**
   - Open `config.yaml`
   - Keep `openclaw.enabled: false` for local-only run
   - Set `scaling.profile: low` for CPU-only machines
3. **Run import smoke check**
   - `python -m compileall ebm_math_discovery.py backend/app.py core_agent.py skills/symbolic_log.py skills/ebm_solve.py`
4. **Initialize with training (small profile)**
   - `python ebm_math_discovery.py --train --cpu`
5. **Try interactive CLI**
   - `python ebm_math_discovery.py`
   - Example:
     - Problem: `x squared plus 5x plus 6`
     - Solution: `(x + 2) times (x + 3)`
6. **Start API server**
   - `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
   - Check `GET /api/status`
7. **Run one research query**
   - `POST /api/query`
   - Then inspect `GET /api/research/graph`
8. **Enable OpenClaw only after local success**
   - Set `openclaw.enabled: true`
   - Fill `base_url`, `agent_id`, and endpoint paths
9. **Inspect output artifacts**
   - `math_knowledge.db`
   - `discoveries/math_log.jsonl`
   - `discoveries/proof_attempts.jsonl`
   - `discoveries/discovery_notifications.jsonl`

### Developer Architecture Diagram

For a full runtime diagram with dataflow and integration boundaries, see:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

### Local Training Deployment Architecture

For a codified local deployment stack (trainer + API + orchestrator), see:

- [`docs/LOCAL_TRAINING_DEPLOYMENT.md`](docs/LOCAL_TRAINING_DEPLOYMENT.md)
- `docker-compose.local-training.yml`
- `requirements.txt`

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

### 2. Launching the Graphic Vibe Interface (React UI)
To launch the modern 2026 EBM Dashboard where you can visually inspect node graphs and input vibes natively:
1. Ensure your backend FastAPI server is running:
   ```bash
   python backend/app.py
   ```
2. Open a new terminal and navigate to the `web-ui` directory.
   ```bash
   cd web-ui
   npm install
   npm run dev
   ```
3. **How to navigate and use it**: 
   * Open your browser to the local network port provided by Vite (`http://localhost:5173`).
   * **PDF Ingestion Zone**: Drag and drop scientific documents to systematically extract mathematical graphics and structural derivations.
   * **EBM Discovery Shell**: Type your natural language scientific intent (e.g., "describe force acting on an object's mass") into the **Vibe Coding Intent** panel.
   * Click **Evaluate Structural Energy** to invoke the dual-encoder models. The React frontend will wait for the EBM solver to ping back the sound energy validation via the backend API.

### 3. Launching the Proof-of-Concept IDE Plugin (ACP Simulator)
We provide a standalone Python POC that simulates how an external IDE (like Zed or VS Code) bridges into the SciOracle validation engine using **Agent Context Protocol (ACP)** WebSockets.
1. Ensure `python backend/app.py` is running natively in one terminal.
2. In a separate terminal, launch the simulator:
   ```bash
   pip install websockets
   python ide_acp_simulator.py
   ```
3. Type any vibe (e.g. "I want an equation for Kinetic Energy"). 
4. Watch as the terminal mirrors the bi-directional IDE WebSockets! The `Planner_Agent`, `Symbolic_Critic`, and `Executor_Agent` logic faults, energy scores, and Cryptographic Hash Signatures stream directly back to your "IDE" terminal.

## Deployment and Orchestration

### 1. OpenClaw Direct Interface (OpenDeepClaw Bridge)
SciOracle operates an autonomous bridging module (`openclaw_interface.py`) to connect the isolated mathematical validation sandbox to a broader, global conversational OpenClaw orchestrator.

**How to use the OpenClaw aspects:**
1. Open your `config.yaml` file and locate the root `openclaw` block.
2. Flip the state:
   - `enabled: true`
   - Map `base_url` to wherever your OpenDeepClaw API suite is deployed (e.g., `http://127.0.0.1:8080`).
3. Start the Master Execution Loop:
   ```bash
   python core_agent.py
   ```
4. **Behavior**: SciOracle will immediately begin pushing persistent `.oracle` heartbeat diffs dynamically to the global framework. If the global OpenClaw LLM commands a `set_conjecture`, the local SciOracle hardware forces an isolated validation sandbox, runs the Symbolic Critic, and securely loops the verified cryptographic handshake hash back up to the Global LLM.

### Research & Development Graphical Analysis Workflow

SciOracle now supports research-oriented databasing and discovery declarations:

1. Every conjecture evaluation stores:
   - energy,
   - symbolic soundness,
   - physics-domain attribution,
   - conjecture signature for novelty tracking.
2. Symbolically sound, low-energy discoveries are declared as theorem/law candidates and written to:
   - `discoveries/discovery_notifications.jsonl`
3. Query research timeline and domain distribution from API:
   ```bash
   GET /api/research/graph
   ```
4. Query derivation lineage graph:
   ```bash
   GET /api/research/lineage
   ```
5. Use conversational research endpoint:
   ```bash
   POST /api/chat
   ```
   This updates `state.json` context memory and returns domain inference + analogical candidate formulas.

### Scalable Compute Configuration (Any Computer)

Use the `scaling` block in `config.yaml` to adapt runtime and training footprint to available hardware:

- `profile: low` for CPU-only or low-memory machines.
- `profile: medium` for consumer GPUs and mixed workloads.
- `profile: high` for large GPU servers.

The profile automatically controls model width/depth, dataset sizes, epochs, batch sizes, and Langevin steps in `train_ebm(...)`, while agent loop delays and backend retrain intervals are also configurable for throughput tuning.

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

### 3. Cloud-Based Scalable Solutions for High Computation

When you want to evaluate equations much larger than 6GB of VRAM allows, or you want to process millions of algorithmic identities faster than a local GPU can handle, you can migrate to the cloud.

#### 1. GPU-Optimized IaaS (AWS EC2, GCP Compute Engine)
- **The Setup:** Rent a Virtual Machine equipped with enterprise GPUs (e.g., NVIDIA A10g, A100, or H100) and substantial system RAM (128GB+).
- **Execution:** Deploy your exact Docker container onto this VM. Because it's a dedicated machine, you can run Ollama locally on the VM to retain the "Local LLM Backend Privacy" rule (your data stays on your isolated server).
- **Benefit:** You can modify `config.yaml` to remove the `vram_gate` limit (or raise it to 40GB/80GB). This allows the Math-EBM's Langevin Dynamics to run massively parallel batches natively on the GPU without CPU fallback.

#### 2. Cost-Effective GPU Compute Clusters (RunPod, Vast.ai, Lambda Labs)
- **The Setup:** These platforms rent consumer and workstation GPUs (like RTX 4090s, A6000s) at a fraction of the cost of AWS/GCP.
- **Execution:** RunPod allows you to deploy a Docker image directly to a GPU instance. You can map a cloud volume to the `/app/discoveries` folder to persist your EBM weights and SQLite databases (`math_knowledge.db`).
- **Benefit:** Extremely cost-efficient scaling. You get 24GB-48GB of VRAM for cents on the dollar, allowing you to train on the 10,000 JSON identities exponentially faster.

#### 3. Kubernetes Orchestration (EKS, GKE) for Distributed Agents
- **The Setup:** If you intend to run *multiple* OpenClaw agents simultaneously (e.g., one agent tackling geometry, another tackling algebra).
- **Execution:** Deploy the SciOracle container as a Kubernetes `StatefulSet` attached to persistent volumes. You can route computation tasks using a queue system (like RabbitMQ or Redis) to distribute SymPy validation workloads across dozens of CPU-only nodes, while funneling the Math-EBM training to a dedicated GPU Node Pool.

#### 4. Hybrid Cloud API Approach (Computation vs. Privacy Trade-off)
- **The Setup:** Offload the Heavy LLM (`qwen2.5-coder:32b`) inference to a managed cloud provider (e.g., changing the OpenClaw backend to use the OpenAI API, Anthropic, or an Ollama instance hosted on Azure).
- **Execution:** Your local machine (or a cheap cloud VM) only handles the Math-EBM and SymPy AST verifications, while the heavy lifting of natural language and code generation happens via API.
- **Trade-off:** This completely bypasses the need for massive LLM VRAM, freeing up your entire GPU for the Math-EBM. However, it violates the strict data privacy constraint, as your mathematical states and queries will be sent over the network to the LLM provider.
