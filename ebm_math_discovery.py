import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import random
import os
import re
import sqlite3
import json
import argparse
import hashlib
from datetime import datetime
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

# Task 1: Energy Network PyTorch class (GNN Version)

class GCNLayer(nn.Module):
    """
    Standard Graph Convolutional Network Layer
    H_{l+1} = GELU(A_adj @ H_l W)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: Node features [Batch, Nodes, Features]
        adj: Normalized Adjacency Matrix [Batch, Nodes, Nodes]
        """
        # Linear transformation of features
        h = self.linear(x) # [B, N, Dout]
        
        # Message passing from neighbors
        out = torch.bmm(adj, h) # [B, N, Dout]
        return F.gelu(out)

class MathEBM(nn.Module):
    def __init__(self, llm_vocab_size, math_vocab_size, d_model=256, num_layers=4):
        """
        Energy-Based Model for mathematical symbolic discovery using Graph Neural Networks.
        Designed to run on 6GB VRAM by using a small dimension GNN 
        and gradient checkpointing. Incorporates a Self-Improvement Cross-Attention Architecture.
        """
        super().__init__()
        self.d_model = d_model
        
        # Dual Embeddings for Language (LLM) and Math (AST)
        self.llm_embedding = nn.Embedding(llm_vocab_size, d_model)
        self.math_embedding = nn.Embedding(math_vocab_size, d_model)
        
        # Graph Neural Network Encoders (Shared weights for structural processing)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(d_model, d_model) for _ in range(num_layers)
        ])
        
        # Self-Improvement Neural Architecture (Cross-Attention)
        # Allows Language Sequence to attend to Mathematical Graph Nodes
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Scalar Energy Head (E_theta) evaluates the combined problem and solution
        self.energy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def encode_graph_nodes(self, node_emb, adj):
        """Processes the graph through the GCN layers with checkpointing, returning node embeddings"""
        h = node_emb
        for gcn in self.gcn_layers:
            # Memory Optimization: Iteratively checkpoint each graph layer to drastically reduce peak VRAM
            if self.training and h.requires_grad:
                h = checkpoint(gcn, h, adj, use_reentrant=False)
            else:
                h = gcn(h, adj)
        return h

    def forward(self, x_nodes, x_adj, y_soft_nodes, y_adj):
        """
        Calculates the Energy of a given Language Problem (x) and Proposed Math Graph (y_soft).
        x_nodes: [B, max_nodes] - Problem Language sequence tokens (discrete)
        x_adj: [B, max_nodes, max_nodes] - Adjacency matrix for 1D language graph
        y_soft_nodes: [B, max_nodes, math_vocab_size] - Solution AST soft-tokens (bridge continuous)
        y_adj: [B, max_nodes, max_nodes] - Adjacency matrix for solution math graph
        """
        # Embed discrete language tokens
        x_emb = self.llm_embedding(x_nodes) # [B, N_x, d_model]
        
        # Embed continuous solution soft-tokens (Discrete-to-Continuous Bridge for Math)
        y_emb = torch.matmul(y_soft_nodes, self.math_embedding.weight) # [B, N_y, d_model]
        
        # Encode both graphs into node features
        h_x_nodes = self.encode_graph_nodes(x_emb, x_adj) # [B, N_x, d_model]
        h_y_nodes = self.encode_graph_nodes(y_emb, y_adj) # [B, N_y, d_model]
        
        # Self-Improvement Cross-Attention (Language querying Math)
        attn_output, _ = self.cross_attention(query=h_x_nodes, key=h_y_nodes, value=h_y_nodes)
        
        # Global mean pooling
        h_x = attn_output.mean(dim=1) # [B, d_model]
        h_y = h_y_nodes.mean(dim=1)   # [B, d_model]
        
        # Combine embeddings and output scalar energy
        out = torch.cat([h_x, h_y], dim=-1) # [B, 2 * d_model]
        energy = self.energy_head(out).squeeze(-1)
        return energy


# Task 2: MCMC Langevin Sampling Loop for Graph node optimization

def sample_langevin(model, x_nodes, x_adj, y_logits_init, y_adj, steps=20, step_size=0.1, temp=1.0):
    """
    Performs gradient-based Langevin Dynamics to optimize the continuous
    representation (y_logits) by minimizing the predicted energy E(x, y).

    We optimize the graph's node features continuously, keeping the tree's adjacency structure fixed.
    """
    # Create parameter to optimize over without disrupting model tracking
    y_logits = y_logits_init.clone().detach().requires_grad_(True)
    
    # We use SGD strictly to apply the pure gradient step manually and cleanly
    optimizer = torch.optim.SGD([y_logits], lr=step_size)
    
    # Freeze model weights during generation
    model.eval() 
    
    for step in range(steps):
        optimizer.zero_grad()
        annealed_temp = max(0.3, temp * (0.98 ** step))
        
        # Gumbel-Softmax discrete-to-continuous bridge using the proposed logits
        y_soft = F.gumbel_softmax(y_logits, tau=annealed_temp, hard=False)
        
        # Energy Forward Pass
        energy = model(x_nodes, x_adj, y_soft, y_adj)
        
        # We want to MINIMIZE energy, hence no negative sign needed for gradient backward
        loss = energy.sum()
        loss.backward()

        with torch.no_grad():
            grad_norm = y_logits.grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            y_logits.grad.div_(grad_norm)
        
        # Apply standard gradient step: theta = theta - step_size * grad
        optimizer.step()
        
        # Inject standard Langevin noise term
        with torch.no_grad():
            noise = torch.randn_like(y_logits) * (step_size ** 0.5)
            y_logits.add_(noise)
            
    return y_logits.detach()


# Task 3: Tokenizers and SymPy dataset generator 

class ASTGraphTokenizer:
    """Tokenizer to convert SymPy AST into Node Lists and Adjacency Matrices"""
    def __init__(self):
        self.vocab = {"PAD": 0, "UNK": 1}
        self.inv_vocab = {0: "PAD", 1: "UNK"}
        self.vocab_size = 2

    def add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.vocab_size
            self.inv_vocab[self.vocab_size] = token
            self.vocab_size += 1

    def parse_ast(self, expr):
        """Returns node labels and a list of edges (parent, child)."""
        nodes = []
        edges = []
        
        def traverse(node):
            node_id = len(nodes)
            
            if isinstance(node, sp.Symbol) or isinstance(node, sp.Integer) or isinstance(node, sp.Rational):
                nodes.append(str(node))
            else:
                op = node.__class__.__name__
                nodes.append(op)
                for arg in node.args:
                    child_id = traverse(arg)
                    edges.append((node_id, child_id))
                    # Make graph undirected for better message passing
                    edges.append((child_id, node_id))
            return node_id
            
        traverse(expr)
        return nodes, edges

    def encode_graph(self, expr, max_nodes=30):
        nodes, edges = self.parse_ast(expr)
        
        # Update dynamic vocabulary
        for n in nodes:
            self.add_token(n)
            
        node_ids = [self.vocab.get(n, self.vocab["UNK"]) for n in nodes]
        
        # Construct dense adjacency matrix
        adj = torch.zeros(max_nodes, max_nodes)
        
        # Add self-loops to maintain current node features during message passing
        for i in range(min(len(nodes), max_nodes)):
            adj[i, i] = 1.0
            
        for u, v in edges:
            if u < max_nodes and v < max_nodes:
                adj[u, v] = 1.0
                
        # Degree Normalization D^-1 A
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / torch.clamp(row_sum, min=1e-8)
        
        # Pad nodes sequence
        if len(node_ids) < max_nodes:
            node_ids += [self.vocab["PAD"]] * (max_nodes - len(node_ids))
        else:
            node_ids = node_ids[:max_nodes]
            
        return torch.tensor(node_ids, dtype=torch.long), adj

class LLMSeqTokenizer:
    """Tokenizer to convert expressions into LLM Token Sequences and 1D Adjacency Matrices"""
    def __init__(self, model_id="gpt2"):
        # We use a standard HuggingFace tokenizer. GPT-2 by default.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode_graph(self, text, max_nodes=50):
        # We use padding="max_length" to pad directly to max_nodes
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            max_length=max_nodes, 
            padding="max_length", 
            return_tensors="pt"
        )
        node_ids = tokens["input_ids"][0]
        
        # Construct sequence adjacency matrix (treating the sequence as a 1D graph)
        adj = torch.zeros(max_nodes, max_nodes)
        
        # Add self-loops and sequence edges (i-1 <-> i <-> i+1)
        # We only add edges up to the actual sequence length, the rest are just self loops for padding
        seq_len = min(len(self.tokenizer(text, truncation=True, max_length=max_nodes)["input_ids"]), max_nodes)
        
        for i in range(max_nodes):
            adj[i, i] = 1.0 # Self Loop
            if i < seq_len:
                if i > 0:
                    adj[i, i-1] = 1.0
                if i < seq_len - 1:
                    adj[i, i+1] = 1.0
                
        # Degree Normalization D^-1 A
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / torch.clamp(row_sum, min=1e-8)
        
        return node_ids.long(), adj

def sympy_to_nl_str(expr):
    """Assigns proper natural language reading to a SymPy equation."""
    text = str(expr)
    replacements = {
        "**3": " cubed ",
        "**2": " squared ",
        "**": " to the power of ",
        "*": " times ",
        "+": " plus ",
        "- ": " minus ",
        "/": " divided by ",
        "=": " equals "
    }
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.capitalize()

def generate_sympy_data(num_samples=100, max_nodes=50, mode="algebraic", llm_tokenizer=None, ast_tokenizer=None):
    """
    Generate dataset of Dual-Encoder correct pairs and adversarial mutations using SymPy.
    Supports 'arithmetic' for cold starts and 'algebraic' for complex identities.
    """
    if llm_tokenizer is None:
        llm_tokenizer = LLMSeqTokenizer()
    if ast_tokenizer is None:
        ast_tokenizer = ASTGraphTokenizer()
        
    x = sp.Symbol('x')
    dataset = []
    raw_json_data = []
    
    for _ in range(num_samples):
        if mode == "arithmetic":
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            op_choice = random.choice(["add", "mul", "sub"])
            
            if op_choice == "add":
                expanded = sp.Add(a, b, evaluate=False)
                factored = sp.sympify(a + b)
            elif op_choice == "mul":
                expanded = sp.Mul(a, b, evaluate=False)
                factored = sp.sympify(a * b)
            else:
                expanded = sp.Add(a, -b, evaluate=False)
                factored = sp.sympify(a - b)
                
            adversarial = factored + random.randint(1, 5)
        else:
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            factored = (x + a) * (x + b)
            expanded = sp.expand(factored)
            
            mutation_type = random.choice([1, 2, 3])
            if mutation_type == 1:
                adversarial = (x - a) * (x + b) # Wrong sign
            elif mutation_type == 2:
                adversarial = (x + a + 1) * (x + b) # Wrong constant
            else:
                adversarial = (x + b) * (x + b) # Duplicate term

        # Problem is Natural Language sequence encoded by LLM tokenizer
        p_nl = sympy_to_nl_str(expanded)
        problem_nodes, problem_adj = llm_tokenizer.encode_graph(p_nl, max_nodes)
        
        # Solutions are AST graphs encoded by AST Graph tokenizer
        correct_nodes, correct_adj = ast_tokenizer.encode_graph(factored, max_nodes)
        advers_nodes, advers_adj = ast_tokenizer.encode_graph(adversarial, max_nodes)
        
        raw_json_data.append({
            "problem": str(expanded),
            "correct": str(factored),
            "adversarial": str(adversarial),
            "difficulty": float(sp.count_ops(expanded) + 1),
            "symbolic_margin": float(abs(sp.count_ops(factored) - sp.count_ops(adversarial)) + 1)
        })

        difficulty = float(sp.count_ops(expanded) + 1)
        symbolic_margin = float(abs(sp.count_ops(factored) - sp.count_ops(adversarial)) + 1)
        
        dataset.append({
            "problem_expr": expanded,
            "correct_expr": factored,
            "adversarial_expr": adversarial,
            "problem_nodes": problem_nodes,
            "problem_adj": problem_adj,
            "correct_nodes": correct_nodes,
            "correct_adj": correct_adj,
            "adversarial_nodes": advers_nodes,
            "adversarial_adj": advers_adj,
            "difficulty": difficulty,
            "symbolic_margin": symbolic_margin
        })
    
    return dataset, llm_tokenizer, ast_tokenizer, raw_json_data

def save_checkpoint(model, llm_tokenizer, ast_tokenizer, path="math_ebm.pt"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "llm_vocab_size": llm_tokenizer.vocab_size,
        "ast_vocab": ast_tokenizer.vocab,
        "ast_inv_vocab": ast_tokenizer.inv_vocab,
        "ast_vocab_size": ast_tokenizer.vocab_size,
        "llm_embedding_num": model.llm_embedding.num_embeddings,
        "math_embedding_num": model.math_embedding.num_embeddings
    }, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Reconstruct tokenizers
    llm_tokenizer = LLMSeqTokenizer()
    ast_tokenizer = ASTGraphTokenizer()
    ast_tokenizer.vocab = checkpoint["ast_vocab"]
    ast_tokenizer.inv_vocab = checkpoint["ast_inv_vocab"]
    ast_tokenizer.vocab_size = checkpoint["ast_vocab_size"]
    
    # Init model
    model = MathEBM(
        llm_vocab_size=checkpoint["llm_vocab_size"], 
        math_vocab_size=checkpoint["ast_vocab_size"], 
        d_model=256, 
        num_layers=4
    ).to(device)
    model.llm_embedding = nn.Embedding(checkpoint["llm_embedding_num"], 256).to(device)
    model.math_embedding = nn.Embedding(checkpoint["math_embedding_num"], 256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, llm_tokenizer, ast_tokenizer

def infer_physics_application(problem_nl, problem_math, solution_math):
    """Heuristic attribution of conjectures to a physics domain."""
    text = " ".join([str(problem_nl), str(problem_math), str(solution_math)]).lower()
    rules = [
        ("electromagnetism", ["charge", "electric", "magnetic", "maxwell", "coulomb", "voltage", "current", "field"]),
        ("quantum_mechanics", ["hbar", "psi", "schrodinger", "wavefunction", "operator", "eigen", "quantum"]),
        ("relativity", ["einstein", "lorentz", "spacetime", "gamma", "mass energy"]),
        ("thermodynamics", ["entropy", "temperature", "heat", "boltzmann", "thermo", "pressure"]),
        ("fluid_dynamics", ["navier", "stokes", "viscosity", "fluid", "reynolds", "vorticity"]),
        ("classical_mechanics", ["force", "momentum", "newton", "lagrangian", "hamiltonian", "kinetic", "potential"]),
        ("waves_optics", ["wavelength", "frequency", "amplitude", "interference", "diffraction", "optics"]),
    ]
    for domain, keywords in rules:
        if any(k in text for k in keywords):
            return domain, "keyword_heuristic"
    return "symbolic_algebra", "default_symbolic_inference"

def conjecture_signature(problem_math, solution_math):
    canonical = f"{str(problem_math).strip()}=>{str(solution_math).strip()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def record_discovery_notification(notification_path, payload):
    os.makedirs(os.path.dirname(notification_path) or ".", exist_ok=True)
    with open(notification_path, "a") as f:
        f.write(json.dumps(payload) + "\n")

def declare_theorem_if_sound(conn, p_nl, p_math, s_nl, s_math, energy, is_sound, threshold=0.05):
    """Declare theorem/new law candidates for symbolically sound, low-energy discoveries."""
    if not is_sound:
        return None

    energy_value = float(energy) if energy is not None else 0.0
    theorem_status = "theorem_verified" if energy_value <= threshold else "symbolically_verified"
    law_declaration = theorem_status == "theorem_verified"
    physics_domain, attribution = infer_physics_application(p_nl, p_math, s_math)
    signature = conjecture_signature(p_math, s_math)

    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, notification_sent FROM math_discoveries WHERE conjecture_signature = ? ORDER BY id DESC LIMIT 1",
        (signature,),
    )
    row = cursor.fetchone()

    if row and row[1]:
        return {
            "theorem_status": theorem_status,
            "law_declaration": bool(law_declaration),
            "physics_domain": physics_domain,
            "attribution": attribution,
            "signature": signature,
            "notification": "already_sent",
        }

    message = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "problem_nl": str(p_nl),
        "problem_math": str(p_math),
        "solution_math": str(s_math),
        "energy": energy_value,
        "theorem_status": theorem_status,
        "law_declaration": bool(law_declaration),
        "physics_domain": physics_domain,
        "attribution": attribution,
        "conjecture_signature": signature,
        "notification": "NEW_DISCOVERY",
    }

    if row:
        cursor.execute(
            '''
            UPDATE math_discoveries
            SET theorem_status = ?,
                law_declaration = ?,
                physics_domain = ?,
                attribution = ?,
                conjecture_signature = ?,
                notification_sent = 1
            WHERE id = ?
            ''',
            (theorem_status, bool(law_declaration), physics_domain, attribution, signature, row[0]),
        )
    else:
        cursor.execute(
            '''
            INSERT INTO math_discoveries (
                problem_nl, problem_math, solution_nl, solution_math, energy, is_sound,
                theorem_status, law_declaration, physics_domain, attribution,
                conjecture_signature, notification_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                str(p_nl), str(p_math), str(s_nl), str(s_math), energy_value, bool(is_sound),
                theorem_status, bool(law_declaration), physics_domain, attribution,
                signature, True,
            ),
        )

    conn.commit()
    record_discovery_notification("discoveries/discovery_notifications.jsonl", message)
    return message

def init_db(db_path="math_knowledge.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS math_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_nl TEXT,
            problem_math TEXT,
            solution_nl TEXT,
            solution_math TEXT,
            energy REAL,
            is_sound BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            theorem_status TEXT DEFAULT 'unverified',
            law_declaration BOOLEAN DEFAULT 0,
            physics_domain TEXT,
            attribution TEXT,
            conjecture_signature TEXT,
            notification_sent BOOLEAN DEFAULT 0,
            image_path TEXT
        )
    ''')

    required_columns = {
        "theorem_status": "TEXT DEFAULT 'unverified'",
        "law_declaration": "BOOLEAN DEFAULT 0",
        "physics_domain": "TEXT",
        "attribution": "TEXT",
        "conjecture_signature": "TEXT",
        "notification_sent": "BOOLEAN DEFAULT 0",
        "image_path": "TEXT",
    }
    cursor.execute("PRAGMA table_info(math_discoveries)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, col_type in required_columns.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE math_discoveries ADD COLUMN {col} {col_type}")

    conn.commit()
    return conn

def log_to_db(conn, p_nl, p_math, s_nl, s_math, energy, is_sound, image_path=None):
    cursor = conn.cursor()
    domain, attribution = infer_physics_application(p_nl, p_math, s_math)
    signature = conjecture_signature(p_math, s_math)
    cursor.execute('''
        INSERT INTO math_discoveries 
        (problem_nl, problem_math, solution_nl, solution_math, energy, is_sound,
         theorem_status, law_declaration, physics_domain, attribution,
         conjecture_signature, notification_sent, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(p_nl), str(p_math), str(s_nl), str(s_math), float(energy) if energy is not None else 0.0, bool(is_sound),
        "unverified", False, domain, attribution,
        signature, False, image_path
    ))
    conn.commit()

def nl_to_sympy_str(text):
    """
    Very basic Natural Language to Math string converter.
    For production, hook this up to a local LLM parser.
    """
    text = text.lower()
    replacements = {
        "squared": "**2",
        "cubed": "**3",
        "to the power of": "**",
        "plus": "+",
        "minus": "-",
        "times": "*",
        "divided by": "/",
        "equals": "=",
        " equals ": "=",
        " and ": " "
    }
    for word, symbol in replacements.items():
        text = text.replace(word, symbol)
    # Fix instances like "2 x" -> "2*x" or "5x" -> "5*x"
    text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)
    return text

def train_ebm(save_path="math_ebm.pt", db_conn=None, use_cpu=False):
    if db_conn is None:
        db_conn = init_db()
        
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Training on device: {device}")
    
    MAX_NODES = 50 # Increased max_nodes to accommodate larger generated trees
    llm_tokenizer = LLMSeqTokenizer()
    ast_tokenizer = ASTGraphTokenizer()
    
    print("Generating Cold Start Arithmetic Dataset...")
    arith_dataset, llm_tokenizer, ast_tokenizer, _ = generate_sympy_data(2000, max_nodes=MAX_NODES, mode="arithmetic", llm_tokenizer=llm_tokenizer, ast_tokenizer=ast_tokenizer)
    
    print(f"Generating 10,000 Algebraic Identities Dataset... (SymPy executing on CPU)")
    algeb_dataset, llm_tokenizer, ast_tokenizer, raw_json_data = generate_sympy_data(10000, max_nodes=MAX_NODES, mode="algebraic", llm_tokenizer=llm_tokenizer, ast_tokenizer=ast_tokenizer)
    
    json_path = "algebraic_identities.json"
    with open(json_path, "w") as f:
        json.dump(raw_json_data, f, indent=4)
    print(f"Saved 10,000 basic algebraic identities to {json_path}")
    
    print("Logging mathematically sound ground-truth generation to knowledge database...")
    # Log subset to db for brevity
    for item in algeb_dataset[:100]:
        p_math = str(item["problem_expr"])
        s_math = str(item["correct_expr"])
        p_nl = sympy_to_nl_str(item["problem_expr"])
        s_nl = sympy_to_nl_str(item["correct_expr"])
        # Log to db, energy initially 0.0 or lowest theoretical bound
        log_to_db(db_conn, p_nl, p_math, s_nl, s_math, 0.0, True)
    
    # Target 6GB VRAM constraint Memory Optimizations
    BATCH_SIZE = 16 
    GRAD_ACCUM_STEPS = 4 
    
    # 256 dim fits comfortably within 6GB threshold
    model = MathEBM(
        llm_vocab_size=llm_tokenizer.vocab_size, 
        math_vocab_size=ast_tokenizer.vocab_size, 
        d_model=256, 
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) # Fixed Pyright warn
    
    # Resize embedding layers to add buffering padding space for new tokens safely
    model.llm_embedding = nn.Embedding(llm_tokenizer.vocab_size + 100, 256).to(device)
    model.math_embedding = nn.Embedding(ast_tokenizer.vocab_size + 100, 256).to(device)
    
    # Self-Improvement Buffer (Experience Replay for network's own dynamic mathematical discoveries)
    self_improvement_buffer = []

    def run_training_loop(dataset, epochs, phase_name):
        print(f"--- Starting {phase_name} ({epochs} Epochs) ---")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            # Combine generated self-improvement discoveries with the original epoch stream
            active_dataset = dataset + self_improvement_buffer
            random.shuffle(active_dataset)
            
            for i in range(0, len(active_dataset), BATCH_SIZE):
                batch = active_dataset[i:i+BATCH_SIZE]
                
                x_nodes = torch.stack([item["problem_nodes"] for item in batch]).to(device)
                x_adj = torch.stack([item["problem_adj"] for item in batch]).to(device)
                
                y_pos_discrete = torch.stack([item["correct_nodes"] for item in batch]).to(device)
                y_pos_adj = torch.stack([item["correct_adj"] for item in batch]).to(device)
                
                y_pos_soft = F.one_hot(y_pos_discrete, num_classes=model.math_embedding.num_embeddings).float()
                
                y_neg_init_discrete = torch.stack([item["adversarial_nodes"] for item in batch]).to(device)
                y_neg_adj = torch.stack([item["adversarial_adj"] for item in batch]).to(device)
                
                y_neg_init = F.one_hot(y_neg_init_discrete, num_classes=model.math_embedding.num_embeddings).float()
                y_neg_init = y_neg_init * 5.0 + torch.randn_like(y_neg_init) 
                
                # Langevin dynamics generates continuous soft-token landscape updates mathematically
                y_neg_logits = sample_langevin(model, x_nodes, x_adj, y_neg_init, y_neg_adj, steps=15, step_size=0.1, temp=1.0)
                y_neg_soft = F.gumbel_softmax(y_neg_logits, tau=1.0, hard=False)
                
                # SELF IMPROVEMENT NEURAL ARCHITECTURE (Algorithmic Verification)
                # Check if the network accidentally proved a problem logically correct during gradient walking:
                with torch.no_grad():
                    # For a primitive algorithmic heuristic check matching exact true topologies:
                    # (In a hyper-advanced system, this would explicitly walk the Gumbel logits against SymPy)
                    # We inject matching logic into the Replay Buffer here.
                    predicted_argmax = y_neg_soft.argmax(dim=-1)
                    for j in range(len(batch)):
                        if torch.equal(predicted_argmax[j], y_pos_discrete[j]):
                            # The model has successfully proven/discovered a valid configuration during Langevin
                            if len(self_improvement_buffer) < 500: # Limit size
                                self_improvement_buffer.append(batch[j])
                
                model.train() 
                
                pos_energy = model(x_nodes, x_adj, y_pos_soft, y_pos_adj)
                neg_energy = model(x_nodes, x_adj, y_neg_soft, y_neg_adj)
                
                sample_difficulty = torch.tensor(
                    [item.get("difficulty", 1.0) for item in batch],
                    device=device,
                    dtype=pos_energy.dtype
                )
                sample_margin = torch.tensor(
                    [item.get("symbolic_margin", 1.0) for item in batch],
                    device=device,
                    dtype=pos_energy.dtype
                )

                # Margin grows with symbolic structural gap, making harder contrasts contribute more.
                adaptive_margin = 0.2 + 0.05 * torch.log1p(sample_margin)
                contrastive = F.relu(adaptive_margin + pos_energy - neg_energy)
                weighted_contrastive = (contrastive * torch.log1p(sample_difficulty)).mean()

                energy_reg = 0.05 * (pos_energy.pow(2) + neg_energy.pow(2)).mean()
                loss = weighted_contrastive + energy_reg
                
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                
                total_loss += loss.item() * GRAD_ACCUM_STEPS
                
                if (i // BATCH_SIZE + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            avg_loss = total_loss / max(1, (len(active_dataset)//BATCH_SIZE))
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | CD Loss: {avg_loss:.4f} | Replay Buffer (Self-Discovered): {len(self_improvement_buffer)}")

    # Phase 1: 500 epochs on Arithmetic
    run_training_loop(arith_dataset, 500, "Cold Start (Arithmetic)")
    
    # Phase 2: 10 epochs on Algebraic dataset
    run_training_loop(algeb_dataset, 10, "Discovery Phase (Algebraic)")
        
    print(f"Saving model checkpoint to {save_path}...")
    save_checkpoint(model, llm_tokenizer, ast_tokenizer, save_path)
    print("Training complete.")

def evaluate_energy(model, llm_tokenizer, ast_tokenizer, problem_nl, solution_str, device):
    """Parses arbitrary strings into LLM/ASTs and returns the model's energy assigned to the pair."""
    try:
        # Solution has to be math logic evaluation
        solution_expr = sp.sympify(solution_str)
        
        # Dual Encoding logic
        p_nodes, p_adj = llm_tokenizer.encode_graph(problem_nl, max_nodes=50)
        s_nodes, s_adj = ast_tokenizer.encode_graph(solution_expr, max_nodes=50)
        
        p_nodes = p_nodes.unsqueeze(0).to(device)
        p_adj = p_adj.unsqueeze(0).to(device)
        s_nodes = s_nodes.unsqueeze(0).to(device)
        s_adj = s_adj.unsqueeze(0).to(device)
        
        # Continuous bridge formatting over the Math Network
        s_soft = F.one_hot(s_nodes, num_classes=model.math_embedding.num_embeddings).float()
        
        with torch.no_grad():
            energy = model(p_nodes, p_adj, s_soft, s_adj)
        return energy.item()
    except Exception as e:
        return f"Error parsing equations: {str(e)}"

def interactive_interface():
    print("=== Mathematical EBM Interface (Dual-Encoder Architecture) ===")
    model_path = "math_ebm.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    db_conn = init_db()
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint '{model_path}' not found. Starting training loop first...")
        train_ebm(save_path=model_path, db_conn=db_conn)
        
    print("Loading model weights...")
    model, llm_tokenizer, ast_tokenizer = load_checkpoint(model_path, device)
    print("Model ready!")
    print("Instructions:")
    print("- Enter a 'Problem' (e.g., 'x squared plus 5x plus 6')")
    print("- Enter a 'Proposed Solution' (e.g., '(x + 2) times (x + 3)')")
    print("- Leave blank to exit.\n")
    
    while True:
        try:
            p_text = input("Problem (Natural Language): ").strip()
            if not p_text: break
            s_text = input("Proposed Solution (Natural Language or Math): ").strip()
            if not s_text: break
            
            p_math_guess = nl_to_sympy_str(p_text)
            s_math = nl_to_sympy_str(s_text)
            
            # Formats
            try:
                # Check for Mathematical correctness purely through SymPy 
                is_sound = (sp.simplify(sp.sympify(p_math_guess) - sp.sympify(s_math)) == 0)
            except Exception as e:
                is_sound = False
            
            print(f"\nNatural Language (Problem): {p_text}")
            print(f"Parsed Math (Problem Hint): {p_math_guess}")
            print(f"Natural Language (Sol):     {s_text}")
            print(f"Parsed Math (Sol):          {s_math}")
            
            energy = evaluate_energy(model, llm_tokenizer, ast_tokenizer, p_text, s_math, device)
            
            if isinstance(energy, str):
                print(f"[!] {energy}\n")
            else:
                print(f"==> EBM Predicted Dual-Architecture Energy: {energy:.4f}")
                print(f"    (Mathematically sound logically: {str(is_sound).upper()})")
                print("    (Saved to knowledge database)\n")
                
                log_to_db(db_conn, p_text, p_math_guess, s_text, s_math, energy, is_sound)
                declaration = declare_theorem_if_sound(db_conn, p_text, p_math_guess, s_text, s_math, energy, is_sound)
                if declaration and declaration.get("notification") == "NEW_DISCOVERY":
                    print(
                        f"    [Discovery] {declaration['theorem_status']} in {declaration['physics_domain']} "
                        f"(signature: {declaration['signature'][:12]})"
                    )
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mathematical Energy-Based Model")
    parser.add_argument("--train", action="store_true", help="Run the full training pipeline (Arithmetic Cold Start + Algebraic Discovery)")
    parser.add_argument("--cpu", action="store_true", help="Force execution on CPU for large tree sizes exceeding VRAM")
    args = parser.parse_args()
    
    if args.train:
        train_ebm(use_cpu=args.cpu)
    else:
        interactive_interface()
