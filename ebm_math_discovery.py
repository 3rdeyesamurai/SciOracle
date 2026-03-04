import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import random
import os
import re
import sqlite3
from torch.utils.checkpoint import checkpoint

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
    def __init__(self, vocab_size, d_model=256, num_layers=4):
        """
        Energy-Based Model for mathematical symbolic discovery using Graph Neural Networks.
        Designed to run on 6GB VRAM by using a small dimension GNN 
        and gradient checkpointing.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Linear embedding layer for both discrete tokens and continuous bridge
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Graph Neural Network Encoder
        self.gcn_layers = nn.ModuleList([
            GCNLayer(d_model, d_model) for _ in range(num_layers)
        ])
        
        # Scalar Energy Head (E_theta) evaluates the combined problem and solution
        self.energy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def encode_graph(self, node_emb, adj):
        """Processes the graph through the GCN layers with checkpointing"""
        h = node_emb
        for gcn in self.gcn_layers:
            # Memory Optimization: Iteratively checkpoint each graph layer to drastically reduce peak VRAM
            if self.training and h.requires_grad:
                h = checkpoint(gcn, h, adj, use_reentrant=False)
            else:
                h = gcn(h, adj)
                
        # Global mean pooling to derive a single vector for the entire graph
        return h.mean(dim=1)

    def forward(self, x_nodes, x_adj, y_soft_nodes, y_adj):
        """
        Calculates the Energy of a given Problem (x) and Proposed Solution Graph (y_soft).
        x_nodes: [B, max_nodes] - Problem AST tokens (discrete)
        x_adj: [B, max_nodes, max_nodes] - Adjacency matrix for problem graph
        y_soft_nodes: [B, max_nodes, vocab_size] - Solution AST soft-tokens (bridge continuous)
        y_adj: [B, max_nodes, max_nodes] - Adjacency matrix for solution graph (fixed structure)
        """
        # Embed discrete problem tokens
        x_emb = self.embedding(x_nodes) # [B, N, d_model]
        
        # Embed continuous solution soft-tokens (Discrete-to-Continuous Bridge)
        y_emb = torch.matmul(y_soft_nodes, self.embedding.weight) # [B, N_y, d_model]
        
        # Encode both graphs
        h_x = self.encode_graph(x_emb, x_adj) # [B, d_model]
        h_y = self.encode_graph(y_emb, y_adj) # [B, d_model]
        
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
        
        # Gumbel-Softmax discrete-to-continuous bridge using the proposed logits
        y_soft = F.gumbel_softmax(y_logits, tau=temp, hard=False)
        
        # Energy Forward Pass
        energy = model(x_nodes, x_adj, y_soft, y_adj)
        
        # We want to MINIMIZE energy, hence no negative sign needed for gradient backward
        loss = energy.sum()
        loss.backward()
        
        # Apply standard gradient step: theta = theta - step_size * grad
        optimizer.step()
        
        # Inject standard Langevin noise term
        with torch.no_grad():
            noise = torch.randn_like(y_logits) * torch.sqrt(torch.tensor(step_size))
            y_logits.add_(noise)
            
    return y_logits.detach()


# Task 3: Training loop and SymPy dataset generator 

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

def generate_sympy_data(num_samples=100, max_nodes=30):
    """
    Generate dataset of correct pairs and adversarial mutations using SymPy.
    """
    x = sp.Symbol('x')
    tokenizer = ASTGraphTokenizer()
    dataset = []
    
    for _ in range(num_samples):
        # Generate random identity: factored form <-> expanded polynomial
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        factored = (x + a) * (x + b)
        expanded = sp.expand(factored)
        
        problem_nodes, problem_adj = tokenizer.encode_graph(expanded, max_nodes)
        correct_nodes, correct_adj = tokenizer.encode_graph(factored, max_nodes)
        
        # Create an 'Adversarial' mutation (e.g., incorrect factor)
        mutation_type = random.choice([1, 2, 3])
        if mutation_type == 1:
            adversarial = (x - a) * (x + b) # Wrong sign
        elif mutation_type == 2:
            adversarial = (x + a + 1) * (x + b) # Wrong constant
        else:
            adversarial = (x + b) * (x + b) # Duplicate term

        advers_nodes, advers_adj = tokenizer.encode_graph(adversarial, max_nodes)
        
        dataset.append({
            "problem_expr": expanded,
            "correct_expr": factored,
            "adversarial_expr": adversarial,
            "problem_nodes": problem_nodes,
            "problem_adj": problem_adj,
            "correct_nodes": correct_nodes,
            "correct_adj": correct_adj,
            "adversarial_nodes": advers_nodes,
            "adversarial_adj": advers_adj
        })
    
    return dataset, tokenizer

def save_checkpoint(model, tokenizer, path="math_ebm.pt"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "inv_vocab": tokenizer.inv_vocab,
        "vocab_size": tokenizer.vocab_size,
        "embedding_num": model.embedding.num_embeddings
    }, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Reconstruct tokenizer
    tokenizer = ASTGraphTokenizer()
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.inv_vocab = checkpoint["inv_vocab"]
    tokenizer.vocab_size = checkpoint["vocab_size"]
    
    # Init model
    model = MathEBM(vocab_size=1000, d_model=256, num_layers=4).to(device)
    model.embedding = nn.Embedding(checkpoint["embedding_num"], 256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def log_to_db(conn, p_nl, p_math, s_nl, s_math, energy, is_sound):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO math_discoveries 
        (problem_nl, problem_math, solution_nl, solution_math, energy, is_sound)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (str(p_nl), str(p_math), str(s_nl), str(s_math), float(energy) if energy is not None else 0.0, bool(is_sound)))
    conn.commit()

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

def train_ebm(save_path="math_ebm.pt", db_conn=None):
    if db_conn is None:
        db_conn = init_db()
        
    print("Generating SymPy dataset of correct identities and adversarials...")
    MAX_NODES = 30
    dataset, tokenizer = generate_sympy_data(1000, max_nodes=MAX_NODES)
    
    print("Logging mathematically sound ground-truth generation to knowledge database...")
    for item in dataset:
        p_math = str(item["problem_expr"])
        s_math = str(item["correct_expr"])
        p_nl = sympy_to_nl_str(item["problem_expr"])
        s_nl = sympy_to_nl_str(item["correct_expr"])
        # Log to db, energy initially 0.0 or lowest theoretical bound
        log_to_db(db_conn, p_nl, p_math, s_nl, s_math, 0.0, True)
    
    # Target 6GB VRAM constraint Memory Optimizations
    BATCH_SIZE = 16 
    GRAD_ACCUM_STEPS = 4 
    EPOCHS = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 256 dim fits comfortably within 6GB threshold
    model = MathEBM(vocab_size=1000, d_model=256, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Resize embedding layer to fit the dynamic token vocabulary exactly
    model.embedding = nn.Embedding(tokenizer.vocab_size + 100, 256).to(device)
    
    print(f"Training on device: {device}")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        random.shuffle(dataset)
        
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            
            x_nodes = torch.stack([item["problem_nodes"] for item in batch]).to(device)
            x_adj = torch.stack([item["problem_adj"] for item in batch]).to(device)
            
            y_pos_discrete = torch.stack([item["correct_nodes"] for item in batch]).to(device)
            y_pos_adj = torch.stack([item["correct_adj"] for item in batch]).to(device)
            
            y_pos_soft = F.one_hot(y_pos_discrete, num_classes=model.embedding.num_embeddings).float()
            
            y_neg_init_discrete = torch.stack([item["adversarial_nodes"] for item in batch]).to(device)
            y_neg_adj = torch.stack([item["adversarial_adj"] for item in batch]).to(device)
            
            y_neg_init = F.one_hot(y_neg_init_discrete, num_classes=model.embedding.num_embeddings).float()
            y_neg_init = y_neg_init * 5.0 + torch.randn_like(y_neg_init) 
            
            y_neg_logits = sample_langevin(model, x_nodes, x_adj, y_neg_init, y_neg_adj, steps=15, step_size=0.1, temp=1.0)
            y_neg_soft = F.gumbel_softmax(y_neg_logits, tau=1.0, hard=False)
            
            model.train() 
            
            pos_energy = model(x_nodes, x_adj, y_pos_soft, y_pos_adj)
            neg_energy = model(x_nodes, x_adj, y_neg_soft, y_neg_adj)
            
            loss = (pos_energy - neg_energy).mean() + 0.1 * (pos_energy**2 + neg_energy**2).mean()
            
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if (i // BATCH_SIZE + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        avg_loss = total_loss / (len(dataset)//BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS} | CD Loss: {avg_loss:.4f}")
        
    print(f"Saving model checkpoint to {save_path}...")
    save_checkpoint(model, tokenizer, save_path)
    print("Training complete.")

def evaluate_energy(model, tokenizer, problem_str, solution_str, device):
    """Parses arbitrary strings into ASTs and returns the model's energy assigned to the pair."""
    try:
        problem_expr = sp.sympify(problem_str)
        solution_expr = sp.sympify(solution_str)
        
        p_nodes, p_adj = tokenizer.encode_graph(problem_expr, max_nodes=30)
        s_nodes, s_adj = tokenizer.encode_graph(solution_expr, max_nodes=30)
        
        p_nodes = p_nodes.unsqueeze(0).to(device)
        p_adj = p_adj.unsqueeze(0).to(device)
        s_nodes = s_nodes.unsqueeze(0).to(device)
        s_adj = s_adj.unsqueeze(0).to(device)
        
        # Continuous bridge formatting
        s_soft = F.one_hot(s_nodes, num_classes=model.embedding.num_embeddings).float()
        
        with torch.no_grad():
            energy = model(p_nodes, p_adj, s_soft, s_adj)
        return energy.item()
    except Exception as e:
        return f"Error parsing equations: {str(e)}"

def interactive_interface():
    print("=== Mathematical EBM Interface ===")
    model_path = "math_ebm.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    db_conn = init_db()
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint '{model_path}' not found. Starting training loop first...")
        train_ebm(save_path=model_path, db_conn=db_conn)
        
    print("Loading model weights...")
    model, tokenizer = load_checkpoint(model_path, device)
    print("Model ready!")
    print("Instructions:")
    print("- Enter a 'Problem' (e.g., 'x squared plus 5x plus 6' or 'x**2 + 5*x + 6')")
    print("- Enter a 'Proposed Solution' (e.g., '(x + 2) times (x + 3)')")
    print("- Leave blank to exit.\n")
    
    while True:
        try:
            p_text = input("Problem (Natural Language or Math): ").strip()
            if not p_text: break
            s_text = input("Proposed Solution (Natural Language or Math): ").strip()
            if not s_text: break
            
            p_math = nl_to_sympy_str(p_text)
            s_math = nl_to_sympy_str(s_text)
            
            # Reconstruct proper NL representation incase user typed raw math
            try:
                p_nl = sympy_to_nl_str(sp.sympify(p_math))
                s_nl = sympy_to_nl_str(sp.sympify(s_math))
                
                # Check for Mathematical correctness purely through SymPy 
                is_sound = (sp.simplify(sp.sympify(p_math) - sp.sympify(s_math)) == 0)
            except Exception as e:
                p_nl = p_text
                s_nl = s_text
                is_sound = False
            
            print(f"\nNatural Language (Problem): {p_nl}")
            print(f"Parsed Math (Problem):      {p_math}")
            print(f"Natural Language (Sol):     {s_nl}")
            print(f"Parsed Math (Sol):          {s_math}")
            
            energy = evaluate_energy(model, tokenizer, p_math, s_math, device)
            
            if isinstance(energy, str):
                print(f"[!] {energy}\n")
            else:
                print(f"==> EBM Predicted Energy: {energy:.4f}")
                print(f"    (Mathematically sound logically: {str(is_sound).upper()})")
                print("    (Saved to knowledge database)\n")
                
                log_to_db(db_conn, p_nl, p_math, s_nl, s_math, energy, is_sound)
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_ebm()
    else:
        interactive_interface()
