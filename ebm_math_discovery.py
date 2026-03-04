import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import random
from torch.utils.checkpoint import checkpoint

# Task 1: Energy Network PyTorch class

class MathEBM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024):
        """
        Energy-Based Model for mathematical symbolic discovery.
        Designed to run on 6GB VRAM by using a small dimension Transformer 
        and gradient checkpointing.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Linear embedding layer for both discrete tokens and continuous bridge
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Small Transformer Encoders (128-256 dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Scalar Energy Head (E_theta)
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y_soft):
        """
        Calculates the Energy of a given Problem (x) and Proposed Solution (y_soft).
        x: [B, seq_len_x] - Problem AST tokens (discrete)
        y_soft: [B, seq_len_y, vocab_size] - Solution AST soft-tokens (bridge continuous)
        """
        B, seq_len_x = x.shape
        _, seq_len_y, _ = y_soft.shape
        
        # Embed discrete problem tokens
        x_emb = self.embedding(x) # [B, seq_len_x, d_model]
        
        # Embed continuous solution soft-tokens (Discrete-to-Continuous Bridge)
        # We multiply soft-probabilities directly with the vocabulary embeddings
        y_emb = torch.matmul(y_soft, self.embedding.weight) # [B, seq_len_y, d_model]
        
        # Concatenate x and y representations
        tokens = torch.cat([x_emb, y_emb], dim=1) # [B, seq_len_x + seq_len_y, d_model]
        
        # Add positional encodings
        tokens = tokens + self.pos_encoder[:, :tokens.size(1), :]
        
        # Memory Optimization: Checkpoint transformer processing
        out = tokens
        if self.training and out.requires_grad:
            # Iteratively checkpoint each layer to drastically reduce peak VRAM
            for layer in self.transformer.layers:
                out = checkpoint(layer, out, use_reentrant=False)
        else:
            out = self.transformer(out)
            
        # Global pooling to derive a single vector for the entire expression pair
        pooled = out.mean(dim=1)
        
        # Output scalar energy per sample
        energy = self.energy_head(pooled).squeeze(-1)
        return energy


# Task 2: MCMC Langevin Sampling Loop

def sample_langevin(model, x, y_logits_init, steps=20, step_size=0.1, temp=1.0):
    """
    Performs gradient-based Langevin Dynamics to optimize the continuous
    representation (y_logits) by minimizing the predicted energy E(x, y).

    x: [B, len_x] discrete problem tokens
    y_logits_init: [B, len_y, vocab_size] initial logits for the proposed solution
    steps: number of Langevin updates
    step_size: step size for gradient descent
    temp: temperature for Gumbel-Softmax
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
        energy = model(x, y_soft)
        
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

class ASTTokenizer:
    """Simple Tokenizer to flatten SymPy AST into prefix array expressions"""
    def __init__(self):
        self.vocab = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
        self.inv_vocab = {0: "PAD", 1: "UNK", 2: "BOS", 3: "EOS"}
        self.vocab_size = 4

    def add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.vocab_size
            self.inv_vocab[self.vocab_size] = token
            self.vocab_size += 1

    def tokenize_ast(self, expr):
        """Recursively turn SymPy AST into a prefix sequential list."""
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        if isinstance(expr, sp.Integer) or isinstance(expr, sp.Rational):
            return [str(expr)]
        # Add, Mul, Pow, etc.
        op = expr.__class__.__name__
        tokens = [op]
        for arg in expr.args:
            tokens.extend(self.tokenize_ast(arg))
        return tokens

    def encode(self, tokens, max_len=None):
        for t in tokens:
            self.add_token(t)
        enc = [self.vocab.get(t, self.vocab["UNK"]) for t in tokens]
        if max_len:
            enc = enc[:max_len]
            enc += [self.vocab["PAD"]] * max(0, max_len - len(enc))
        return torch.tensor(enc, dtype=torch.long)

def generate_sympy_data(num_samples=100):
    """
    Generate dataset of correct pairs and adversarial mutations using SymPy.
    """
    x = sp.Symbol('x')
    tokenizer = ASTTokenizer()
    dataset = []
    
    for _ in range(num_samples):
        # Generate random identity: factored form <-> expanded polynomial
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        factored = (x + a) * (x + b)
        expanded = sp.expand(factored)
        
        problem_tokens = tokenizer.tokenize_ast(expanded)
        solution_tokens = tokenizer.tokenize_ast(factored)
        
        # Create an 'Adversarial' mutation (e.g., incorrect factor)
        mutation_type = random.choice([1, 2, 3])
        if mutation_type == 1:
            adversarial = (x - a) * (x + b) # Wrong sign
        elif mutation_type == 2:
            adversarial = (x + a + 1) * (x + b) # Wrong constant
        else:
            adversarial = (x + b) * (x + b) # Duplicate term

        adversarial_tokens = tokenizer.tokenize_ast(adversarial)
        
        dataset.append({
            "problem": problem_tokens,
            "correct_solution": solution_tokens,
            "adversarial_solution": adversarial_tokens
        })
    
    return dataset, tokenizer

def train_ebm():
    print("Generating SymPy dataset of correct identities and adversarials...")
    dataset, tokenizer = generate_sympy_data(1000)
    
    # Target 6GB VRAM constraint Memory Optimizations
    BATCH_SIZE = 16 
    GRAD_ACCUM_STEPS = 4 
    MAX_SEQ_LEN = 30
    EPOCHS = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 256 dim fits comfortably within 6GB threshold
    model = MathEBM(vocab_size=1000, d_model=256, nhead=4, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Update embedding space size to fit observed dynamic token vocabulary
    model.embedding = nn.Embedding(tokenizer.vocab_size + 100, 256).to(device)
    
    print(f"Training on device: {device}")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        random.shuffle(dataset)
        
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            
            # 1. Prepare Problems and Ground-Truth Solutions
            x_batch = torch.stack([tokenizer.encode(item["problem"], MAX_SEQ_LEN) for item in batch]).to(device)
            y_pos_discrete = torch.stack([tokenizer.encode(item["correct_solution"], MAX_SEQ_LEN) for item in batch]).to(device)
            
            # Make the positive solutions "soft" continuous tokens for bridge compatibility
            y_pos_soft = F.one_hot(y_pos_discrete, num_classes=model.embedding.num_embeddings).float()
            
            # 2. Prepare Adversarial starting points and apply Langevin MCMC Dynamics
            y_neg_init_discrete = torch.stack([tokenizer.encode(item["adversarial_solution"], MAX_SEQ_LEN) for item in batch]).to(device)
            y_neg_init = F.one_hot(y_neg_init_discrete, num_classes=model.embedding.num_embeddings).float()
            
            # Slightly scale logits to give model room to sample 
            y_neg_init = y_neg_init * 5.0 + torch.randn_like(y_neg_init) 
            
            # Generate Negative pairs internally
            y_neg_logits = sample_langevin(model, x_batch, y_neg_init, steps=15, step_size=0.1, temp=1.0)
            y_neg_soft = F.gumbel_softmax(y_neg_logits, tau=1.0, hard=False)
            
            model.train() # Resume training after MCMC freeze
            
            # 3. Compute Energies
            pos_energy = model(x_batch, y_pos_soft)
            neg_energy = model(x_batch, y_neg_soft)
            
            # 4. Contrastive Divergence Loss: Minimize Pos energy, Maximize Neg energy.
            # Adds L2 Regularization term on the energy to keep scalar values stable.
            loss = (pos_energy - neg_energy).mean() + 0.1 * (pos_energy**2 + neg_energy**2).mean()
            
            # Gradient Accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if (i // BATCH_SIZE + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        avg_loss = total_loss / (len(dataset)//BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS} | CD Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_ebm()
