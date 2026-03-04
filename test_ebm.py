import torch
from ebm_math_discovery import LLMSeqTokenizer, MathEBM, sample_langevin, generate_sympy_data
import sympy as sp

def test():
    print("Testing LLM Tokenizer...")
    tokenizer = LLMSeqTokenizer()
    expr = sp.sympify("x**2 + 5*x + 6")
    nodes, adj = tokenizer.encode_graph(expr, max_nodes=50)
    print("Tokenizer output shape:", nodes.shape, adj.shape)

    print("Testing EBM Forward Pass...")
    model = MathEBM(vocab_size=tokenizer.vocab_size, d_model=256, num_layers=4)
    model.embedding = torch.nn.Embedding(tokenizer.vocab_size + 100, 256)
    
    # Fake batch
    batch_size = 2
    x_nodes = nodes.unsqueeze(0).repeat(batch_size, 1)
    x_adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
    
    y_soft = torch.nn.functional.one_hot(x_nodes, num_classes=model.embedding.num_embeddings).float()
    y_adj = x_adj.clone()

    # The model uses gradient checkpointing which requires requires_grad=True for inputs usually, 
    # but we are just running forward
    model.train()
    x_nodes_emb = model.embedding(x_nodes).requires_grad_(True)
    energy = model(x_nodes, x_adj, y_soft, y_adj)
    print("Energy output:", energy)

    print("Testing Langevin MCMC...")
    y_neg_init = y_soft.clone() * 5.0 + torch.randn_like(y_soft)
    y_neg_logits = sample_langevin(model, x_nodes, x_adj, y_neg_init, y_adj, steps=2, step_size=0.1)
    print("Langevin output shape:", y_neg_logits.shape)

    print("Testing generate_sympy_data...")
    ds, _, _ = generate_sympy_data(2, max_nodes=50, mode="arithmetic", tokenizer=tokenizer)
    print("Data gen successful, items:", len(ds))
    
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test()
