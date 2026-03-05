import torch
from ebm_math_discovery import LLMSeqTokenizer, ASTGraphTokenizer, MathEBM, sample_langevin, generate_sympy_data
import sympy as sp

def test():
    print("Testing Tokenizers...")
    llm_tokenizer = LLMSeqTokenizer()
    ast_tokenizer = ASTGraphTokenizer()
    
    p_nl = "x squared plus five times x plus six"
    s_expr = sp.sympify("x**2 + 5*x + 6")
    
    x_nodes, x_adj = llm_tokenizer.encode_graph(p_nl, max_nodes=50)
    y_nodes, y_adj = ast_tokenizer.encode_graph(s_expr, max_nodes=50)
    print("LLM Sequence shape:", x_nodes.shape, x_adj.shape)
    print("AST Graph shape:", y_nodes.shape, y_adj.shape)

    print("Testing Dual-Encoder EBM Forward Pass & Cross-Attention...")
    model = MathEBM(llm_vocab_size=llm_tokenizer.vocab_size, math_vocab_size=ast_tokenizer.vocab_size, d_model=256, num_layers=4)
    model.llm_embedding = torch.nn.Embedding(llm_tokenizer.vocab_size + 100, 256)
    model.math_embedding = torch.nn.Embedding(ast_tokenizer.vocab_size + 100, 256)
    
    # Fake batch
    batch_size = 2
    x_nodes_b = x_nodes.unsqueeze(0).repeat(batch_size, 1)
    x_adj_b = x_adj.unsqueeze(0).repeat(batch_size, 1, 1)
    
    y_nodes_b = y_nodes.unsqueeze(0).repeat(batch_size, 1)
    y_adj_b = y_adj.unsqueeze(0).repeat(batch_size, 1, 1)
    
    y_soft = torch.nn.functional.one_hot(y_nodes_b, num_classes=model.math_embedding.num_embeddings).float()

    model.train()
    energy = model(x_nodes_b, x_adj_b, y_soft, y_adj_b)
    print("Energy output:", energy)

    print("Testing Langevin MCMC (Optimizing Math Graph)...")
    y_neg_init = y_soft.clone() * 5.0 + torch.randn_like(y_soft)
    y_neg_logits = sample_langevin(model, x_nodes_b, x_adj_b, y_neg_init, y_adj_b, steps=2, step_size=0.1)
    print("Langevin output shape:", y_neg_logits.shape)

    print("Testing generate_sympy_data (Dual Outputs)...")
    ds, _, _, _ = generate_sympy_data(2, max_nodes=50, mode="arithmetic", llm_tokenizer=llm_tokenizer, ast_tokenizer=ast_tokenizer)
    print("Data gen successful, items:", len(ds))
    
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test()
