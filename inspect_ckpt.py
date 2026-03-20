import torch
import os
import sys

path = "math_ebm.pt"
print(f"Loading: {path}  size={os.path.getsize(path)//1024}KB", flush=True)
ckpt = torch.load(path, map_location="cpu", weights_only=False)

is_wrapped = "model_state_dict" in ckpt or "llm_vocab_size" in ckpt
print(f"Is new wrapped format: {is_wrapped}", flush=True)

print("\n=== CHECKPOINT KEYS ===", flush=True)
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    if hasattr(v, "shape"):
        info = f"Tensor{tuple(v.shape)}"
    elif isinstance(v, dict):
        info = f"dict({len(v)} entries)"
    else:
        info = repr(v)[:80]
    print(f"  {k}: {info}", flush=True)

print("\n=== DETECTED ARCHITECTURE ===", flush=True)
for k, v in ckpt.items():
    if "embed" in k.lower() and hasattr(v, "shape"):
        print(f"  {k}: vocab_size={v.shape[0]}, d_model={v.shape[1]}", flush=True)

sys.stdout.flush()
