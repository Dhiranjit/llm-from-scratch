import torch
import numpy as np
from pathlib import Path
from llm_from_scratch.gpt.model import CausalSelfAttentionModel


# Model
vocab_size = 4096
block_size = 256   
n_embd = 384
n_head = 6      
n_layers = 6
dropout = 0.1

# Optimization
batch_size = 32   
lr = 5e-4          
max_steps = 6000     
eval_interval = 500


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "tinystories"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CKPT_DIR = ROOT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "gpt.pt"


def get_batch(split, batch_size, block_size):
    data_file = DATA_DIR / ("train.bin" if split == "train" else "val.bin")
    data = np.memmap(data_file, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.inference_mode()
def estimate_loss(model, eval_iters=100):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, batch_size, block_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    # Initializing the model
    model = CausalSelfAttentionModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layers=n_layers,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(max_steps):

        if i % eval_interval == 0 or i == max_steps - 1:
            losses = estimate_loss(model, eval_iters=50)
            print(f" Step: {i:5d}/{max_steps} Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} ")

        xb, yb = get_batch("train", batch_size, block_size)
        logits, loss = model(xb, yb)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Final eval
    final_losses = estimate_loss(model, eval_iters=500)
    print(f"Final Train Loss: {final_losses['train']:.4f} Final Val Loss: {final_losses['val']:.4f}")

    # Model saving
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_path": str(TOKENIZER_PATH.relative_to(ROOT_DIR)),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layers": n_layers,
                "block_size": block_size,
                "dropout": dropout,
            },
        },
        CKPT_PATH,
    )
    print(f"Saved trained model to {CKPT_PATH}")


if __name__ == "__main__":
    main()