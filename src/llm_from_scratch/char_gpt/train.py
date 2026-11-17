import sys
import torch
from pathlib import Path
from llm_from_scratch.char_gpt.model import CausalSelfAttentionModel




# Hyperparameters
batch_size = 32
block_size = 256
n_embd = 384
n_head = 6
n_layers = 6
dropout = 0.2
lr = 3e-4

max_steps = 6000
eval_interval = 500


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data" / "tiny_shakespeare.txt"
CKPT_DIR = ROOT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "char_gpt.pt"


text = DATA_PATH.read_text("utf-8")
print(f"Text loaded from {DATA_PATH}")

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)


# Loading data
data = torch.tensor(encode(text), dtype=torch.long, device=device)

# Splitting into train and val set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size, block_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ), device=device)
    offsets = torch.arange(block_size, device=device)
    xb = data[ix[:, None] + offsets]
    yb = data[ix[:, None] + offsets + 1]

    return xb, yb


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
            "stoi": stoi,
            "itos": itos,
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