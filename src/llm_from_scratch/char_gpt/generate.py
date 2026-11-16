import argparse
from pathlib import Path

import torch

from llm_from_scratch.char_gpt.model import CausalSelfAttentionModel

CKPT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "char_gpt.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CKPT_PATH, weights_only=False, map_location=device)

    itos = checkpoint["itos"]
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    model = CausalSelfAttentionModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = model.generate(idx, args.tokens)
    print("".join(itos[i] for i in output[0].tolist()))


if __name__ == "__main__":
    main()
