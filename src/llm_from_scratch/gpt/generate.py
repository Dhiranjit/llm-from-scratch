import argparse
import json
from pathlib import Path

import torch

from llm_from_scratch.gpt.model import CausalSelfAttentionModel

ROOT_DIR = Path(__file__).resolve().parents[3]
CKPT_PATH = ROOT_DIR / "checkpoints" / "gpt.pt"


def build_vocab(tokenizer_path: Path) -> dict[int, bytes]:
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for p0, p1, idx in data["merges"]:
        vocab[idx] = vocab[p0] + vocab[p1]
    for token, idx in data.get("special_tokens", {}).items():
        vocab[idx] = token.encode("utf-8")
    return vocab


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CKPT_PATH, weights_only=False, map_location=device)

    tokenizer_path = ROOT_DIR / checkpoint["tokenizer_path"]
    vocab = build_vocab(tokenizer_path)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    model = CausalSelfAttentionModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = model.generate(idx, args.tokens)
    print(decode(output[0].tolist(), vocab))


if __name__ == "__main__":
    main()
