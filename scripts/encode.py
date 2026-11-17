import argparse
from collections.abc import Iterator
from itertools import batched, islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llm_from_scratch.tokenizer.bpe import BPETokenizer

DOC_SEP = "<|endoftext|>"


def _iter_documents(
    path: Path,
    sep: str = DOC_SEP,
    chunk_size: int = 4 * 1024 * 1024,
) -> Iterator[str]:
    """Yield documents from a UTF-8 file"""
    buf = ""
    with path.open("r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            buf += chunk
            parts = buf.split(sep)
            buf = parts.pop()
            for part in parts:
                if doc := part.lstrip("\n"):
                    yield doc + sep

def _encode_to_bin(
    docs: Iterator[str],
    tokenizer: BPETokenizer,
    out_path: Path,
    batch_size: int = 1000,
    pbar: tqdm | None = None,
) -> int:
    """Encode documents in batches and write uint16 token IDs to out_path."""
    total_tokens = 0
    with out_path.open("wb") as f:
        for batch in batched(docs, batch_size):
            ids = tokenizer.encode("".join(batch))
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total_tokens += len(ids)
            if pbar is not None:
                pbar.update(len(batch))
    return total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05,)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {args.val_ratio}.")

    out_dir = args.output_dir or Path("data") / args.input.stem
    tokenizer_path = args.tokenizer or out_dir / "tokenizer.json"

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    tokenizer = BPETokenizer.from_file(tokenizer_path)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            f"vocab_size ({tokenizer.vocab_size}) exceeds uint16 max (65536)."
        )

    total_docs = sum(1 for _ in _iter_documents(args.input))
    split_idx = int(total_docs * (1.0 - args.val_ratio))
    
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    doc_iter = _iter_documents(args.input)
    with tqdm(total=total_docs, desc="Encoding", unit="doc", disable=args.quiet) as pbar:
        train_tokens = _encode_to_bin(islice(doc_iter, split_idx), tokenizer, train_path, pbar=pbar)
        val_tokens = _encode_to_bin(doc_iter, tokenizer, val_path, pbar=pbar)

    print(f"Saved {train_tokens:,} train tokens to {train_path}")
    print(f"Saved {val_tokens:,} val tokens to {val_path}")


if __name__ == "__main__":
    main()
