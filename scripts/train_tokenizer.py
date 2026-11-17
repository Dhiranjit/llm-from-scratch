import argparse
from pathlib import Path

from llm_from_scratch.tokenizer.bpe import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output or Path("data") / args.input.stem / "tokenizer.json"

    tokenizer = BPETokenizer()
    tokenizer.train(
        file=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        verbose=not args.quiet,
    )
    tokenizer.save(output_path)
    print(f"Saved tokenizer (vocab_size={tokenizer.vocab_size}) to {output_path}")


if __name__ == "__main__":
    main()