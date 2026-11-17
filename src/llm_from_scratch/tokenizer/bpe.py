import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import regex as re
from tqdm import tqdm

PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")


def _merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
    """Replace all occurrences of pair with new_token in a new list."""
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


class BPETokenizer:
    """Byte-level BPE tokenizer with regex pre-tokenization and special token support."""

    def __init__(self, pattern: re.Pattern = PATTERN) -> None:
        self.pattern = pattern
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}
        self.vocab: dict[int, bytes] = {}
        self._trained = False

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            vocab[new_id] = vocab[p0] + vocab[p1]
        for tok, idx in self.special_tokens.items():
            vocab[idx] = tok.encode("utf-8")
        return vocab

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Tokenizer is not trained. Call train() or from_file() first.")

    @property
    def vocab_size(self) -> int:
        """Total vocab size: 256 base bytes + learned merges + special tokens."""
        return len(self.vocab)

    def token_to_id(self, token: str) -> int | None:
        """Look up ID for a special token or UTF-8 token string; None if absent."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        target = token.encode("utf-8")
        return next((idx for idx, b in self.vocab.items() if b == target), None)

    def id_to_token(self, idx: int) -> bytes | None:
        """Look up raw byte sequence for a token ID; None if out of range."""
        return self.vocab.get(idx)

    def train(
        self,
        file: str | Path,
        vocab_size: int = 4096,
        special_tokens: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        """Train BPE merges from a UTF-8 text file."""
        special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be at least {256 + len(special_tokens)}."
            )

        text = Path(file).read_text("utf-8")
        word_counts = Counter()

        if special_tokens:
            special_pattern = "|".join(re.escape(k) for k in special_tokens)
            docs = re.split(special_pattern, text)
        else:
            docs = [text]

        for doc in docs:
            if doc:
                word_counts.update(self.pattern.findall(doc))

        words = [[list(w.encode("utf-8")), freq] for w, freq in word_counts.items()]
        original_length = sum(len(t) * freq for t, freq in words)
        merges: dict[tuple[int, int], int] = {}
        pbar = tqdm(range(num_merges), desc="BPE Training", unit="merge", disable=not verbose)

        for i in pbar:
            # 1. Count adjacent pairs weighted by word frequency
            pair_counts: Counter[tuple[int, int]] = Counter()
            for tokens, freq in words:
                for pair in pairwise(tokens):
                    pair_counts[pair] += freq
            if not pair_counts:
                break

            # 2. Pick top pair & record merge
            pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
            new_token = 256 + i
            merges[pair] = new_token

            # 3. Merge inside each unique word
            for item in words:
                if len(item[0]) >= 2 and pair[0] in item[0]:
                    item[0] = _merge(item[0], pair, new_token)

            if i % 100 == 0:
                total_tokens = sum(len(t) * freq for t, freq in words)
                pbar.set_postfix(ratio=f"{original_length / total_tokens:.2f}x")

        base = 256 + len(merges)
        self.merges = merges
        self.special_tokens = {tok: base + j for j, tok in enumerate(special_tokens)}
        self.vocab = self._build_vocab()
        self._trained = True

    def _encode_chunk(self, chunk: str) -> list[int]:
        tokens = list(chunk.encode("utf-8"))
        while len(tokens) >= 2:
            # Apply the earliest-learned merge first (lowest ID in merges)
            pair = min(pairwise(tokens), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = _merge(tokens, pair, self.merges[pair])
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode a single string into token IDs (recognizes registered special tokens)."""
        self._require_trained()
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        cache: dict[str, list[int]] = {}
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                for chunk in self.pattern.findall(part):
                    if chunk not in cache:
                        cache[chunk] = self._encode_chunk(chunk)
                    ids.extend(cache[chunk])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into a UTF-8 string."""
        self._require_trained()
        if skip_special_tokens and self.special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [i for i in ids if i not in special_ids]
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save pattern, merges, and special_tokens to a single tokenizer.json."""
        self._require_trained()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pattern": self.pattern.pattern,
            "merges": [[p0, p1, new_id] for (p0, p1), new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        """Load a trained BPETokenizer from tokenizer.json."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(pattern=re.compile(data["pattern"]))
        tokenizer.merges = {(int(p0), int(p1)): int(new_id) for p0, p1, new_id in data["merges"]}
        tokenizer.special_tokens = {
            tok: int(idx) for tok, idx in data.get("special_tokens", {}).items()
        }
        tokenizer.vocab = tokenizer._build_vocab()
        tokenizer._trained = True
        return tokenizer
