import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
EOT_BYTES = b"<|endoftext|>"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def trim_to_complete_stories(raw: bytes) -> bytes:
    """Trim byte buffer to the end of the last complete story."""
    last_eot = raw.rfind(EOT_BYTES)
    return raw if last_eot == -1 else raw[: last_eot + len(EOT_BYTES)]


def stream_prefix(url: str, max_bytes: int, chunk_size: int = 64 * 1024) -> bytes:
    """Stream up to `max_bytes` from `url`."""
    buf = bytearray()
    with urllib.request.urlopen(url) as resp, tqdm(
        total=max_bytes, unit="B", unit_scale=True, desc="Downloading TinyStories"
    ) as pbar:
        while len(buf) < max_bytes:
            chunk = resp.read(min(chunk_size, max_bytes - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
            pbar.update(len(chunk))
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Download a TinyStories subset.")
    parser.add_argument("--mb", type=float, default=50.0, help="Size in MB to download (default: 50)")
    args = parser.parse_args()
    if args.mb <= 0:
        raise ValueError("--mb must be > 0.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = stream_prefix(URL, int(args.mb * 1024 * 1024))
    text = trim_to_complete_stories(raw).decode("utf-8", errors="ignore")

    out_path = DATA_DIR / "tinystories.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
