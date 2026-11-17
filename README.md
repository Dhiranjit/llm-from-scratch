# LLM from Scratch

Building a character level GPT language model from the ground up in PyTorch from bigram -> MLP -> Wavenet -> Attention, and recreated a BPE and trained the GPT model with vocab_size = 4096

## The Journey

1. **Bigram** — Started from simple counts and probabilities to build intuition that language modeling, at its core, is next-token prediction and sampling.
2. **N-gram & MLP** — Expanded from a single character to multi-token context windows, seeing firsthand how autoregressive modeling works.
3. **Weight Initialization, BatchNorm & Diagnostics** — Explored why poor initialization saturates activations and kills gradients, why BatchNorm stabilizes deeper layers, and how to inspect activation/gradient diagnostics to verify training health.
4. **WaveNet** — Moved from squashing the whole context in one step to a hierarchical architecture that progressively merges representations across layers.
5. **Causal Self-Attention & `char_gpt`** — Built causal self-attention with `tril` masking, optimized attention by fusing Q, K, and V into a single matrix projection (`3 * n_embd`), stacked pre-norm residual Transformer blocks, and trained a character-level GPT.
6. **Byte-Level BPE Tokenizer** — Built a BPE tokenizer from scratch (`vocab_size = 4096` + `<|endoftext|>`), added regex pre-tokenization to prevent merges across word/punctuation boundaries, and sped up training by counting unique chunks with `word_counts` instead of re-scanning the raw corpus on every merge.
7. **TinyStories GPT (`gpt`)** — Compressed ~50M characters down to ~13M tokens into `uint16` `train.bin` / `val.bin` shards, streamed random batches off disk with `np.memmap`, trained a 6-layer, 6-head, 384-dim GPT (`block_size = 256`), and wired up standalone generation decoding directly from `tokenizer.json`.

## Scripts

### 1. Train the tokenizer

```bash
uv run python scripts/train_tokenizer.py \
  --input data/tinystories.txt \
  --vocab-size 4096 \
  --special-tokens '<|endoftext|>'
```

### 2. Encode the data

```bash
uv run python scripts/encode.py \
  --input data/tinystories.txt \
  --val-ratio 0.05
```

### 3. Train the model

**Character-level GPT (`char_gpt`)**

```bash
uv run python src/llm_from_scratch/char_gpt/train.py
```

**BPE GPT (`gpt`)**

```bash
uv run python src/llm_from_scratch/gpt/train.py
```

### 4. Generate text

**Character-level GPT (`char_gpt`)**

```bash
uv run python src/llm_from_scratch/char_gpt/generate.py --tokens 500
```

**BPE GPT (`gpt`)**

```bash
uv run python src/llm_from_scratch/gpt/generate.py --tokens 500
```

## Credits

- **Andrej Karpathy** — for teaching the world how LLMs actually work under the hood and making building one from scratch feel possible.
