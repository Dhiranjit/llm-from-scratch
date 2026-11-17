### Train the tokenizer

uv run python scripts/train_tokenizer.py \
 --input data/tinystories.txt \
 --vocab-size 4096 \
 --special-tokens '<|endoftext|>'

### Encode the data

uv run python scripts/encode.py \
 --input data/tinystories.txt \
 --val-ratio 0.05
