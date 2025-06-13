import pytest
import json
import sys
import os

# Add the parent directory to sys.path to resolve "data" as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.script import cache_or_process_bpe

# Your usage will now work correctly:
tokenizer_path = fr"D:\Programming\Python\Transformer-from-Scratch\bpe_models"
input_path = fr"D:\Programming\Python\Transformer-from-Scratch\bpe_data"

train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer = cache_or_process_bpe(
    tokenizer_path=tokenizer_path,
    output_path=input_path,
    batch_size=32,
    force_download=False
)
# @pytest.mark.parametrize("name,tokenizer", [
#     ("en", en_tokenizer),
#     ("vi", vi_tokenizer)
# ])
# def test_unknown_token_used(name, tokenizer):
#     test_input = "xyztokennotinidata"  # unlikely to be in vocab
#
#     # Save vocab to JSON
#     vocab = tokenizer.get_vocab()
#     with open(f"{name}_vocab.json", "w", encoding="utf-8") as f:
#         json.dump(vocab, f, indent=4, ensure_ascii=False)
#
#     # Encode the test input
#     encoded = tokenizer.encode(test_input)
#
#     # Get unknown token ID
#     unk_token_id = tokenizer.token_to_id("<unk>")
#
#     # Debug print
#     print(f"[{name}] Encoded IDs:", encoded.ids)
#     print(f"[{name}] Decoded:", [tokenizer.id_to_token(i) for i in encoded.ids])
#
#     # Assert that <unk> token ID is present
#     assert unk_token_id in encoded.ids, f"[{name}] Unknown token not used for: {test_input}"
