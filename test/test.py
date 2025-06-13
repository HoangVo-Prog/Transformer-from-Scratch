import pytest
import sys
import os

# Add the parent directory to sys.path to resolve "data" as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.script import cache_or_process_bpe, cache_or_process_pretrained

_, _, _, vinai_tokenizer = cache_or_process_pretrained(
    tokenizer_type="vinai",
    tokenizer_path=fr"D:\Programming\Python\Transformer-from-Scratch\vinai_models",
    output_path=fr"D:\Programming\Python\Transformer-from-Scratch\vinai_data",
    batch_size=32,
)

_, _, _, mbart_tokenizer = cache_or_process_pretrained(
    tokenizer_type="mbart",
    tokenizer_path=fr"D:\Programming\Python\Transformer-from-Scratch\mbart_models",
    output_path=fr"D:\Programming\Python\Transformer-from-Scratch\mbart_data",
    batch_size=32,
)

_, _, _, en_tokenizer, vi_tokenizer = cache_or_process_bpe(
    tokenizer_path=fr"D:\Programming\Python\Transformer-from-Scratch\bpe_models",
    output_path=fr"D:\Programming\Python\Transformer-from-Scratch\bpe_data",
    batch_size=32,
)

def get_token_id(tokenizer, token: str) -> int:
    """
    Get the token ID for a given token string using a HuggingFace tokenizer (e.g., VinAI PhoBERT).
    This supports only pre-tokenized tokens.

    Args:
        tokenizer: A HuggingFace tokenizer like AutoTokenizer.from_pretrained("vinai/phobert-base")
        token (str): The string token to look up

    Returns:
        int: Token ID, or tokenizer.unk_token_id if not found
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id if token_id is not None else tokenizer.unk_token_id



@pytest.mark.parametrize("name,tokenizer", [
    ("vinai", vinai_tokenizer),
    ("mbart", mbart_tokenizer),
])
def test_unknown_token_used(name, tokenizer):
    test_input = "xyztokennotinidata"  # unlikely to be in vocab
    # Get unknown token ID

    id = get_token_id(tokenizer, test_input)
    unk_token_id = get_token_id(tokenizer, "<unk>")
    print(f"[{name}] Encoded IDs:", id)
    print(f"[{name}] Decoded:", unk_token_id)

    # Assert that <unk> token ID is present
    assert (unk_token_id in id) if isinstance(id, list) else (unk_token_id == id), f"[{name}] Unknown token not used for: {test_input}"


@pytest.mark.parametrize("name,tokenizer", [
    ("en", en_tokenizer),
    ("vi", vi_tokenizer),
])
def test_bpe_unknown_token_not_used(name, tokenizer):
    test_input = "xyztokennotinidata"  # unlikely to be in vocab
    # Get unknown token ID
    unk_token_id = tokenizer.token_to_id("<unk>")
    encoded = tokenizer.encode(test_input)
    id = encoded.ids
    print(f"[{name}] Encoded IDs:", id)
    print(f"[{name}] Decoded:", [tokenizer.id_to_token(i) for i in id])

    # Assert that <unk> token ID is present
    assert (unk_token_id not in id) if isinstance(id, list) else (unk_token_id != id), f"[{name}] Unknown token not used for: {test_input}"
