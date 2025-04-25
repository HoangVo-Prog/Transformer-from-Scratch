import pytest
import torch
from utils import subsequent_mask
from model import Transformer


def test_subsequent_mask():
    size = torch.randint(0, 10, (1,)).item()
    
    mask = subsequent_mask(size)
    for i in range(size):
        for j in range(size):
           assert mask[0, i, j] == (1 if j <= i else 0), f"Failed for size {size}: expected mask[{i}, {j}] to be {(1 if j <= i else 0)}"
         
           
@pytest.mark.parametrize(
    "N, d_model, d_ff, h, dropout, src_vocab, tgt_vocab, batch_size, src_len, tgt_len", 
    [
        (6, 512, 2048, 8, 0.1, 10000, 10000, 32, 20, 20),  # Example configuration
        (6, 256, 1024, 4, 0.2, 5000, 5000, 16, 30, 30),    # Another configuration
    ]
)
def test_transformer_output_shape(N, d_model, d_ff, h, dropout, src_vocab, tgt_vocab, batch_size, src_len, tgt_len):
    model = Transformer(N, d_model, d_ff, h, dropout)
    
    src = torch.randint(0, src_vocab, (batch_size, src_len))  # (batch_size, src_len)
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))  # (batch_size, tgt_len)
    
    src_mask = torch.zeros(batch_size, 1, src_len, src_len)  # Source mask, typically all ones for the source
    tgt_mask = subsequent_mask(tgt_len).unsqueeze(0).expand(batch_size, -1, tgt_len, tgt_len)  # Correct expansion for batch_size

    output = model(src_vocab=src_vocab, tgt_vocab=tgt_vocab)(src, tgt, src_mask, tgt_mask)

    assert output.shape == (batch_size, tgt_len, tgt_vocab)
    