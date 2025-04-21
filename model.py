import copy

import torch
import torch.nn as nn

from utils import *
from attention import MultiHeadAttention
from config import *
from Data.data import load_data_loaders


def make_model(
    src_vocab, tgt_vocab, N=N, d_model=D_MODEL, d_ff=D_FF, h=N_HEAD, dropout=DROPOUT
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


_, _, _, en_tokenizer, vi_tokenizer = load_data_loaders()
VOCAB_SIZE = len(en_tokenizer.get_vocab())  
OUTPUT_DIM = len(vi_tokenizer.get_vocab())  


model = make_model(VOCAB_SIZE, OUTPUT_DIM)
print(model)
