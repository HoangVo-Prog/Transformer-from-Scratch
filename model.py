from config import *
import copy
import math
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import MultiHeadAttention


class PositionalWiseFeedForward(nn.Module):
    """
    Implements FFN equation.
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)
    

class PositionalEncoding(nn.Module):
    
    """
    Implement the old positional embedding function ~ very good with small dataset
        PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute this positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
    
    
class Transformer(nn.Module):
    def __init__(self, N, d_model, d_ff, h, dropout):
        super(Transformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.c = copy.deepcopy
        self.attn = MultiHeadAttention(h, d_model)
        self.ff = PositionalWiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout)
                    
    def forward(self, src_vocab, tgt_vocab):
        model = EncoderDecoder(
        Encoder(EncoderLayer(self.d_model, self.c(self.attn), self.c(self.ff), self.dropout), self.N),
        Decoder(DecoderLayer(self.d_model, self.c(self.attn), self.c(self.attn), self.c(self.ff), self.dropout), self.N),
        nn.Sequential(Embedding(self.d_model, src_vocab), self.c(self.position)),
        nn.Sequential(Embedding(self.d_model, tgt_vocab), self.c(self.position)),
        Generator(self.d_model, tgt_vocab),
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model
    