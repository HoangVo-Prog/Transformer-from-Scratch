from config import *
import torch
import torch.nn as nn
from utils import clones


"""
    There are 3 different ways of using multi-head attention
        - "Encoder-Decoder attention" layers: 
            Q: decoder layer
            K, V: encoder outputs
            
        - "Self-attention" in Encoder:
            Q, K, V: output of the previous layer
            Each position in the encoder can attend to all positions in the previous layer of the encoder.

        - "Self-attention" in Decoder: 
            Each position in the decoder to attend to all positions in the decoder up to and including that position.
            Prevent leftward information flow in the decoder to preserve the auto-represive property.
            Implement inside of scaled dot-product attention by masking out (setting to -inf in solfmax)

"""


class ScaledDotProductAttention(nn.Module):
    """
        Attention(Q, K, V) = softmax(QK^T/sqrt(dk))V
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1) / (d_k ** 0.5))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        p_attn = scores.softmax(dim=-1)
        if dropout:
            p_attn = dropout(p_attn)
            
        return torch.matmul(p_attn, value), p_attn
            
    
class MultiHeadAttention(nn.Module):
    """
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
            where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
            
        h = 8 parallel attention layers, or heads
        For each, using d_k (EMBEDDING_DIM) = d_v = d_model/h = 64 
    """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadAttention, self).__init__()
        assert d_model%h==0
        
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        ""
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        Attention = ScaledDotProductAttention()
        x, self.attn = Attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        
        # 3) "Concat" using a view and apply a final linear
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        
        del query, key, value
        return self.linears[-1](x)
       