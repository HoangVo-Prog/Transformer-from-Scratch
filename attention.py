import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import Clone


class ScaledDotProductAttention(nn.Module):
    """
        Attention(Q, K, V) = softmax(Q@K^T / sqrt(d_k))@V
    """
    def __init__(self, d_model, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn
    
    
class MultiHeadAttention(nn.Module):
    """
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)@W^O
            where head_i = Attention(Q@W_i^Q, K@W_i^K, V@W_i^V)
                W_i^Q, W_i^K, W_i^V are parameter matrices for each head
    """
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(d_model, dropout)
        
        self.linear_layers = Clone(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.p_attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
 
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]
        
        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        # Perform attention on all the projected vectors in batch
        x, self.p_attn = self.attention(query, key, value, mask=mask)
        
        # Concat heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # d_model = num_heads * d_k
        
        del query
        del key
        del value
        
        return self.linear_layers[-1](x)
    
    
    
    
