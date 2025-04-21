import torch
import torch.nn as nn
from Data.data import load_data_loaders, VOCAB_SIZE, OUTPUT_DIM, train_data, valid_data, test_data
from config import *


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * (self.embedding_dim ** 0.5)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]

if __name__ == "__main__":
    
    # train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer = load_data_loaders()
    pass