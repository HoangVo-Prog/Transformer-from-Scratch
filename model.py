import torch
import torch.nn as nn
from data import load_data



class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * (self.embedding_dim ** 0.5)
    # Save the data

print("Loading data...")
train_data, valid_data, test_data = load_data()
print("Data loaded.")



