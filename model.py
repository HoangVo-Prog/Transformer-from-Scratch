import torch
import torch.nn as nn
from Data.data import load_data_loaders



class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * (self.embedding_dim ** 0.5)
    # Save the data


# train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer = load_data_loaders()