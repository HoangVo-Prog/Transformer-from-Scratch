import pandas as pd
import pickle
import torch
from datasets import load_dataset  
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LENGTH, BATCH_SIZE, sos_token, eos_token, pad_token, unk_token, special_tokens


# Load the dataset from Hugging Face
def save_data(train_data, valid_data, test_data, filename='dataset.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((train_data, valid_data, test_data), f)
    print("Dataset loaded and saved successfully.")


# Load the data
def load_data(filename='dataset.pkl'):
    with open(f"Data\{filename}", 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    return train_data, valid_data, test_data


train_data, valid_data, test_data = load_data()


def bytepair_tokenize(train_data):
    # Define tokenizers for English and Vietnamese
    en_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_trainer = BpeTrainer(special_tokens=special_tokens)
    en_tokenizer.train_from_iterator(train_data["en"], trainer=en_trainer)

    vi_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    vi_tokenizer.pre_tokenizer = Whitespace()
    vi_trainer = BpeTrainer(special_tokens=special_tokens)
    vi_tokenizer.train_from_iterator(train_data["vi"], trainer=vi_trainer)
    
    return en_tokenizer, vi_tokenizer


def tokenize(data, tokenizer, max_length=MAX_LENGTH):
    encoding = tokenizer.encode(data)
    ids = [tokenizer.token_to_id(sos_token)] + encoding.ids + [tokenizer.token_to_id(eos_token)]
    return ids


def tokenize_and_numericalize(data, src_tokenizer, trg_tokenizer, max_length=MAX_LENGTH):
    def pad_or_truncate(ids, pad_index):
        # Truncate if longer than max_length
        if len(ids) > max_length:
            return ids[:max_length]
        # Pad if shorter than max_length
        return ids + [pad_index] * (max_length - len(ids))
    
    src_ids = tokenize(data["en"], src_tokenizer)
    trg_ids = tokenize(data["vi"], trg_tokenizer)
    
    src_pad_index = src_tokenizer.token_to_id(pad_token)
    trg_pad_index = trg_tokenizer.token_to_id(pad_token)
    
    return {
        "src_ids": pad_or_truncate(src_ids, src_pad_index),
        "trg_ids": pad_or_truncate(trg_ids, trg_pad_index),
    }


def get_data_loader(dataset, batch_size, shuffle=False):
    # No need for collate_fn
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def data_loader():     
    en_tokenizer, vi_tokenizer = bytepair_tokenize(train_data)
    
    train_data = train_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))
    valid_data = valid_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))
    test_data = test_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))

    train_data_loader = get_data_loader(train_data, BATCH_SIZE, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, BATCH_SIZE)
    test_data_loader  = get_data_loader(test_data, BATCH_SIZE)

    return train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer


def save_data_loaders(train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer):
    data_dict = {
        'train_loader': train_loader,
        'valid_loader': valid_loader, 
        'test_loader': test_loader,
        'en_tokenizer': en_tokenizer,
        'vi_tokenizer': vi_tokenizer
    }

    with open('Data/data_loader.pkl', 'wb') as file:
        pickle.dump(data_dict, file)
    print("Data loaders and tokenizers saved successfully.")


def load_data_loaders():
    with open(f'Data/data_loader.pkl', 'rb') as f:
        data_dict = pickle.load(f)
        train_loader = data_dict['train_loader']
        valid_loader = data_dict['valid_loader']
        test_loader = data_dict['test_loader']
        en_tokenizer = data_dict['en_tokenizer']
        vi_tokenizer = data_dict['vi_tokenizer']
    
    return train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer


if __name__ == "__main__":
    # # Load the dataset from Hugging Face
    # print("Loading dataset...")
    # ds = load_dataset("thainq107/iwslt2015-en-vi")
    
    # train_data, valid_data, test_data = ds["train"], ds["validation"], ds["test"]
    # save_data(train_data, valid_data, test_data)
    
    
    # print("Saving dataloaders & tokenizers...")
    # train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer = data_loader()
    # save_data_loaders(train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer)
    pass