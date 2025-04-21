import pandas as pd
import pickle
import torch
from datasets import load_dataset  
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from config import MAX_LENGTH, BATCH_SIZE, sos_token, eos_token, pad_token, unk_token, special_tokens


# Load the dataset from Hugging Face
def save_data(train_data, valid_data, test_data, filename='dataset.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((train_data, valid_data, test_data), f)


# Load the data
def load_data(filename='dataset.pkl'):
    with open(filename, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    return train_data, valid_data, test_data


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
    return {"src_ids": tokenize(data["en"], src_tokenizer),
            "trg_ids": tokenize(data["vi"], trg_tokenizer)    
           }


def get_collate_fn(src_pad_index, trg_pad_index):
    def collate_fn(batch):
        batch_src_ids = [torch.tensor(example["src_ids"]) for example in batch]
        batch_trg_ids = [torch.tensor(example["trg_ids"]) for example in batch]
        
        # Manually pad the sequences to MAX_LENGTH (ensure truncation if necessary)
        for i in range(len(batch_src_ids)):
            # Truncate source sequences that are longer than MAX_LENGTH
            if len(batch_src_ids[i]) > MAX_LENGTH:
                batch_src_ids[i] = batch_src_ids[i][:MAX_LENGTH]
            # Pad source sequences that are shorter than MAX_LENGTH
            elif len(batch_src_ids[i]) < MAX_LENGTH:
                batch_src_ids[i] = torch.cat([batch_src_ids[i], torch.full((MAX_LENGTH - len(batch_src_ids[i]),), src_pad_index)])
            
            # Truncate target sequences that are longer than MAX_LENGTH
            if len(batch_trg_ids[i]) > MAX_LENGTH:
                batch_trg_ids[i] = batch_trg_ids[i][:MAX_LENGTH]
            # Pad target sequences that are shorter than MAX_LENGTH
            elif len(batch_trg_ids[i]) < MAX_LENGTH:
                batch_trg_ids[i] = torch.cat([batch_trg_ids[i], torch.full((MAX_LENGTH - len(batch_trg_ids[i]),), trg_pad_index)])

        # Stack all the sequences to create the batch
        batch_src_ids = torch.stack(batch_src_ids)
        batch_trg_ids = torch.stack(batch_trg_ids)

        return {"src_ids": batch_src_ids, "trg_ids": batch_trg_ids}
    
    return collate_fn


def get_data_loader(dataset, batch_size, src_pad_index, trg_pad_index, shuffle=False):
    collate_fn = get_collate_fn(src_pad_index, trg_pad_index)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


def get_data_loader(train_data, valid_data, test_data, en_tokenizer, vi_tokenizer, batch_size=BATCH_SIZE):
    
    train_data = train_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))
    valid_data = valid_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))
    test_data = test_data.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer))

    src_pad_index = en_tokenizer.token_to_id(pad_token)
    trg_pad_index = vi_tokenizer.token_to_id(pad_token)

    train_data_loader = get_data_loader(train_data, BATCH_SIZE, src_pad_index, trg_pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, BATCH_SIZE, src_pad_index, trg_pad_index)
    test_data_loader  = get_data_loader(test_data,  BATCH_SIZE, src_pad_index, trg_pad_index)

    return train_data_loader, valid_data_loader, test_data_loader


if __name__ == "__main__":
    # Load the dataset from Hugging Face
    ds = load_dataset("thainq107/iwslt2015-en-vi")
    train_data, valid_data, test_data = ds["train"], ds["validation"], ds["test"]
    save_data(train_data, valid_data, test_data)
