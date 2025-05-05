import os
import pickle
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LENGTH, sos_token, eos_token, pad_token, unk_token, special_tokens

# ------------------------
# Collate Function
# ------------------------
def collate_fn(batch):
    src_ids = torch.stack([
        torch.tensor(item['src_ids']) if not isinstance(item['src_ids'], torch.Tensor) else item['src_ids']
        for item in batch
    ])
    trg_ids = torch.stack([
        torch.tensor(item['trg_ids']) if not isinstance(item['trg_ids'], torch.Tensor) else item['trg_ids']
        for item in batch
    ])
    return {'src_ids': src_ids, 'trg_ids': trg_ids}


# ------------------------
# Tokenization Utilities
# ------------------------
def bytepair_tokenize(raw_data):
    en_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_trainer = BpeTrainer(special_tokens=special_tokens)
    en_tokenizer.train_from_iterator(raw_data["en"], trainer=en_trainer)

    vi_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    vi_tokenizer.pre_tokenizer = Whitespace()
    vi_trainer = BpeTrainer(special_tokens=special_tokens)
    vi_tokenizer.train_from_iterator(raw_data["vi"], trainer=vi_trainer)

    return en_tokenizer, vi_tokenizer

def tokenize(data, tokenizer):
    encoding = tokenizer.encode(data)
    return [tokenizer.token_to_id(sos_token)] + encoding.ids

def tokenize_and_numericalize(data, src_tokenizer, trg_tokenizer, max_length=MAX_LENGTH):
    def pad_or_truncate(ids, pad_index, eos_index):
        if len(ids) >= max_length:
            return ids[:max_length-1] + [eos_index]
        else:
            return ids + [pad_index] * (max_length - len(ids) - 1) + [eos_index]

    src_ids = tokenize(data['en'], src_tokenizer)
    trg_ids = tokenize(data['vi'], trg_tokenizer)
    src_pad_index = src_tokenizer.token_to_id(pad_token)
    trg_pad_index = trg_tokenizer.token_to_id(pad_token)
    src_eos_index = src_tokenizer.token_to_id(eos_token)
    trg_eos_index = trg_tokenizer.token_to_id(eos_token)

    return {
        'src_ids': torch.tensor(pad_or_truncate(src_ids, src_pad_index, src_eos_index)),
        'trg_ids': torch.tensor(pad_or_truncate(trg_ids, trg_pad_index, trg_eos_index)),
    }

# ------------------------
# DataLoader Wrapper
# ------------------------
def get_data_loader(dataset, batch_size, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ------------------------
# Save/Load Dataset State
# ------------------------
def save_preprocessed_datasets(train, valid, test, path="Data/tokenized"):
    DatasetDict({"train": train, "validation": valid, "test": test}).save_to_disk(path)

def save_tokenizers(en_tokenizer, vi_tokenizer, path="Data/tokenizers.pkl"):
    with open(path, 'wb') as f:
        pickle.dump((en_tokenizer, vi_tokenizer), f)

def load_tokenizers(path="Data/tokenizers.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ------------------------
# Central Caching Loader
# ------------------------
def cache_or_process(BATCH_SIZE=32):
    cache_path = "Data/tokenized"
    tokenizer_path = "Data/tokenizers.pkl"

    if os.path.exists(cache_path) and os.path.exists(tokenizer_path):
        print("✅ Loading cached preprocessed datasets and tokenizers...")
        datasets = load_from_disk(cache_path)
        en_tokenizer, vi_tokenizer = load_tokenizers(tokenizer_path)
    else:
        print("🔄 Preprocessing and tokenizing datasets...")
        raw = load_dataset("thainq107/iwslt2015-en-vi")
        train_raw, valid_raw, test_raw = raw["train"], raw["validation"], raw["test"]
        en_tokenizer, vi_tokenizer = bytepair_tokenize(train_raw)

        train = train_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)
        valid = valid_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)
        test = test_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)

        save_preprocessed_datasets(train, valid, test, cache_path)
        save_tokenizers(en_tokenizer, vi_tokenizer, tokenizer_path)
        datasets = {"train": train, "validation": valid, "test": test}

    train_loader = get_data_loader(datasets["train"], BATCH_SIZE, shuffle=True)
    valid_loader = get_data_loader(datasets["validation"], BATCH_SIZE)
    test_loader = get_data_loader(datasets["test"], BATCH_SIZE)

    return train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer

def main():
    print("🚀 Initializing data pipeline...")
    train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer = cache_or_process()

    print("✅ DataLoaders ready:")
    print(f"  ├─ Train batches: {len(train_loader)}")
    print(f"  ├─ Valid batches: {len(valid_loader)}")
    print(f"  └─ Test batches : {len(test_loader)}")

    # Example: Peek at a single batch
    sample = next(iter(train_loader))
    print("🔍 Sample batch:")
    print(f"  src_ids shape: {sample['src_ids'].shape}")
    print(f"  trg_ids shape: {sample['trg_ids'].shape}")

if __name__ == "__main__":
    main()
