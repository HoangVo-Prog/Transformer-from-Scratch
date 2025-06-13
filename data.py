import os
import pickle
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LENGTH, sos_token, eos_token, pad_token, unk_token, special_tokens

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

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
def cache_or_process(BATCH_SIZE=32, FORCE_DOWNLOAD=False):
    cache_path = "Data/tokenized"
    tokenizer_path = "Data/tokenizers.pkl"

    if FORCE_DOWNLOAD or not os.path.exists(cache_path) or not os.path.exists(tokenizer_path):
        print("🔄 Preprocessing and tokenizing datasets...")
        for i in [cache_path, tokenizer_path]:
            if os.path.isfile(i):
                os.remove(i)
            elif os.path.isdir(i):
                shutil.rmtree(i)

        raw = load_dataset("thainq107/iwslt2015-en-vi")
        train_raw, valid_raw, test_raw = raw["train"], raw["validation"], raw["test"]
        en_tokenizer, vi_tokenizer = bytepair_tokenize(train_raw)

        train = train_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)
        valid = valid_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)
        test = test_raw.map(lambda x: tokenize_and_numericalize(x, en_tokenizer, vi_tokenizer), num_proc=4)

        save_preprocessed_datasets(train, valid, test, cache_path)
        save_tokenizers(en_tokenizer, vi_tokenizer, tokenizer_path)
        datasets = {"train": train, "validation": valid, "test": test}
    else:
        print("✅ Loading cached preprocessed datasets and tokenizers...")
        datasets = load_from_disk(cache_path)
        en_tokenizer, vi_tokenizer = load_tokenizers(tokenizer_path)

    train_loader = get_data_loader(datasets["train"], BATCH_SIZE, shuffle=True)
    valid_loader = get_data_loader(datasets["validation"], BATCH_SIZE)
    test_loader = get_data_loader(datasets["test"], BATCH_SIZE)

    return train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer


def debug_token_sequence(tokenizer, ids):
    """Convert token IDs back to readable tokens for debugging"""
    token_map = {
        tokenizer.token_to_id(sos_token): sos_token,
        tokenizer.token_to_id(eos_token): eos_token,
        tokenizer.token_to_id(pad_token): pad_token,
        tokenizer.token_to_id(unk_token): unk_token
    }
    
    # Convert IDs to readable tokens
    readable_tokens = []
    for id in ids:
        if id in token_map:
            readable_tokens.append(token_map[id])
        else:
            # For regular tokens, get them from the tokenizer vocabulary if possible
            try:
                readable_tokens.append(tokenizer.id_to_token(id))
            except:
                readable_tokens.append(f"ID_{id}")
    
    return readable_tokens


def main():
    print("🚀 Initializing data pipeline...")
    train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer = cache_or_process(FORCE_DOWNLOAD=True)

    print("✅ DataLoaders ready:")
    print(f"  ├─ Train batches: {len(train_loader)}")
    print(f"  ├─ Valid batches: {len(valid_loader)}")
    print(f"  └─ Test batches : {len(test_loader)}")

    # Get a sample batch
    sample = next(iter(train_loader))
    print("🔍 Sample batch:")
    print(f"  src_ids shape: {sample['src_ids'].shape}")
    print(f"  trg_ids shape: {sample['trg_ids'].shape}")
    
    # Debug first example in batch
    print("\n🔍 Examining first example in batch:")
    src_tokens = debug_token_sequence(en_tokenizer, sample['src_ids'][0])
    trg_tokens = debug_token_sequence(vi_tokenizer, sample['trg_ids'][0])
    
    print("Source sequence structure:")
    print(src_tokens)
    print(sample['src_ids'][0])
    
    print("Target sequence structure:")
    print(trg_tokens)
    print(sample['trg_ids'][0])
    
    # Count special tokens
    src_sos_count = src_tokens.count(sos_token)
    src_eos_count = src_tokens.count(eos_token)
    src_pad_count = src_tokens.count(pad_token)
    
    trg_sos_count = trg_tokens.count(sos_token)
    trg_eos_count = trg_tokens.count(eos_token)
    trg_pad_count = trg_tokens.count(pad_token)
    
    print("\nSpecial token counts:")
    print(f"Source: SOS={src_sos_count}, EOS={src_eos_count}, PAD={src_pad_count}")
    print(f"Target: SOS={trg_sos_count}, EOS={trg_eos_count}, PAD={trg_pad_count}")
    
    
if __name__ == "__main__":
    main()
