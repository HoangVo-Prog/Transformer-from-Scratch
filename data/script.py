import os
import pickle
import torch
import argparse
import shutil
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
import sys

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LENGTH, sos_token, eos_token, pad_token, unk_token, special_tokens

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ------------------------
# Configuration
# ------------------------
TOKENIZER_TYPES = {
    'bpe': 'Custom BPE tokenizer',
    'vinai': 'VinAI pretrained tokenizer (vinai/vinai-translate-en2vi)',
    'mbart': 'mBART multilingual tokenizer',
    'mt5': 'mT5 multilingual tokenizer'
}

PRETRAINED_MODEL_PATHS = {
    'vinai': "vinai/vinai-translate-en2vi",
    'mbart': "facebook/mbart-large-50-many-to-many-mmt",
    'mt5': "google/mt5-base"
}

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



# ------------------------
# Tokenizer Management
# ------------------------
class TokenizerManager:
    """Centralized tokenizer management"""

    @staticmethod
    def get_pretrained_tokenizer(tokenizer_type):
        """Load pretrained tokenizer"""
        if tokenizer_type not in PRETRAINED_MODEL_PATHS:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATHS[tokenizer_type])

    @staticmethod
    def save_tokenizer(tokenizer, tokenizer_type, path_dir):
        """Save tokenizer with metadata"""
        Path(path_dir).mkdir(parents=True, exist_ok=True)
        tokenizer_path = Path(path_dir) / f"tokenizer_{tokenizer_type}.pkl"
        config_path = Path(path_dir) / f"config_{tokenizer_type}.pkl"

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)

        config = {
            'tokenizer_type': tokenizer_type,
            'vocab_size': getattr(tokenizer, 'vocab_size', len(tokenizer.get_vocab())),
            'model_max_length': getattr(tokenizer, 'model_max_length', MAX_LENGTH)
        }

        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

    @staticmethod
    def load_tokenizer(tokenizer_type, path_dir):
        """Load tokenizer with verification"""
        tokenizer_path = Path(path_dir) / f"tokenizer_{tokenizer_type}.pkl"
        config_path = Path(path_dir) / f"config_{tokenizer_type}.pkl"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                print(f"✅ Loaded {config['tokenizer_type']} tokenizer (vocab_size: {config['vocab_size']})")

        return tokenizer

    @staticmethod
    def save_bpe_tokenizers(en_tokenizer, vi_tokenizer, path):
        """Save BPE tokenizers"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump((en_tokenizer, vi_tokenizer), f)

    @staticmethod
    def load_bpe_tokenizers(path):
        """Load BPE tokenizers"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ------------------------
# Tokenization Functions
# ------------------------
class DataTokenizer:
    """Handles tokenization for different tokenizer types"""

    @staticmethod
    def tokenize_pretrained(text, tokenizer, max_length=MAX_LENGTH):
        """Tokenize using pretrained tokenizer"""
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0)

    @staticmethod
    def tokenize_and_numericalize_pretrained(data, tokenizer, max_length=MAX_LENGTH):
        """Process data using pretrained tokenizer"""
        src_ids = DataTokenizer.tokenize_pretrained(data['en'], tokenizer, max_length)
        trg_ids = DataTokenizer.tokenize_pretrained(data['vi'], tokenizer, max_length)

        return {
            'src_ids': src_ids,
            'trg_ids': trg_ids,
        }

    @staticmethod
    def tokenize_and_numericalize_bpe(data, src_tokenizer, trg_tokenizer, max_length=MAX_LENGTH):
        """Process data using BPE tokenizers"""

        def pad_or_truncate(ids, pad_index, eos_index):
            if len(ids) >= max_length:
                return ids[:max_length - 1] + [eos_index]
            else:
                return ids + [pad_index] * (max_length - len(ids) - 1) + [eos_index]

        def tokenize_bpe(text, tokenizer):
            encoding = tokenizer.encode(text)
            return [tokenizer.token_to_id(sos_token)] + encoding.ids

        src_ids = tokenize_bpe(data['en'], src_tokenizer)
        trg_ids = tokenize_bpe(data['vi'], trg_tokenizer)

        src_pad_index = src_tokenizer.token_to_id(pad_token)
        trg_pad_index = trg_tokenizer.token_to_id(pad_token)
        src_eos_index = src_tokenizer.token_to_id(eos_token)
        trg_eos_index = trg_tokenizer.token_to_id(eos_token)

        return {
            'src_ids': torch.tensor(pad_or_truncate(src_ids, src_pad_index, src_eos_index)),
            'trg_ids': torch.tensor(pad_or_truncate(trg_ids, trg_pad_index, trg_eos_index)),
        }


# ------------------------
# Data Processing Pipeline
# ------------------------
class DataProcessor:
    """Main data processing pipeline"""

    def __init__(self, tokenizer_type='vinai', tokenizer_path=None, output_path=None, batch_size=32):
        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else Path("pretrained")
        self.output_path = Path(output_path) if output_path else Path(f"pretrained/tokenized_{tokenizer_type}")
        self.batch_size = batch_size
        self.tokenizer_manager = TokenizerManager()
        self.data_tokenizer = DataTokenizer()

    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        src_ids = torch.stack([
            torch.tensor(item['src_ids']) if not isinstance(item['src_ids'], torch.Tensor) else item['src_ids']
            for item in batch
        ])
        trg_ids = torch.stack([
            torch.tensor(item['trg_ids']) if not isinstance(item['trg_ids'], torch.Tensor) else item['trg_ids']
            for item in batch
        ])
        return {'src_ids': src_ids, 'trg_ids': trg_ids}

    def _get_data_loader(self, dataset, shuffle=False):
        """Create DataLoader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def _save_datasets(self, train, valid, test):
        """Save processed datasets"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": train, "validation": valid, "test": test}).save_to_disk(str(self.output_path))

    def _clean_cache(self):
        """Clean existing cache"""
        for i in [self.output_path, self.tokenizer_path]:
            if i.exists():
                if i.is_file():
                    i.unlink()
                elif i.is_dir():
                    shutil.rmtree(i)

    def _load_raw_data(self):
        """Load raw dataset"""
        raw = load_dataset("thainq107/iwslt2015-en-vi")
        return raw["train"], raw["validation"], raw["test"]

    def _process_with_pretrained(self, force_download=False):
        """Process data with pretrained tokenizer"""
        tokenizer_exists = (self.tokenizer_path / f"tokenizer_{self.tokenizer_type}.pkl").exists()
        cache_exists = self.output_path.exists()

        if force_download or not cache_exists or not tokenizer_exists:
            print(f"🔄 Processing with {TOKENIZER_TYPES[self.tokenizer_type]}...")
            self._clean_cache()

            # Load data and tokenizer
            train_raw, valid_raw, test_raw = self._load_raw_data()
            tokenizer = self.tokenizer_manager.get_pretrained_tokenizer(self.tokenizer_type)

            # Process datasets
            process_func = lambda x: self.data_tokenizer.tokenize_and_numericalize_pretrained(x, tokenizer)
            train = train_raw.map(process_func, num_proc=4)
            valid = valid_raw.map(process_func, num_proc=4)
            test = test_raw.map(process_func, num_proc=4)

            # Save results
            self._save_datasets(train, valid, test)
            self.tokenizer_manager.save_tokenizer(tokenizer, self.tokenizer_type, str(self.tokenizer_path))

            datasets = {"train": train, "validation": valid, "test": test}
        else:
            print(f"✅ Loading cached data for {TOKENIZER_TYPES[self.tokenizer_type]}...")
            datasets = load_from_disk(str(self.output_path))
            tokenizer = self.tokenizer_manager.load_tokenizer(self.tokenizer_type, str(self.tokenizer_path))

        return datasets, tokenizer

    def _process_with_bpe(self, force_download=False):
        """Process data with BPE tokenizer"""
        tokenizer_path = self.tokenizer_path / f"tokenizers_{self.tokenizer_type}.pkl"
        cache_exists = self.output_path.exists()

        if force_download or not cache_exists or not tokenizer_path.exists():
            print("🔄 Processing with BPE tokenizer...")
            self._clean_cache()

            # Load data and create BPE tokenizers
            train_raw, valid_raw, test_raw = self._load_raw_data()
            en_tokenizer, vi_tokenizer = bytepair_tokenize(train_raw)

            # Process datasets
            process_func = lambda x: self.data_tokenizer.tokenize_and_numericalize_bpe(x, en_tokenizer, vi_tokenizer)
            train = train_raw.map(process_func, num_proc=4)
            valid = valid_raw.map(process_func, num_proc=4)
            test = test_raw.map(process_func, num_proc=4)

            # Save results
            self._save_datasets(train, valid, test)
            self.tokenizer_manager.save_bpe_tokenizers(en_tokenizer, vi_tokenizer, str(tokenizer_path))

            datasets = {"train": train, "validation": valid, "test": test}
            tokenizer = {'en': en_tokenizer, 'vi': vi_tokenizer}
        else:
            print("✅ Loading cached BPE data...")
            datasets = load_from_disk(str(self.output_path))
            en_tokenizer, vi_tokenizer = self.tokenizer_manager.load_bpe_tokenizers(str(tokenizer_path))
            tokenizer = {'en': en_tokenizer, 'vi': vi_tokenizer}

        return datasets, tokenizer

    def process(self, force_download=False):
        """Main processing function"""
        if self.tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}")

        if self.tokenizer_type == 'bpe':
            datasets, tokenizer = self._process_with_bpe(force_download)
        else:
            datasets, tokenizer = self._process_with_pretrained(force_download)

        # Create data loaders
        train_loader = self._get_data_loader(datasets["train"], shuffle=True)
        valid_loader = self._get_data_loader(datasets["validation"])
        test_loader = self._get_data_loader(datasets["test"])

        return train_loader, valid_loader, test_loader, tokenizer


# ------------------------
# Debug Utilities
# ------------------------
class DebugUtils:
    """Debugging utilities for token sequences"""

    @staticmethod
    def debug_pretrained_tokens(tokenizer, ids):
        """Debug pretrained tokenizer tokens"""
        try:
            decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            return tokens, decoded_text
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            return [f"ID_{id}" for id in ids], "Unable to decode"

    @staticmethod
    def debug_bpe_tokens(tokenizer, ids):
        """Debug BPE tokenizer tokens"""
        token_map = {
            tokenizer.token_to_id(sos_token): sos_token,
            tokenizer.token_to_id(eos_token): eos_token,
            tokenizer.token_to_id(pad_token): pad_token,
            tokenizer.token_to_id(unk_token): unk_token
        }

        readable_tokens = []
        for id in ids:
            if id in token_map:
                readable_tokens.append(token_map[id])
            else:
                try:
                    readable_tokens.append(tokenizer.id_to_token(id))
                except:
                    readable_tokens.append(f"ID_{id}")

        return readable_tokens

    @staticmethod
    def print_sample_analysis(sample, tokenizer, tokenizer_type):
        """Print detailed analysis of a sample batch"""
        print("🔍 Sample batch analysis:")
        print(f"  src_ids shape: {sample['src_ids'].shape}")
        print(f"  trg_ids shape: {sample['trg_ids'].shape}")

        if tokenizer_type == 'bpe':
            src_tokens = DebugUtils.debug_bpe_tokens(tokenizer['en'], sample['src_ids'][0])
            trg_tokens = DebugUtils.debug_bpe_tokens(tokenizer['vi'], sample['trg_ids'][0])

            print("\nSource sequence:")
            print("Tokens:", src_tokens[:20], "..." if len(src_tokens) > 20 else "")
            print("IDs:", sample['src_ids'][0][:20].tolist(), "..." if len(sample['src_ids'][0]) > 20 else "")

            print("\nTarget sequence:")
            print("Tokens:", trg_tokens[:20], "..." if len(trg_tokens) > 20 else "")
            print("IDs:", sample['trg_ids'][0][:20].tolist(), "..." if len(sample['trg_ids'][0]) > 20 else "")

            # Count special tokens
            src_pad_count = src_tokens.count(pad_token)
            trg_pad_count = trg_tokens.count(pad_token)
            print(f"\nPadding counts - Source: {src_pad_count}, Target: {trg_pad_count}")

        else:
            src_tokens, src_decoded = DebugUtils.debug_pretrained_tokens(tokenizer, sample['src_ids'][0])
            trg_tokens, trg_decoded = DebugUtils.debug_pretrained_tokens(tokenizer, sample['trg_ids'][0])

            print("\nSource sequence:")
            print("Tokens:", src_tokens[:20], "..." if len(src_tokens) > 20 else "")
            print("Decoded:", src_decoded[:100], "..." if len(src_decoded) > 100 else "")
            print("IDs:", sample['src_ids'][0][:20].tolist(), "..." if len(sample['src_ids'][0]) > 20 else "")

            print("\nTarget sequence:")
            print("Tokens:", trg_tokens[:20], "..." if len(trg_tokens) > 20 else "")
            print("Decoded:", trg_decoded[:100], "..." if len(trg_decoded) > 100 else "")
            print("IDs:", sample['trg_ids'][0][:20].tolist(), "..." if len(sample['trg_ids'][0]) > 20 else "")

            # Count padding tokens
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            src_pad_count = (sample['src_ids'][0] == pad_token_id).sum().item()
            trg_pad_count = (sample['trg_ids'][0] == pad_token_id).sum().item()
            print(f"\nPadding counts - Source: {src_pad_count}, Target: {trg_pad_count}")

            print(f"\nTokenizer info:")
            print(f"Vocab size: {tokenizer.vocab_size}")
            print(f"Model max length: {tokenizer.model_max_length}")


# ------------------------
# CLI Interface
# ------------------------
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Data processing pipeline for translation models")
    parser.add_argument("--tokenizer-type", choices=TOKENIZER_TYPES.keys(), default='vinai',
                        help="Type of tokenizer to use")
    parser.add_argument("--tokenizer-path", default="pretrained",
                        help="Path to save/load tokenizers")
    parser.add_argument("--output-path",
                        help="Path to save processed datasets (default: pretrained/tokenized_{tokenizer_type})")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Batch size for data loaders")
    parser.add_argument("--force-download", action="store_true",
                        help="Force reprocessing even if cache exists")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug information")

    return parser.parse_args()


def print_tokenizer_options():
    """Print available tokenizer options"""
    print("Available tokenizer types:")
    for key, desc in TOKENIZER_TYPES.items():
        print(f"  - {key}: {desc}")
    print()


def main():
    """Main function with CLI interface"""
    args = parse_args()

    # Set default output path if not provided
    if not args.output_path:
        args.output_path = f"pretrained/tokenized_{args.tokenizer_type}"

    print(f"🚀 Initializing data pipeline with {TOKENIZER_TYPES[args.tokenizer_type]}...")
    print(f"Configuration:")
    print(f"  ├─ Tokenizer type: {args.tokenizer_type}")
    print(f"  ├─ Tokenizer path: {args.tokenizer_path}")
    print(f"  ├─ Output path: {args.output_path}")
    print(f"  ├─ Batch size: {args.batch_size}")
    print(f"  └─ Force download: {args.force_download}")

    # Initialize processor
    processor = DataProcessor(
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )

    # Process data
    train_loader, valid_loader, test_loader, tokenizer = processor.process(args.force_download)

    print("✅ DataLoaders ready:")
    print(f"  ├─ Train batches: {len(train_loader)}")
    print(f"  ├─ Valid batches: {len(valid_loader)}")
    print(f"  └─ Test batches: {len(test_loader)}")

    # Debug information if requested
    if args.debug:
        sample = next(iter(train_loader))
        DebugUtils.print_sample_analysis(sample, tokenizer, args.tokenizer_type)


# ------------------------
# Backward Compatibility Functions
# ------------------------
def cache_or_process_pretrained(tokenizer_type='vinai', tokenizer_path=None, output_path=None, batch_size=32,
                                force_download=False):
    """Backward compatibility function for pretrained tokenizers"""
    if tokenizer_path is None:
        tokenizer_path = "pretrained"
    if output_path is None:
        output_path = f"pretrained/tokenized_{tokenizer_type}"

    processor = DataProcessor(
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        batch_size=batch_size
    )
    return processor.process(force_download)


def cache_or_process_bpe(tokenizer_path=None, output_path=None, batch_size=32, force_download=False):
    """Backward compatibility function for BPE with custom paths"""
    # Handle the case where tokenizer_path points directly to the pkl file
    if tokenizer_path and str(tokenizer_path).endswith('.pkl'):
        # Extract the directory and use it as tokenizer_path
        tokenizer_dir = str(Path(tokenizer_path).parent)
        expected_filename = f"tokenizers_bpe.pkl"
        actual_filename = Path(tokenizer_path).name

        # If the filename doesn't match expected pattern, we need to handle it
        if actual_filename != expected_filename:
            # Create a temporary processor to handle the custom file path
            processor = BPEProcessor(
                tokenizer_path=tokenizer_path,
                output_path=output_path or "bpe/tokenized",
                batch_size=batch_size
            )
            return processor.process(force_download)

    # Standard path handling
    if tokenizer_path is None:
        tokenizer_path = "bpe"
    if output_path is None:
        output_path = "bpe/tokenized"

    processor = DataProcessor(
        tokenizer_type='bpe',
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        batch_size=batch_size
    )
    train_loader, valid_loader, test_loader, tokenizer = processor.process(force_download)
    return train_loader, valid_loader, test_loader, tokenizer['en'], tokenizer['vi']


def cache_or_process(tokenizer_path=None, output_path=None, batch_size=32, force_download=False):
    """Default backward compatibility function"""
    return cache_or_process_pretrained('vinai', tokenizer_path, output_path, batch_size, force_download)


# ------------------------
# Special BPE Processor for Custom File Paths
# ------------------------
class BPEProcessor:
    """Special processor for BPE with custom file paths"""

    def __init__(self, tokenizer_path, output_path, batch_size=32):
        self.tokenizer_path = Path(tokenizer_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.tokenizer_manager = TokenizerManager()
        self.data_tokenizer = DataTokenizer()

    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        src_ids = torch.stack([
            torch.tensor(item['src_ids']) if not isinstance(item['src_ids'], torch.Tensor) else item['src_ids']
            for item in batch
        ])
        trg_ids = torch.stack([
            torch.tensor(item['trg_ids']) if not isinstance(item['trg_ids'], torch.Tensor) else item['trg_ids']
            for item in batch
        ])
        return {'src_ids': src_ids, 'trg_ids': trg_ids}

    def _get_data_loader(self, dataset, shuffle=False):
        """Create DataLoader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def _save_datasets(self, train, valid, test):
        """Save processed datasets"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": train, "validation": valid, "test": test}).save_to_disk(str(self.output_path))

    def _clean_cache(self):
        """Clean existing cache"""
        if self.output_path.exists():
            if self.output_path.is_file():
                self.output_path.unlink()
            elif self.output_path.is_dir():
                shutil.rmtree(self.output_path)

    def _load_raw_data(self):
        """Load raw dataset"""
        raw = load_dataset("thainq107/iwslt2015-en-vi")
        return raw["train"], raw["validation"], raw["test"]

    def process(self, force_download=False):
        """Process with custom BPE tokenizer path"""
        cache_exists = self.output_path.exists()
        tokenizer_exists = self.tokenizer_path.exists()

        if force_download or not cache_exists or not tokenizer_exists:
            print("🔄 Processing with BPE tokenizer (custom path)...")
            self._clean_cache()

            if not tokenizer_exists:
                # Create BPE tokenizers if they don't exist
                train_raw, valid_raw, test_raw = self._load_raw_data()
                en_tokenizer, vi_tokenizer = bytepair_tokenize(train_raw)
                self.tokenizer_manager.save_bpe_tokenizers(en_tokenizer, vi_tokenizer, str(self.tokenizer_path))
            else:
                # Load existing BPE tokenizers
                en_tokenizer, vi_tokenizer = self.tokenizer_manager.load_bpe_tokenizers(str(self.tokenizer_path))

            # Load and process data
            train_raw, valid_raw, test_raw = self._load_raw_data()
            process_func = lambda x: self.data_tokenizer.tokenize_and_numericalize_bpe(x, en_tokenizer, vi_tokenizer)
            train = train_raw.map(process_func, num_proc=4)
            valid = valid_raw.map(process_func, num_proc=4)
            test = test_raw.map(process_func, num_proc=4)

            # Save results
            self._save_datasets(train, valid, test)
            datasets = {"train": train, "validation": valid, "test": test}
        else:
            print("✅ Loading cached BPE data (custom path)...")
            datasets = load_from_disk(str(self.output_path))
            en_tokenizer, vi_tokenizer = self.tokenizer_manager.load_bpe_tokenizers(str(self.tokenizer_path))

        # Create data loaders
        train_loader = self._get_data_loader(datasets["train"], shuffle=True)
        valid_loader = self._get_data_loader(datasets["validation"])
        test_loader = self._get_data_loader(datasets["test"])

        return train_loader, valid_loader, test_loader, en_tokenizer, vi_tokenizer


if __name__ == "__main__":
    main()