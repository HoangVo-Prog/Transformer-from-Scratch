from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import unk_token, special_tokens


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

def vi_en_pretrained_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi")
    return tokenizer

