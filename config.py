import torch
from data import bytepair_tokenize, load_data


train_data, valid_data, test_data = load_data()
en_tokenizer, vi_tokenizer = bytepair_tokenize(train_data)

unk_token = "[UNK]"
pad_token = "[PAD]"
sos_token = "<sos>"
eos_token = "<eos>"
special_tokens = [unk_token, pad_token, sos_token, eos_token]

MAX_LENGTH = 50
VOCAB_SIZE = len(en_tokenizer.get_vocab())   
OUTPUT_DIM = len(vi_tokenizer.get_vocab())    
EMBEDDING_DIM = 256 
BATCH_SIZE = 32
HIDDEN_SIZE = 512            
N_LAYERS = 2
ENCODER_DROPOUT = 0.2
DECODER_DROPOUT = 0.2
BIDIRECTIONAL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")