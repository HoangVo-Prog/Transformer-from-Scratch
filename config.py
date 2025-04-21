import torch


unk_token = "[UNK]"
pad_token = "[PAD]"
sos_token = "<sos>"
eos_token = "<eos>"
special_tokens = [unk_token, pad_token, sos_token, eos_token]

MAX_LENGTH = 50
VOCAB_SIZE, OUTPUT_DIM = 0, 0
EMBEDDING_DIM = 256 
BATCH_SIZE = 32
HIDDEN_SIZE = 512            
N_LAYERS = 2
ENCODER_DROPOUT = 0.2
DECODER_DROPOUT = 0.2
BIDIRECTIONAL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")