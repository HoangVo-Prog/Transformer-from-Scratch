import torch

unk_token = "[UNK]"
pad_token = "[PAD]"
sos_token = "<sos>"
eos_token = "<eos>"
mask_token = "[MASK]"
special_tokens = [unk_token, pad_token, sos_token, eos_token, mask_token]


BATCH_SIZE = 32
N = 6
D_MODEL = 512
D_FF = 2048
DROPOUT = 0.1
N_HEAD = 8

MAX_LENGTH = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")