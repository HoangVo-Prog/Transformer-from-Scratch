sos_token, eos_token, pad_token, unk_token, special_tokens = "<sos>", "<eos>", "<pad>", "<unk>", ["<sos>", "<eos>", "<pad>", "<unk>"]

MAX_LENGTH=50
BATCH_SIZE=32
LEARNING_RATE=0.001
HIDDEN_DIM=1000

DROPOUT=0.2
EMBEDDING_DIM=512
d_k = EMBEDDING_DIM
USE_CUDA=True
DEVICE='cuda' if USE_CUDA else 'cpu'
SEED=42
TEACHER_FORCING_RATIO=0.5