sos_token = "<sos>"
eos_token = "<eos>"
pad_token = "<pad>" 
unk_token = "<unk>"
special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"]

MAX_LENGTH=50
BATCH_SIZE=32
LEARNING_RATE=0.001
HIDDEN_DIM=1000

DROPOUT=0.2
EMBEDDING_DIM=512
N=6
h=8
d_ff=2048
d_model=512
d_k = EMBEDDING_DIM
WARMUP_STEPS=4000


USE_CUDA=True
DEVICE='cuda' if USE_CUDA else 'cpu'
SEED=42
