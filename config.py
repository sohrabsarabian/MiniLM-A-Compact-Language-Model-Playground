import torch

# Architecture parameters
BATCH_SIZE = 8
CONTEXT = 512
EMBED_SIZE = 384
N_LAYERS = 7
N_HEADS = 7
BIAS = True

# Hyperparameters
LR = 3e-4
DROPOUT = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Training parameters
TRAIN_ITERS = 100000
EVAL_INTERVAL = 50
EVAL_ITERS = 3
COMPILE = False

# Paths
CHECKPOINT_DIR = 'models/'
CHECKPOINT_FN = "latest_model.pt"
TOKENIZER_MODEL = 'wiki_trained_tokenizer.model'
WIKI_TXT = 'wiki.txt'

# Device and dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Logging
WANDB_LOG = True
WANDB_PROJECT = "test"

# Default values for command-line arguments
DEFAULT_BATCH_SIZE = BATCH_SIZE
DEFAULT_CONTEXT = CONTEXT
DEFAULT_EMBED_SIZE = EMBED_SIZE
DEFAULT_N_LAYERS = N_LAYERS
DEFAULT_N_HEADS = N_HEADS
DEFAULT_LR = LR
DEFAULT_TRAIN_ITERS = TRAIN_ITERS
