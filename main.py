import torch
import argparse
from config import *
from model import GPT
from utils import load_tokenizer, load_and_encode_dataset, split_dataset
from train import train, setup_wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a small LLM")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="Context length for training")
    parser.add_argument("--embed_size", type=int, default=DEFAULT_EMBED_SIZE, help="Embedding size")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--train_iters", type=int, default=DEFAULT_TRAIN_ITERS, help="Number of training iterations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


def update_config(args):
    global BATCH_SIZE, CONTEXT, EMBED_SIZE, N_LAYERS, N_HEADS, LR, TRAIN_ITERS, WANDB_LOG
    BATCH_SIZE = args.batch_size
    CONTEXT = args.context
    EMBED_SIZE = args.embed_size
    N_LAYERS = args.n_layers
    N_HEADS = args.n_heads
    LR = args.lr
    TRAIN_ITERS = args.train_iters
    WANDB_LOG = args.wandb


def main():
    args = parse_arguments()
    update_config(args)

    # Load tokenizer
    sp, vocab_size = load_tokenizer(TOKENIZER_MODEL)

    # Load and encode dataset
    data = load_and_encode_dataset(WIKI_TXT, sp)
    data = split_dataset(data)

    # Initialize model
    model = GPT(vocab_size)
    model = model.to(DTYPE).to(DEVICE)

    if COMPILE:
        print("Compiling model...")
        model = torch.compile(model)

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    # Setup wandb
    if WANDB_LOG:
        setup_wandb()

    # Train model
    train(model, data, sp)


if __name__ == "__main__":
    main()
