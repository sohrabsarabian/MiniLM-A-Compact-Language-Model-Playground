import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from config import *
from utils import calculate_loss, generate_sample, get_batch


def train(model, data, sp):
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if p.dim() >= 2], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if p.dim() < 2], 'weight_decay': 0.0}
    ], lr=LR, betas=(0.9, 0.99))

    scheduler = CosineAnnealingLR(optimizer, TRAIN_ITERS, eta_min=LR / 10)

    best_val_loss = float('inf')

    for iter in tqdm(range(TRAIN_ITERS)):
        xb, yb = get_batch(data, "train")
        logits, loss = model(xb, yb)

        if iter % EVAL_INTERVAL == 0:
            losses = calculate_loss(model, data)
            print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            sample = generate_sample(model, sp, "The mountain in my city is")
            print(f"Sample: {sample}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"Saving model with val loss: {best_val_loss:.4f}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'iteration': iter,
                }, CHECKPOINT_DIR + CHECKPOINT_FN)

            if WANDB_LOG:
                wandb.log({
                    "loss/train": losses['train'],
                    "loss/val": losses['val'],
                    "lr": scheduler.get_last_lr()[0],
                }, step=iter)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

    if WANDB_LOG:
        wandb.finish()


def setup_wandb():
    if WANDB_LOG:
        wandb.init(project=WANDB_PROJECT, config={
            "batch_size": BATCH_SIZE,
            "context": CONTEXT,
            "embed_size": EMBED_SIZE,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "learning_rate": LR,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        })
