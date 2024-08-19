import torch
import sentencepiece as spm
from config import *


def load_tokenizer(model_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    return sp, sp.get_piece_size()


def encode(sp, text):
    return sp.Encode(text)


def decode(sp, tokens):
    return sp.Decode(tokens)


def get_batch(data, split='train'):
    data = data['train'] if split == 'train' else data['val']
    ix = torch.randint(len(data) - CONTEXT, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT] for i in ix])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def calculate_loss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def generate_sample(model, sp, input_text, max_new_tokens=64):
    input_ids = torch.tensor(encode(sp, input_text), dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated_ids = model.generate(input_ids, max_tokens=max_new_tokens)[0].tolist()
    return decode(sp, generated_ids)


def load_and_encode_dataset(file_path, sp):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(sp, text), dtype=torch.long)
    return data


def split_dataset(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return {'train': train_data, 'val': val_data}
