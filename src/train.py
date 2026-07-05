import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def get_data(vocab_size=5000, max_len=64, num_train=5000):
    # Токенизатор
    dataset_raw = load_dataset("roneneldan/TinyStories", split="train[:2000]")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>","<unk>","<s>","</s>"])
    tokenizer.train_from_iterator(dataset_raw["text"], trainer)
    # Загрузка данных
    full_texts = load_dataset("roneneldan/TinyStories", split=f"train[:{num_train}]")["text"]
    tokenized = tokenizer.encode_batch(full_texts)
    tokenized = [e.ids[:max_len] for e in tokenized if len(e.ids) > 5]
    padded = [t + [0]*(max_len - len(t)) for t in tokenized]
    X = torch.tensor([t[:-1] for t in padded], dtype=torch.long)
    Y = torch.tensor([t[1:] for t in padded], dtype=torch.long)
    dataset = TensorDataset(X, Y)
    return dataset, tokenizer, vocab_size

def train_model(model, loader, vocab_size, epochs=5, lr=3e-3, device='cuda'):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.3f}")
    return model
