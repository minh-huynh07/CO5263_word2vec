
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from word2vec.model import SkipGramModel
import numpy as np
from word2vec.utils import build_huffman_tree

class SkipGramDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0], dtype=torch.long),                torch.tensor(self.pairs[idx][1], dtype=torch.long)

def train_skipgram(
    training_pairs: list,
    vocab_size: int,
    embedding_dim: int = 100,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = SkipGramDataset(training_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            labels = torch.ones(center.shape[0], device=device)
            scores = model(center, context)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    return model, losses

def train_skipgram_softmax(
    training_pairs,
    vocab_size,
    embedding_dim=100,
    batch_size=128,
    epochs=5,
    lr=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = SkipGramDataset(training_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            logits = model(center, mode='softmax')  # (batch_size, vocab_size)
            loss = loss_fn(logits, context)  # context: (batch_size,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[Softmax] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    return model, losses

def get_negative_samples(context_indices, vocab_size, num_negatives):
    # context_indices: (batch_size,)
    # Returns (batch_size, num_negatives) negative context indices
    negatives = np.random.choice(vocab_size, size=(len(context_indices), num_negatives), replace=True)
    return torch.tensor(negatives, dtype=torch.long, device=context_indices.device)

def train_skipgram_neg_sampling(
    training_pairs,
    vocab_size,
    embedding_dim=100,
    batch_size=128,
    epochs=5,
    lr=0.01,
    num_negatives=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = SkipGramDataset(training_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            batch_size_ = center.shape[0]
            # Positive score
            pos_score = model(center, context)  # (batch_size,)
            pos_labels = torch.ones(batch_size_, device=device)
            # Negative samples
            neg_context = get_negative_samples(context, vocab_size, num_negatives)  # (batch_size, num_negatives)
            neg_center = center.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
            neg_context = neg_context.reshape(-1)
            neg_score = model(neg_center, neg_context)  # (batch_size * num_negatives,)
            neg_labels = torch.zeros(neg_score.shape[0], device=device)
            # Loss
            loss = loss_fn(pos_score, pos_labels) + loss_fn(neg_score, neg_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[NegSampling] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    return model, losses

# CBOW Dataset
class CBOWDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], int]]):
        self.pairs = pairs
        # Find max context length to pad
        self.max_context_len = max(len(context) for context, _ in pairs) if pairs else 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context_words, center = self.pairs[idx]
        # Pad context_words to max_context_len
        padded_context = context_words + [0] * (self.max_context_len - len(context_words))
        return torch.tensor(padded_context, dtype=torch.long), torch.tensor(center, dtype=torch.long)

# CBOW Model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_words: torch.Tensor, center: torch.Tensor = None, mode: str = 'dot') -> torch.Tensor:
        # context_words: (batch_size, num_context) or (num_context,)
        # Average context embeddings (ignore padding tokens - index 0)
        context_embeds = self.context_embeddings(context_words)  # (batch_size, num_context, embed_dim) or (num_context, embed_dim)
        
        # Create mask to ignore padding tokens (index 0)
        mask = (context_words != 0).float().unsqueeze(-1)  # (batch_size, num_context, 1) or (num_context, 1)
        
        # Calculate weighted average (only count real tokens)
        masked_embeds = context_embeds * mask
        sum_embeds = masked_embeds.sum(dim=-2)  # (batch_size, embed_dim) or (embed_dim,)
        count = mask.sum(dim=-2)  # (batch_size, 1) or (1,)
        avg_context = sum_embeds / (count + 1e-8)  # Avoid division by zero
        
        if mode == 'softmax':
            # Return logits for entire vocabulary
            logits = torch.matmul(avg_context, self.output_embeddings.weight.t())  # (batch_size, vocab_size) or (vocab_size,)
            return logits
        else:
            # center: (batch_size,) or scalar
            center_embed = self.output_embeddings(center)  # (batch_size, embed_dim) or (embed_dim,)
            score = (avg_context * center_embed).sum(dim=-1)  # (batch_size,) or scalar
            return score

    def get_input_embedding(self):
        return self.context_embeddings.weight.data

def train_cbow_softmax(
    training_pairs,
    vocab_size,
    embedding_dim=100,
    batch_size=128,
    epochs=5,
    lr=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = CBOWDataset(training_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CBOWModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for context_words, center in dataloader:
            context_words, center = context_words.to(device), center.to(device)
            logits = model(context_words, mode='softmax')  # (batch_size, vocab_size)
            loss = loss_fn(logits, center)  # center: (batch_size,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[CBOW-Softmax] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    return model, losses

def train_cbow_neg_sampling(
    training_pairs,
    vocab_size,
    embedding_dim=100,
    batch_size=128,
    epochs=5,
    lr=0.01,
    num_negatives=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = CBOWDataset(training_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CBOWModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for context_words, center in dataloader:
            context_words, center = context_words.to(device), center.to(device)
            batch_size_ = center.shape[0]
            # Positive score
            pos_score = model(context_words, center)  # (batch_size,)
            pos_labels = torch.ones(batch_size_, device=device)
            # Negative samples
            neg_center = get_negative_samples(center, vocab_size, num_negatives)  # (batch_size, num_negatives)
            neg_context = context_words.unsqueeze(1).expand(-1, num_negatives, -1).reshape(-1, context_words.shape[-1])
            neg_center = neg_center.reshape(-1)
            neg_score = model(neg_context, neg_center)  # (batch_size * num_negatives,)
            neg_labels = torch.zeros(neg_score.shape[0], device=device)
            # Loss
            loss = loss_fn(pos_score, pos_labels) + loss_fn(neg_score, neg_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[CBOW-NegSampling] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    return model, losses
