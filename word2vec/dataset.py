
import collections
import random
from typing import List, Tuple, Dict

class Vocab:
    def __init__(self, tokens: List[str], min_freq: int = 1):
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                continue
            self.idx_to_token.append(token)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, 0)

    def to_tokens(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token[i] for i in indices]

def read_corpus(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [token.lower() for line in lines for token in line.strip().split()]

def subsample(tokens: List[str], token_freqs: Dict[str, int], t=1e-4) -> List[str]:
    total = sum(token_freqs.values())
    return [token for token in tokens if random.random() < (
        (t / (token_freqs[token] / total)) ** 0.5)]

def get_skip_gram_pairs(tokens: List[str], vocab: Vocab, max_window: int = 5) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(len(tokens)):
        window_size = random.randint(1, max_window)
        center = vocab[tokens[i]]
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                context = vocab[tokens[j]]
                pairs.append((center, context))
    return pairs

def get_cbow_pairs(tokens: List[str], vocab: Vocab, max_window: int = 5) -> List[Tuple[List[int], int]]:
    pairs = []
    for i in range(len(tokens)):
        window_size = random.randint(1, max_window)
        center = vocab[tokens[i]]
        context_words = []
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                context = vocab[tokens[j]]
                context_words.append(context)
        if context_words:  # only add if there are context words
            pairs.append((context_words, center))
    return pairs
