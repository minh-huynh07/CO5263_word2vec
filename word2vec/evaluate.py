
import torch
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def get_embedding_matrix(model) -> np.ndarray:
    # Return input embeddings as numpy array
    return model.get_input_embedding().cpu().detach().numpy()

def find_similar(word_idx: int, embedding_matrix: np.ndarray, top_k: int = 5) -> List[int]:
    vec = embedding_matrix[word_idx]
    norms = np.linalg.norm(embedding_matrix, axis=1)
    similarities = embedding_matrix @ vec / (norms * np.linalg.norm(vec) + 1e-9)
    sorted_indices = np.argsort(-similarities)
    return sorted_indices[1:top_k + 1].tolist()

def analogy(a_idx: int, b_idx: int, c_idx: int, embedding_matrix: np.ndarray, top_k: int = 1) -> List[int]:
    vec = embedding_matrix[b_idx] - embedding_matrix[a_idx] + embedding_matrix[c_idx]
    norms = np.linalg.norm(embedding_matrix, axis=1)
    similarities = embedding_matrix @ vec / (norms * np.linalg.norm(vec) + 1e-9)
    sorted_indices = np.argsort(-similarities)
    return sorted_indices[:top_k].tolist()

def visualize_embedding_pca_2d(embedding_matrix_before, embedding_matrix_after, vocab, words, title_prefix=""):
    # Lấy index các từ cần vẽ
    indices = [vocab.token_to_idx[w] for w in words if w in vocab.token_to_idx]
    if not indices:
        print("No valid words to visualize.")
        return
    emb_before = embedding_matrix_before[indices]
    emb_after = embedding_matrix_after[indices]
    pca = PCA(n_components=2)
    emb2d_before = pca.fit_transform(emb_before)
    emb2d_after = pca.fit_transform(emb_after)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(emb2d_before[:, 0], emb2d_before[:, 1], c='gray')
    for i, w in enumerate(words):
        if w in vocab.token_to_idx:
            plt.annotate(w, (emb2d_before[i, 0], emb2d_before[i, 1]))
    plt.title(f"{title_prefix}Before Training (PCA 2D)")
    plt.subplot(1, 2, 2)
    plt.scatter(emb2d_after[:, 0], emb2d_after[:, 1], c='blue')
    for i, w in enumerate(words):
        if w in vocab.token_to_idx:
            plt.annotate(w, (emb2d_after[i, 0], emb2d_after[i, 1]))
    plt.title(f"{title_prefix}After Training (PCA 2D)")
    plt.tight_layout()
    plt.show()

def visualize_similarity_heatmap(embedding_matrix_before, embedding_matrix_after, vocab, words, title_prefix=""):
    indices = [vocab.token_to_idx[w] for w in words if w in vocab.token_to_idx]
    if not indices:
        print("No valid words to visualize.")
        return
    emb_before = embedding_matrix_before[indices]
    emb_after = embedding_matrix_after[indices]
    def cosine_sim(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a, b.T)
    sim_before = cosine_sim(emb_before, emb_before)
    sim_after = cosine_sim(emb_after, emb_after)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(sim_before, xticklabels=words, yticklabels=words, cmap='Blues', annot=False)
    plt.title(f"{title_prefix}Similarity Before Training")
    plt.subplot(1, 2, 2)
    sns.heatmap(sim_after, xticklabels=words, yticklabels=words, cmap='Blues', annot=False)
    plt.title(f"{title_prefix}Similarity After Training")
    plt.tight_layout()
    plt.show()

def visualize_analogy_vector(embedding_matrix_before, embedding_matrix_after, vocab, analogy_triplet, title_prefix=""):
    # analogy_triplet: (a, b, c, expected)
    a, b, c, expected = analogy_triplet
    words = [a, b, c, expected]
    indices = [vocab.token_to_idx[w] for w in words if w in vocab.token_to_idx]
    if len(indices) < 4:
        print("Not enough valid words for analogy visualization.")
        return
    emb_before = embedding_matrix_before[indices]
    emb_after = embedding_matrix_after[indices]
    pca = PCA(n_components=2)
    emb2d_before = pca.fit_transform(emb_before)
    emb2d_after = pca.fit_transform(emb_after)
    plt.figure(figsize=(12, 5))
    for i, (emb2d, title) in enumerate(zip([emb2d_before, emb2d_after], ["Before Training", "After Training"])):
        plt.subplot(1, 2, i+1)
        plt.scatter(emb2d[:, 0], emb2d[:, 1], c='red')
        for j, w in enumerate(words):
            plt.annotate(w, (emb2d[j, 0], emb2d[j, 1]))
        # Vẽ vector analogy: b - a + c -> expected
        plt.arrow(emb2d[0, 0], emb2d[0, 1], emb2d[1, 0] - emb2d[0, 0], emb2d[1, 1] - emb2d[0, 1], color='gray', head_width=0.1)
        plt.arrow(emb2d[2, 0], emb2d[2, 1], emb2d[3, 0] - emb2d[2, 0], emb2d[3, 1] - emb2d[2, 1], color='blue', head_width=0.1)
        plt.title(f"{title_prefix}{title} (Analogy)")
    plt.tight_layout()
    plt.show()
