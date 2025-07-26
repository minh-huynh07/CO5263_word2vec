
import time
import csv
import os
from word2vec.dataset import read_corpus, Vocab, get_skip_gram_pairs
from word2vec.train import train_skipgram
from word2vec.evaluate import get_embedding_matrix, find_similar, analogy
import config
from word2vec.evaluate import visualize_embedding_pca_2d, visualize_similarity_heatmap, visualize_analogy_vector

def main():
    # === Load and preprocess data ===
    tokens = read_corpus(config.data_path)
    vocab = Vocab(tokens, min_freq=1)
    print(f"Vocab size: {len(vocab)}")

    # === Create training data ===
    model_type = getattr(config, 'model_type', 'skipgram')
    if model_type == 'cbow':
        from word2vec.dataset import get_cbow_pairs
        pairs = get_cbow_pairs(tokens, vocab, max_window=config.max_window_size)
        print(f"Generated {len(pairs)} CBOW pairs")
    else:
        pairs = get_skip_gram_pairs(tokens, vocab, max_window=config.max_window_size)
        print(f"Generated {len(pairs)} skip-gram pairs")

    # === Train model ===
    train_method = getattr(config, 'train_method', 'softmax')  # 'softmax', 'dot', 'neg_sampling', hoặc 'hierarchical_softmax'
    start = time.time()
    if model_type == 'cbow':
        # CBOW training
        if train_method == 'softmax':
            from word2vec.train import train_cbow_softmax
            model, losses = train_cbow_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate
            )
        elif train_method == 'neg_sampling':
            from word2vec.train import train_cbow_neg_sampling
            model, losses = train_cbow_neg_sampling(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate
            )
        elif train_method == 'hierarchical_softmax':
            from word2vec.train import train_cbow_hierarchical_softmax
            model, losses = train_cbow_hierarchical_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate,
                token_freqs=vocab.token_freqs
            )
        else:
            print(f"Unknown train_method for CBOW: {train_method}")
            return
    else:
        # Skip-gram training (code cũ)
        if train_method == 'softmax':
            from word2vec.train import train_skipgram_softmax
            model, losses = train_skipgram_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate
            )
        elif train_method == 'neg_sampling':
            from word2vec.train import train_skipgram_neg_sampling
            model, losses = train_skipgram_neg_sampling(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate
            )
        elif train_method == 'hierarchical_softmax':
            from word2vec.train import train_skipgram_hierarchical_softmax
            model, losses = train_skipgram_hierarchical_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate,
                token_freqs=vocab.token_freqs
            )
        else:
            from word2vec.train import train_skipgram
            model, losses = train_skipgram(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=config.epochs,
                lr=config.learning_rate
            )
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")

    def log_train_result(result_dict, filename=None):
        if filename is None:
            if model_type == 'cbow':
                filename = 'train_log_cbow.csv'
            else:
                filename = 'train_log_skipgram.csv'
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_dict)

    result = {
        'model_type': model_type,
        'train_method': train_method,
        'embedding_dim': config.embedding_dim,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'min_freq': getattr(config, 'min_freq', 1),
        'vocab_size': len(vocab),
        'num_pairs': len(pairs),
        'train_time': end - start,
        'loss_per_epoch': str(losses),
        'final_loss': losses[-1] if losses else None,
    }
    log_train_result(result)

    # === Evaluate (analogy & similar) ===
    embedding_matrix = get_embedding_matrix(model)

    def show(word_idx_list):
        return [vocab.to_tokens([i])[0] for i in word_idx_list]

    # Example: find top 5 similar for multiple words
    for word in ["alice", "rabbit", "queen", "hatter", "cat"]:
        if word in vocab.token_to_idx:
            idx = vocab[word]
            print(f"Top similar to '{word}':", show(find_similar(idx, embedding_matrix, top_k=5)))

    # === Evaluate similarities before and after training ===
    words_to_check = ["king", "queen", "dog", "cat", "teacher", "student", "car", "apple", "red", "blue"]
    # Khởi tạo model giống như khi train để lấy embedding trước train
    if model_type == 'cbow':
        # CBOW initialization
        if train_method == 'softmax':
            from word2vec.train import train_cbow_softmax
            model_init, _ = train_cbow_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,  # không train
                lr=config.learning_rate
            )
        elif train_method == 'neg_sampling':
            from word2vec.train import train_cbow_neg_sampling
            model_init, _ = train_cbow_neg_sampling(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,
                lr=config.learning_rate
            )
        elif train_method == 'hierarchical_softmax':
            from word2vec.train import train_cbow_hierarchical_softmax
            model_init, _ = train_cbow_hierarchical_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,
                lr=config.learning_rate,
                token_freqs=vocab.token_freqs
            )
        else:
            print(f"Unknown train_method for CBOW: {train_method}")
            return
    else:
        # Skip-gram initialization (code cũ)
        if train_method == 'softmax':
            from word2vec.train import train_skipgram_softmax
            model_init, _ = train_skipgram_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,  # không train
                lr=config.learning_rate
            )
        elif train_method == 'neg_sampling':
            from word2vec.train import train_skipgram_neg_sampling
            model_init, _ = train_skipgram_neg_sampling(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,
                lr=config.learning_rate
            )
        elif train_method == 'hierarchical_softmax':
            from word2vec.train import train_skipgram_hierarchical_softmax
            model_init, _ = train_skipgram_hierarchical_softmax(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,
                lr=config.learning_rate,
                token_freqs=vocab.token_freqs
            )
        else:
            from word2vec.train import train_skipgram
            model_init, _ = train_skipgram(
                training_pairs=pairs,
                vocab_size=len(vocab),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                epochs=0,
                lr=config.learning_rate
            )
    embedding_matrix_init = model_init.get_input_embedding().cpu().detach().numpy()
    embedding_matrix_trained = model.get_input_embedding().cpu().detach().numpy()
    print(f"\n=== Similarities before and after training ({model_type.upper()}) ===")
    for word in words_to_check:
        if word in vocab.token_to_idx:
            idx = vocab[word]
            print(f"\nWord: '{word}'")
            print("  Before train:", show(find_similar(idx, embedding_matrix_init, top_k=5)))
            print("  After train:", show(find_similar(idx, embedding_matrix_trained, top_k=5)))

    # Helper to test analogies
    def test_analogy(a: str, b: str, c: str, expected: str = None):
        try:
            result = analogy(
                a_idx=vocab[a],
                b_idx=vocab[b],
                c_idx=vocab[c],
                embedding_matrix=embedding_matrix,
                top_k=1
            )
            predicted = show(result)[0]
            explanation = f"{b} - {a} + {c} ≈ {predicted}"
            if expected:
                explanation += f" (expected: {expected})"
            print("Analogy:", explanation)
        except KeyError as e:
            print(f"Missing word in vocab: {e}")

    # Run multiple analogy tests (cases relevant to Alice in Wonderland)
    test_analogy("she", "alice", "he")
    test_analogy("he", "rabbit", "she")
    test_analogy("she", "queen", "he")
    test_analogy("he", "hatter", "she")
    test_analogy("she", "cat", "he")
    test_analogy("down", "rabbit", "hole")
    test_analogy("tea", "hatter", "garden")

    # Chọn các từ tiêu biểu cho visualization
    viz_words = [
        "father", "mother", "brother", "sister", "dog", "cat", "lion", "elephant", "teacher", "student", "doctor", "nurse", "apple", "banana", "red", "blue", "green", "car", "bus", "train", "king", "queen", "prince", "princess"
    ]
    # 1. Scatter plot PCA 2D
    visualize_embedding_pca_2d(embedding_matrix_init, embedding_matrix_trained, vocab, viz_words, title_prefix="PCA 2D: ")
    # 2. Heatmap similarity
    visualize_similarity_heatmap(embedding_matrix_init, embedding_matrix_trained, vocab, viz_words[:12], title_prefix="Heatmap: ")
    # 3. Analogy vector (ví dụ: king - man + woman ≈ queen, hoặc father - man + woman ≈ mother)
    analogy_triplet = ("father", "mother", "brother", "sister")  # hoặc đổi thành ("king", "queen", "prince", "princess")
    visualize_analogy_vector(embedding_matrix_init, embedding_matrix_trained, vocab, analogy_triplet, title_prefix="Analogy: ")

if __name__ == "__main__":
    main()
