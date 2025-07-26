# Word2Vec Implementation with Multiple Training Methods

**Author:** Huynh Cong Minh

## Overview

This project implements Word2Vec models with multiple training methods, including Skip-gram and CBOW architectures, along with three different training approaches: Softmax, Negative Sampling, and Hierarchical Softmax. The implementation follows the concepts from the d2l.ai textbook and provides comprehensive evaluation and comparison tools.

## Project Structure

```
word2vec_assignment/
├── config.py                 # Global configuration parameters
├── main.py                   # Main orchestration script
├── compare_word2vec.py       # Comparison and visualization script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── sample.txt           # Main corpus file (toy corpus for visualization)
│   ├── alice.txt            # Alice in Wonderland corpus (backup)
│   └── toy_corpus.txt       # Custom small corpus for better visualization
├── word2vec/
│   ├── __init__.py
│   ├── dataset.py           # Data loading and pair generation
│   ├── model.py             # Model architectures (Skip-gram, CBOW)
│   ├── train.py             # Training functions for all methods
│   ├── evaluate.py          # Evaluation and visualization functions
│   └── utils.py             # Utility functions (Huffman tree)
├── train_log_skipgram.csv   # Training logs for Skip-gram models
├── train_log_cbow.csv       # Training logs for CBOW models
└── word2vec_comparison.png  # Generated comparison plots
```

## Components

### 1. Configuration (`config.py`)
- **embedding_dim**: Dimension of word embeddings (default: 100)
- **batch_size**: Training batch size (default: 128)
- **epochs**: Number of training epochs (default: 100)
- **learning_rate**: Learning rate for optimization (default: 0.01)
- **max_window_size**: Maximum context window size (default: 2)
- **data_path**: Path to the corpus file
- **model_type**: Choose between 'skipgram' or 'cbow'
- **train_method**: Choose between 'softmax', 'neg_sampling', or 'hierarchical_softmax'

### 2. Data Processing (`word2vec/dataset.py`)
- **read_corpus()**: Reads and tokenizes text corpus
- **Vocab class**: Builds vocabulary from tokens with frequency filtering
- **get_skip_gram_pairs()**: Generates (center_word, context_word) pairs for Skip-gram
- **get_cbow_pairs()**: Generates (context_words, center_word) pairs for CBOW
- **SkipGramDataset**: PyTorch dataset for Skip-gram training
- **CBOWDataset**: PyTorch dataset for CBOW training with padding

### 3. Model Architectures (`word2vec/model.py`)
- **SkipGramModel**: Skip-gram architecture with input/output embeddings
- **CBOWModel**: CBOW architecture with context/output embeddings
- Both models support different forward modes for various training methods

### 4. Training Methods (`word2vec/train.py`)

#### Skip-gram Training Functions:
- **train_skipgram_softmax()**: Full softmax training with CrossEntropyLoss
- **train_skipgram_neg_sampling()**: Negative sampling with BCEWithLogitsLoss
- **train_skipgram_hierarchical_softmax()**: Hierarchical softmax with Huffman tree

#### CBOW Training Functions:
- **train_cbow_softmax()**: Full softmax training for CBOW
- **train_cbow_neg_sampling()**: Negative sampling for CBOW
- **train_cbow_hierarchical_softmax()**: Hierarchical softmax for CBOW

### 5. Evaluation (`word2vec/evaluate.py`)
- **find_similar()**: Find most similar words using cosine similarity
- **analogy()**: Solve word analogies (e.g., king - man + woman ≈ queen)
- **visualize_embedding_pca_2d()**: 2D PCA visualization of embeddings
- **visualize_similarity_heatmap()**: Heatmap of word similarities
- **visualize_analogy_vector()**: Visualization of analogy vectors

### 6. Utilities (`word2vec/utils.py`)
- **HuffmanNode**: Node class for Huffman tree construction
- **build_huffman_tree()**: Builds Huffman tree for hierarchical softmax

## Installation and Setup

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies
```
torch
numpy
matplotlib
seaborn
scikit-learn
pandas
```

## Usage

### 1. Basic Training

#### Configure the model in `config.py`:
```python
# Choose model architecture
model_type = 'skipgram'  # or 'cbow'

# Choose training method
train_method = 'softmax'  # 'softmax', 'neg_sampling', or 'hierarchical_softmax'

# Other parameters
embedding_dim = 100
batch_size = 128
epochs = 100
learning_rate = 0.01
```

#### Run training:
```bash
python3 main.py
```

### 2. Training All Methods

To train all methods for comparison, run the following sequence:

#### Skip-gram with Softmax:
```python
# In config.py
model_type = 'skipgram'
train_method = 'softmax'
```
```bash
python3 main.py
```

#### Skip-gram with Negative Sampling:
```python
# In config.py
model_type = 'skipgram'
train_method = 'neg_sampling'
```
```bash
python3 main.py
```

#### Skip-gram with Hierarchical Softmax:
```python
# In config.py
model_type = 'skipgram'
train_method = 'hierarchical_softmax'
```
```bash
python3 main.py
```

#### CBOW with Softmax:
```python
# In config.py
model_type = 'cbow'
train_method = 'softmax'
```
```bash
python3 main.py
```

#### CBOW with Negative Sampling:
```python
# In config.py
model_type = 'cbow'
train_method = 'neg_sampling'
```
```bash
python3 main.py
```

#### CBOW with Hierarchical Softmax:
```python
# In config.py
model_type = 'cbow'
train_method = 'hierarchical_softmax'
```
```bash
python3 main.py
```

## Evaluation

### 1. Word Similarity
The model automatically evaluates word similarities before and after training:
- Finds most similar words for selected vocabulary items
- Uses cosine similarity for comparison
- Shows improvement in semantic relationships

### 2. Word Analogies
Tests word analogies like:
- father - mother + brother ≈ sister
- dog - cat + lion ≈ tiger
- teacher - student + doctor ≈ patient

### 3. Visualizations
Three types of visualizations are generated:
- **PCA 2D Scatter Plot**: Shows word clusters in 2D space
- **Similarity Heatmap**: Displays cosine similarities between words
- **Analogy Vector**: Visualizes analogy relationships

## Comparison and Analysis

### 1. Running Comparison
```bash
python3 compare_word2vec.py
```

### 2. What the Comparison Does
- Loads training logs from `train_log_skipgram.csv` and `train_log_cbow.csv`
- Identifies common methods between both files
- Generates comprehensive comparison table
- Creates 6 visualization plots:
  1. Training Time Comparison
  2. Final Loss Comparison
  3. Training Data Size
  4. Skip-gram Loss Curves
  5. CBOW Loss Curves
  6. Performance vs Speed Scatter Plot

### 3. Analysis Output
The comparison script provides:
- **Best Performance**: Method with lowest final loss
- **Fastest Training**: Method with shortest training time
- **Method Comparison**: CBOW vs Skip-gram for each training method
- **Model Comparison**: Best and fastest methods for each architecture
- **Recommendations**: Overall best choices based on different criteria

## Training Methods Explained

### 1. Softmax
- **Concept**: Full softmax over entire vocabulary
- **Loss Function**: CrossEntropyLoss
- **Pros**: Theoretically optimal
- **Cons**: Computationally expensive for large vocabularies

### 2. Negative Sampling
- **Concept**: Approximates softmax by sampling negative examples
- **Loss Function**: BCEWithLogitsLoss
- **Pros**: Much faster than full softmax
- **Cons**: Approximation may reduce quality slightly

### 3. Hierarchical Softmax
- **Concept**: Uses Huffman tree to approximate softmax
- **Loss Function**: BCEWithLogitsLoss on tree paths
- **Pros**: Efficient for large vocabularies
- **Cons**: More complex implementation, slower than negative sampling

## Model Architectures

### 1. Skip-gram
- **Input**: Center word
- **Output**: Context words
- **Training**: Predicts context words given center word
- **Use Case**: Better for smaller datasets, rare words

### 2. CBOW (Continuous Bag of Words)
- **Input**: Context words
- **Output**: Center word
- **Training**: Predicts center word given context words
- **Use Case**: Better for larger datasets, faster training

## Corpus Files

### 1. Toy Corpus (`data/toy_corpus.txt`)
- **Size**: ~1000 words
- **Content**: Custom-generated text with clear thematic groups
- **Purpose**: Better visualization and understanding of word clusters
- **Themes**: Family, animals, jobs, colors, vehicles, royalty

### 2. Alice in Wonderland (`data/alice.txt`)
- **Size**: ~30,000 words
- **Content**: Full text from Project Gutenberg
- **Purpose**: Larger corpus for better model training
- **Source**: Project Gutenberg

## Output Files

### 1. Training Logs
- **train_log_skipgram.csv**: Training results for all Skip-gram methods
- **train_log_cbow.csv**: Training results for all CBOW methods
- **Columns**: model_type, train_method, embedding_dim, batch_size, epochs, learning_rate, min_freq, vocab_size, num_pairs, train_time, loss_per_epoch, final_loss

### 2. Visualizations
- **word2vec_comparison.png**: Comprehensive comparison plots
- **PCA plots**: Before/after training embeddings
- **Heatmaps**: Word similarity matrices
- **Analogy plots**: Vector relationship visualizations

## Performance Insights

Based on typical results:
- **CBOW** generally outperforms **Skip-gram** in both speed and accuracy
- **Negative Sampling** provides the best balance of speed and performance
- **Hierarchical Softmax** is slower but can be more memory efficient
- **Softmax** is the slowest but theoretically most accurate

## Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Ensure virtual environment is activated
2. **CUDA errors**: Models automatically use CPU if CUDA unavailable
3. **Memory issues**: Reduce batch_size or embedding_dim
4. **Training time**: Hierarchical softmax is significantly slower

### Performance Tips:
1. Use smaller corpus for quick testing
2. Reduce epochs for faster iteration
3. Use negative sampling for best speed/performance balance
4. CBOW is generally faster than Skip-gram

## Future Enhancements

Potential improvements:
1. Add more evaluation metrics (WordSim, SimLex)
2. Implement subword embeddings (FastText)
3. Add support for pre-trained embeddings
4. Implement distributed training
5. Add more visualization options
6. Support for different languages

## References

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [d2l.ai Word2Vec Chapter](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Project Gutenberg](https://www.gutenberg.org/) 