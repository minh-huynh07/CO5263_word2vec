
# Configuration for Word2Vec project

embedding_dim = 100
batch_size = 128
epochs = 20
learning_rate = 0.01
max_window_size = 2

data_path = "data/toy_corpus.txt"

train_method = 'neg_sampling'  # 'softmax', 'dot' (BCE), 'neg_sampling', hoặc 'hierarchical_softmax'
