
# Configuration for Word2Vec project

embedding_dim = 100
batch_size = 128
epochs = 100
learning_rate = 0.01
max_window_size = 2

data_path = "data/toy_corpus.txt"

model_type = 'skipgram'  # 'skipgram' or 'cbow'
train_method = 'softmax'  # 'softmax', 'neg_sampling'
