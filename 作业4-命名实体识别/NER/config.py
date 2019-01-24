class Config(object):
    num_batches = 0
    num_sentences = 0
    num_sentences_test = 0
    num_words = 0
    num_tags = 7
    score = 0.0
    corre = 0.0

    vocab_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    tag2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}

    model_path = 'data/'
    meta_path = 'data/result/-935.meta'
    summary_path = 'data/result/summary/'
    printloss_path = 'data/result/print/'

    word2vec_path = 'data/wiki_100.utf8'

    train_words_path = 'data/source.txt'
    train_tags_path = 'data/target.txt'

    test_words_path = 'data/test.txt'
    test_tags_path = 'data/test_tgt.txt'

    # training
    train_embeddings = False
    pre_embedding = None

    nepochs = 100
    dropout = 0.3
    batch_size = 64
    optimizer = 'rmsprop'
    lr = 0.001
    lr_decay = 0.95
    clip = 5 

    lstm_dim = 250
    word_dim = 300
