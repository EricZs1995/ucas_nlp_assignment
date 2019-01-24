import os
from model import *


def train():
    config = Config()
    train_data, dictionary, config = predata(config, config.train_words_path, config.train_tags_path, IsTrain=True)
    test_data, _dictionary, _config = predata(config, config.test_words_path, config.test_tags_path, dictionary,
                                             IsTrain=False)
    # sentences, sentence_lengths = read_data(config.train_words_path)
    # # print(words)
    # # print(words_lens)
    # tags, tags_lens = read_data(config.train_tags_path)
    #
    # dictionary = build_dicts(sentences)
    # words2ids, ids2words = word_mapping(dictionary)
    #
    # train_words = get_processing_word(sentences, words2ids)
    # train_labels = get_processing_label(tags)
    #
    # train_data = predataset(train_words, train_labels, sentence_lengths)
    #
    # config.num_sentences = len(sentences)
    # config.num_words = len(dictionary)

    model = NerModel(config)
    model.build()
    print('lstm_dim',config.lstm_dim)
    print('word_dim',config.word_dim)
    print('batch_size',config.batch_size)
    model.train(train_data, test_data)


def test():
    config = Config()
    train_data, dictionary, config = predata(config, config.train_words_path, config.train_tags_path, IsTrain=True)

    model = NerModel(config)
    test_data, _dictionary, config = predata(config, config.test_words_path, config.test_tags_path, dictionary,
                                             IsTrain=False)
    #model.config.num_sentences_test = config.num_sentences_test 
    model.build()
    model.evaluate(test_data)


if __name__ == "__main__":
    train()
    #test()


