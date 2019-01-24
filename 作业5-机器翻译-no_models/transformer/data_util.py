# -*- coding: utf-8 -*-
import numpy as np
import regex
import tensorflow as tf
import codecs
import collections
from config import Config
import re

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


def load_data():
    ch_sentences =[re.sub(u"[^0-9\u4e00-\u9fa5。，？！']+",' ',line)
                   for line in codecs.open(Config.source_data_file, 'r', 'utf-8').readlines()]
    en_sentences =[re.sub(u"[.,?!]+", lambda x:' '+x.group(0),re.sub(u"[^0-9a-zA-Z.,?!']+", ' ', line)).lower()
                   for line in codecs.open(Config.target_data_file, 'r', 'utf-8').readlines()]
    split_index = int(len(ch_sentences) * Config.split)
    train_ch_sents = ch_sentences[:split_index]
    train_en_sents = en_sentences[:split_index]
    test_ch_sents = ch_sentences[split_index:]
    test_en_sents = en_sentences[split_index:]
    return train_ch_sents, train_en_sents, test_ch_sents, test_en_sents


def build_dictionary(sentences):
    dic = collections.OrderedDict()
    for sentence in sentences:
        for word in sentence.strip().split():
            word = word.strip()
            if word=='' or word =='\u3000':
                continue
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
    vocab = [word for word in dic if word!='' and word !='\u3000' and  dic[word] >= Config.min_count]
    vocab.insert(0, PAD)
    vocab.insert(1, UNK)
    vocab.insert(2, BOS)
    vocab.insert(3, EOS)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    #print(word2idx)
    return word2idx, idx2word


def word_mapping(ch_sents, en_sents, ch_word2idx, en_word2idx):
    ch_idxes, en_idxes, ches, enes = [], [], [], []
    for ch_sent, en_sent in zip(ch_sents, en_sents):
        ch_idx = [ch_word2idx.get(word, ch_word2idx[UNK])
                  for word in (ch_sent.strip() + ' ' + EOS).split() if word!='' and word !='\u3000']
        en_idx = [en_word2idx.get(word, en_word2idx[UNK])
                  for word in (en_sent.strip() + ' ' + EOS).split() if word!='' and word !='\u3000']
        if max(len(ch_idx), len(en_idx)) <= Config.max_length:
            ch_idxes.append(ch_idx)
            en_idxes.append(en_idx)
            ches.append(ch_sent)
            enes.append(en_sent)
    return ch_idxes, en_idxes, ches, enes


def preprocess_data(isTrain=True):
    train_ch_sents, train_en_sents, test_ch_sents, test_en_sents = load_data()
    ch_word2idx, ch_idx2word = build_dictionary(train_ch_sents+test_ch_sents)
    en_word2idx, en_idx2word = build_dictionary(train_en_sents+test_en_sents)
    len_ch_vocab = len(ch_word2idx)
    len_en_vocab = len(en_word2idx)
    if isTrain:
        ch_idxes, en_idxes, ches, enes = word_mapping(train_ch_sents, train_en_sents,
                                                      ch_word2idx, en_word2idx)
    else:
        ch_idxes, en_idxes, ches, enes = word_mapping(test_ch_sents, test_en_sents,
                                                      ch_word2idx, en_word2idx)
    # padding
    ch_idxes = [seq_idx + (Config.max_length - len(seq_idx)) * [ch_word2idx[PAD]]
                for seq_idx in ch_idxes]
    en_idxes = [seq_idx + (Config.max_length - len(seq_idx)) * [en_word2idx[PAD]]
                for seq_idx in en_idxes]
    if isTrain:
        return ch_idxes, en_idxes, ches, enes, len_ch_vocab, len_en_vocab
    else:
        return ch_idxes, en_idx2word, ches, enes, len_ch_vocab, len_en_vocab


def get_batch_data():
    ch_idxes, en_idxes, ches, enes, len_ch_vocab, len_en_vocab = preprocess_data(True)
    num_batch = len(ch_idxes)//Config.batch_size
    ch_idxes = np.array(ch_idxes)
    en_idxes = np.array(en_idxes)
    ch_idxes = tf.convert_to_tensor(ch_idxes, tf.int32)
    en_idxes = tf.convert_to_tensor(en_idxes, tf.int32)
    print(ch_idxes.get_shape())
    input_queues = tf.train.slice_input_producer([ch_idxes, en_idxes])
    ch, en = tf.train.shuffle_batch(input_queues, num_threads=8, batch_size=Config.batch_size, capacity=Config.batch_size*64, min_after_dequeue=Config.batch_size*32, allow_smaller_final_batch=False)
    return ch, en, num_batch, len_ch_vocab, len_en_vocab

def get_len():
    ch_idxes, en_idx2word, ches, enes, len_ch_vocab, len_en_vocab = preprocess_data(False)
    num_batch = len(ch_idxes) // Config.batch_size
    print(num_batch)
    print(len(ch_idxes))
    return num_batch,len_ch_vocab,len_en_vocab

def get_batch(ches, enes):
    num_batch = (len(ches)-1)//Config.batch_size+1
    for i in range(num_batch):
        yield ches[i*Config.batch_size:(i+1)*Config.batch_size], enes[i*Config.batch_size:(i+1)*Config.batch_size]
