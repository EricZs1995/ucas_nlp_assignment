# encoding=utf8
import sys
import pickle
import os
import re
import random
import numpy as np
import codecs
import io

UNK = "$UNK$"
NONE = "O"

class PreDataset(object):
    def __init__(self, words, tags):
        self.train_words = words
        self.train_tags = tags


def predata(config, words_path, tags_path, dictionary=None, IsTrain = True):
    sentences, sentence_lengths = read_data(words_path)
    # print(words)
    # print(words_lens)
    tags, tags_lens = read_data(tags_path)
    if dictionary is None:
        dictionary = build_dicts(sentences)

    # todo
    # 加入预训练embedding
    if IsTrain :
        if config.train_embeddings:
            vocab, embd, word_dim = loadWord2Vec(config.word2vec_path)
            dictionary, _embedding, word_dim = build_dicts_with_pretrain_emb(sentences, vocab, embd, word_dim)
            embedding = np.asarray(_embedding)

    words2ids, ids2words = word_mapping(dictionary)
    train_words = get_processing_word(sentences, words2ids)
    train_labels = get_processing_label(tags)
    train_data = predataset(train_words, train_labels, sentence_lengths)
    if IsTrain:
        config.num_sentences = len(sentences)
    else:
        config.num_sentences_test= len(sentences)
    if IsTrain:
        config.num_words = len(dictionary)
    if config.train_embeddings:
        config.word_dim = word_dim
        config.pre_embedding = embedding
    return train_data, dictionary, config


def predataset(words, tags, sentence_lengths):
    data = []
    for i in range(len(words)):
        data.append((words[i], tags[i], sentence_lengths[i]))
    return data


def pretrain_embeding(pre_embeding_path):
    assert os.path.isfile(pre_embeding_path)
    pretrained_dictionary = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(pre_embeding_path, 'r', 'utf-8')
        if len(pre_embeding_path) > 0
    ])

#读取预训练词向量
def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = codecs.open(filename, 'r', 'utf-8')
    line = fr.readline().strip()
    # print line
    word_dim = int(line.split(' ')[1])
    vocab.append(UNK)
    embd.append([0] * word_dim)
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print("loaded word2vec")
    fr.close()
    return vocab, embd, word_dim


tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
label2tag = {0: "O",1: "B-PER",2: "I-PER",3: "B-LOC",4: "I-LOC",5: "B-ORG",6: "I-ORG", 7:"ERR"}


def read_data(file_path):
    datas = []
    lengths = []
    data = []
    num = 0
    for line in codecs.open(file_path, 'r', 'utf8'):
        # with io.open(file_path,'r',encoding='utf-8') as fr:
        # 	lines = fr.readlines()
        # for line in lines:
        line = line.strip()
        # print(line)
        num = num + 1
        if not line:
            # if len(sequence) > 0:
            # 	sequences.append(sequence)
            sequence = []
        else:
            data = line.split(' ')
            datas.append(data)
            lengths.append(len(data))
            data = []
    # if num == 100:
    # 	break
    return datas, lengths


def build_dicts(sentences):
    dictionary = set()
    for sentence in sentences:
        for word in sentence:
            dictionary.add(word.lower())
    dictionary.add(UNK)
    return dictionary

#使用预训练词向量构建字典
def build_dicts_with_pretrain_emb(sentences, pre_dictionary, pre_embedding, word_dim):
    dictionary = build_dict(sentences)
    for word in dictionary:
        if word not in pre_dictionary:
            pre_dictionary.append(word)
            pre_embedding.append([0] * word_dim)
    return pre_dictionary, pre_embedding, word_dim

#获取word与id之间映射关系
def word_mapping(dictionary):
    # sorted_items = sorted(dictionary.items(), key=lambda x: (-x[1], x[0]))
    ids2words = {i: v for i, v in enumerate(list(dictionary))}
    words2ids = {v: k for k, v in ids2words.items()}
    return words2ids, ids2words

#word->id映射
def get_processing_word(sentences, words2ids):
    sentences_map = []
    if len(sentences) > 0:
        for sentence in sentences:
            sentence_map = []
            for word in sentence:
                if word in words2ids:
                    sentence_map.append(words2ids[word])
                else:
                    sentence_map.append(words2ids[UNK])
            sentences_map.append(sentence_map)
    return sentences_map

#tag->label映射
def get_processing_label(tagss):
    tags_map = []
    if len(tagss) > 0:
        for tags in tagss:
            tag_map = []
            for tag in tags:
                if tag in tag2label:
                    tag_map.append(tag2label[tag])
                else:
                    tag_map.append(tag2label[UNK])
            tags_map.append(tag_map)
    return tags_map

#label->tag映射
def get_processing_tag(tagss):
    tags_map = []
    if len(tagss) > 0:
        for tags in tagss:
            tag_map = []
            for tag in tags:
                if tag in label2tag:
                    tag_map.append(label2tag[tag])
                else:
                    tag_map.append(label2tag[7])
            tags_map.append(tag_map)
    return tags_map


def batches(data, batch_size):
    x_batch, y_batch, z_batch = [], [], []
    for (x, y, z) in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []

        # if type(x[0]) == tuple:
        #     x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        z_batch.append(z)

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch


def padding_sentence(sentences, lengths, pad_token):
    if sentences is not None:
        max_length = max(lengths)
        sentence_padded = []
        for sent in sentences:
            sent = list(sent)
            sent = sent[:max_length] + [pad_token] * max(max_length - len(sent), 0)
            sentence_padded += [sent]
        return sentence_padded


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    tagtype2id = {"PER":0, "LOC":1, "ORG":2}
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunks_devs = [[],[],[]]
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunks_devs[tagtype2id[chunk_type]].append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunks_devs[tagtype2id[chunk_type]].append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
        chunks_devs[tagtype2id[chunk_type]].append(chunk)

    return chunks, chunks_devs

#评估测试结果
def evalu(correct_preds,total_preds,total_correct):
	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0	
	return p,r,f1
