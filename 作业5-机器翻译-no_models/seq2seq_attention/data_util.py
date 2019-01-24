# -*- coding: utf-8 -*-
import os,re
import numpy as np
import pickle
import random


def batch_generator( en_arrs, zh_arrs, batchsize):
    assert len(en_arrs) == len(zh_arrs), 'error: incorrect length english&chinese samples'
    n = len(en_arrs)
    print('samples number:',n)
    samples = [en_arrs[i] + zh_arrs[i] for i in range(n)]

    while True:
        random.shuffle(samples)  
        for i in range(0, n, batchsize):
            batch_samples = samples[i:i + batchsize]
            batch_en = []
            batch_en_len = []
            batch_zh = []
            batch_zh_len = []
            batch_zh_label = []
            for sample in batch_samples:
#                print('sample',sample)
 #               print('sample0',sample[0])
  #              print('sample-1',sample[-1])
   #             print('sample2',sample[2])
    #            print('sample3',sample[3])
                batch_en.append(sample[0])
                batch_en_len.append(sample[1])
                batch_zh.append(sample[2][:-1])
                batch_zh_len.append(sample[3] - 1)
                batch_zh_label.append(sample[2][1:])
            yield np.array(batch_en), np.array(batch_en_len), np.array(batch_zh), np.array(batch_zh_len), np.array(
                batch_zh_label)

class Preprocess():
    def __init__(self):
        pass

    def clears(self,):
        data_en = 'data/en.txt'
        data_zh = 'data/cn.txt'
        data_en_clear = 'data/en_pro.txt'
        data_zh_clear = 'data/zh_pro.txt'
        with open(data_en, 'r',encoding='utf-8') as f_en:
            f_en_w = open(data_en_clear, 'w', encoding='utf-8')
            lineID = 0
            for line in f_en:
                if '<'==line[0] and '>' ==line[-2]:
                    continue
                lineID += 1
                line = re.sub(u"[^0-9a-zA-Z.,?!']+",' ',line)  
                line = re.sub(u"[.,?!]+", lambda x:' '+x.group(0),line)  
                line = line.lower()  
                f_en_w.write(line+'\n')
            print('english lines number:',lineID)
            f_en_w.close()

        with open(data_zh, 'r',encoding='utf-8') as f_zh:
            f_zh_w = open(data_zh_clear, 'w', encoding='utf-8')
            lineID = 0
            for line in f_zh:
                if '<' == line[0] and '>' == line[-2]:
                    continue
                lineID += 1
                line = re.sub(u"[^0-9\u4e00-\u9fa5。，？！']+",'',line)  # 清除不需要的字符
                line = '<s> '+' '.join(line) +' </s>' 
                f_zh_w.write(line + '\n')
            print('chinese lines number:',lineID)
            f_zh_w.close()

    def get_text(self):
        data_en_clear = 'data/en_pro.txt'
        data_zh_clear = 'data/zh_pro.txt'
        en_list = []  
        zh_list = [] 
        with open(data_en_clear, 'r', encoding='utf-8') as f_en:
            for line in f_en:
                en_list += line.split()

        with open(data_zh_clear, 'r', encoding='utf-8') as f_zh:
            for line in f_zh:
                zh_list += line.split()

        return en_list,zh_list

    def get_seq(self):
        data_en_clear = 'data/en_pro.txt'
        data_zh_clear = 'data/zh_pro.txt'
        en_list = []  
        zh_list = []  
        with open(data_en_clear, 'r', encoding='utf-8') as f_en:
            for line in f_en:
                en_list.append(line)

        with open(data_zh_clear, 'r', encoding='utf-8') as f_zh:
            for line in f_zh:
                zh_list.append(line)

        return en_list,zh_list

class TextConverter(object):
    def __init__(self, text=None, save_dir=None, max_vocab=5000 , seq_length = 20):
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print('�~W符�~U��~G~O�~Z%s ' % len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab
            with open(save_dir, 'wb') as f:
                pickle.dump(self.vocab, f)

        self.seq_length = seq_length  
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length:
            arr += [last_num] * (self.seq_length - query_len)
        else:
            arr = arr[:self.seq_length]
            query_len = self.seq_length
        if query_len == 0:
            query_len = 1
        return np.array(arr), np.array(query_len)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return " ".join(words)

    def get_en_arrs(self, file_path):
        arrs_list = []  #
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                arr, arr_len = self.text_to_arr(line)
                arrs_list.append([arr, arr_len])
        return arrs_list




if __name__ == '__main__':
    pass

    pre =  Preprocess()
    pre.clears()
    a, b = pre.get_text()

    et = TextConverter(text=a,save_dir='data/en_vocab.pkl', max_vocab=11000, seq_length = 50)
    zt = TextConverter(text=b,save_dir='data/zh_vocab.pkl', max_vocab=2700, seq_length = 50)
    # print(et.vocab)
    # print(zt.vocab)
    print(et.vocab_size)
    print(zt.vocab_size)
