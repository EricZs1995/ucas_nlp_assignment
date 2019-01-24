# -*- coding: utf-8 -*-

import tensorflow as tf

import os
import codecs
from data_util import *
import tqdm
from config import Config
from transformer import *


class Model:
    def __init__(self, istraining=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.pre_data(istraining) # 设置placeholder，设置模型输入
            self.encoder(istraining)  # encoder层
            self.decoder(istraining)  # decoder层
            self.pred(istraining)     # 训练

    def pre_data(self, istraining = True):
        if istraining:
            # 获取训练数据集、词典大小
            self.ch, self.en, self.num_batch, self.ch_vocab_size, self.en_vocab_size = get_batch_data()
        else:
            # 设置测试集placeholder
            self.ch = tf.placeholder(tf.int32, shape=[None, Config.max_length])
            self.en = tf.placeholder(tf.int32, shape=[None, Config.max_length])
            # 获取词典大小
            self.num_batch, self.ch_vocab_size, self.en_vocab_size = get_len()
        # 设置decoder第一个block的第一个输入数据，将target sequence由'.... </s>'变换为'<s>....'
        self.decoder_inputs = tf.concat((tf.ones_like(self.en[:, :1]) * 2, self.en[:, :-1]), -1)

    def encoder(self, istraining=True):
        with tf.variable_scope('encoder'):
            # 获取源语言的输入embedding
            self.encoder_input_embedding = EmbeddingsWithSoftmax(self.ch, self.ch_vocab_size, zero_pad=True,
                                                                 scope='encoder_emb')
            # 位置编码也使用embedding
            self.encoder_input_embedding += EmbeddingsWithSoftmax(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.ch)[1]), 0), [tf.shape(self.ch)[0], 1]),
                self.ch_vocab_size, zero_pad=False, scope='encoder_pos')
            # self.encoder_input_embedding += PositionalEncoding(self.ch,scope='enoder_pos')

            self.encoder_input_embedding = tf.layers.dropout(self.encoder_input_embedding,
                                                             rate=Config.dropout_rate,
                                                             training=tf.convert_to_tensor(istraining))
            # 执行模型中encoder部分任务
            self.encoder_outputs = Encoder(self.encoder_input_embedding, istraining)


    def decoder(self, istraining=True):
        with tf.variable_scope('decoder'):
            # 获取decoder的输入embedding
            self.decoder_input_embedding = EmbeddingsWithSoftmax(self.decoder_inputs, self.en_vocab_size,
                                                                 zero_pad=True, scope='decoder_emb')
            self.decoder_input_embedding += EmbeddingsWithSoftmax(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.en)[1]), 0), [tf.shape(self.en)[0], 1]),
                self.en_vocab_size, zero_pad=False, scope='decoder_pos')
            #   self.decoder_input_embedding += PositionalEncoding(self.decoder_inputs,scope='decoder_pos')
            self.decoder_input_embedding = tf.layers.dropout(self.decoder_input_embedding,
                                                             rate=Config.dropout_rate,
                                                             training=tf.convert_to_tensor(istraining))
            # 执行模型中decoder部分任务
            self.decoder_outputs = Decoder(self.decoder_input_embedding, self.encoder_outputs, istraining)


    def pred(self, istraining=True):
        self.logits = tf.layers.dense(self.decoder_outputs, self.en_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.en, 0))
        self.acc = tf.reduce_sum(tf.to_float(
            tf.equal(self.preds, self.en)) * self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)
        # 训练
        if istraining:
            self.en_smoothed = LabelSmoothing(tf.one_hot(self.en, depth=self.en_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.en_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()