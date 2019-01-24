# -*- coding: utf-8 -*-

import tensorflow as tf

import os
import codecs
from data_util import *
import tqdm
from config import config
from transformer import *
#from modules import *
class Graph:
    def __init__(self, istraining=True):
        self.graph = tf.Graph()
#        tf.reset_default_graph()
        with self.graph.as_default():
            #self.ch_vocab_size = ch_vocab_size
            #self.en_vocab_size = en_vocab_size
            if istraining:
                self.ch, self.en, self.num_batch,self.ch_vocab_size,self.en_vocab_size= get_batch_data()
            else:
                self.ch = tf.placeholder(tf.int32, shape=[None, config.max_length])
                self.en = tf.placeholder(tf.int32, shape=[None, config.max_length])
                self.num_batch, self.ch_vocab_size, self.en_vocab_size = get_len()
            self.decoder_inputs = tf.concat((tf.ones_like(self.en[:, :1]) * 2, self.en[:, :-1]), -1)
            # todo load_dict

            with tf.variable_scope('encoder'):
                self.encoder_input_embedding = EmbeddingsWithSoftmax(self.ch, self.ch_vocab_size,zero_pad=True,scope='encoder_emb')
                
#                self.encoder_input_embedding = embedding(self.ch, self.ch_vocab_size,num_units=config.dim_model,zero_pad=True,scope='encoder_emb')
                self.encoder_input_embedding += EmbeddingsWithSoftmax(tf.tile(tf.expand_dims(tf.range(tf.shape(self.ch)[1]), 0), [tf.shape(self.ch)[0], 1]),self.ch_vocab_size,zero_pad=False,scope='encoder_pos')
#                self.encoder_input_embedding += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.ch)[1]), 0), [tf.shape(self.ch)[0], 1]),self.ch_vocab_size,num_units=config.dim_model,zero_pad=False,scope='encoder_pos')
#                self.encoder_input_embedding += PositionalEncoding(self.ch,scope='enoder_pos')

                self.encoder_input_embedding = tf.layers.dropout(self.encoder_input_embedding,
                                                                 rate=config.dropout_rate,
                                                                 training=tf.convert_to_tensor(istraining))

                self.encoder_outputs = Encoder(self.encoder_input_embedding,istraining)

            with tf.variable_scope('decoder'):
                self.decoder_input_embedding = EmbeddingsWithSoftmax(self.decoder_inputs, self.en_vocab_size,zero_pad=True,scope='decoder_emb')
#                self.decoder_input_embedding = embedding(self.decoder_inputs, self.en_vocab_size,num_units=config.dim_model,zero_pad=True,scope='decoder_emb')
#                self.decoder_input_embedding += PositionalEncoding(self.decoder_inputs,scope='decoder_pos')
                self.decoder_input_embedding += EmbeddingsWithSoftmax(tf.tile(tf.expand_dims(tf.range(tf.shape(self.en)[1]), 0), [tf.shape(self.en)[0], 1]),self.en_vocab_size,zero_pad=False,scope='decoder_pos')
#                self.decoder_input_embedding += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.en)[1]), 0), [tf.shape(self.en)[0], 1]),self.en_vocab_size,num_units=config.dim_model,zero_pad=False,scope='decoder_pos')
                self.decoder_input_embedding = tf.layers.dropout(self.decoder_input_embedding,
                                                                 rate=config.dropout_rate,
                                                                 training=tf.convert_to_tensor(istraining))

                self.decoder_outputs = Decoder(self.decoder_input_embedding, self.encoder_outputs, istraining)

            self.logits = tf.layers.dense(self.decoder_outputs, self.en_vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.en, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.en)) * self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if istraining:
                self.en_smoothed = LabelSmoothing(tf.one_hot(self.en, depth=self.en_vocab_size))
                #self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.en_smoothed)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.en_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


