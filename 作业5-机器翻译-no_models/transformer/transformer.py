# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

from config import Config


# 词向量编码
def EmbeddingsWithSoftmax(inputs, vocab_size, zero_pad=True, scope='embedding'):
    with tf.variable_scope(scope):
        lookup_table = tf.get_variable('lookup_table',
                                       shape=[vocab_size, Config.dim_model],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, Config.dim_model]), lookup_table[1:, :]), 0)
            embedding = tf.nn.embedding_lookup(lookup_table, inputs)
            embedding = embedding * (Config.dim_model ** 0.5)
        else:
            embedding = tf.nn.embedding_lookup(lookup_table, inputs)
        return embedding

# 位置编码
# PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
# PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
def PositionalEncoding(inputs, scope='pos_embedding'):
    len_batch, len_sent = inputs.get_shape().as_list()
    with tf.variable_scope(scope):
        pos_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
        pos_enc = np.array(
            [[pos / np.power(10000., 2. * (i) / Config.dim_model)
              for i in range(Config.dim_model)]
             for pos in range(len_sent)
             ])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1
        lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)
        embedding = tf.nn.embedding_lookup(lookup_table, pos_ind)
        return embedding


def MultiHeadedAttention(query, key, value, istraining=True, mask=True):
    Q = tf.layers.dense(query, Config.dim_model, activation=tf.nn.relu)
    K = tf.layers.dense(key,   Config.dim_model, activation=tf.nn.relu)
    V = tf.layers.dense(value, Config.dim_model, activation=tf.nn.relu)
    # 切分
    Q = tf.concat(tf.split(Q, Config.num_header, axis=2), axis=0)
    K = tf.concat(tf.split(K, Config.num_header, axis=2), axis=0)
    V = tf.concat(tf.split(V, Config.num_header, axis=2), axis=0)
    outputs = Attention(Q, K, V, mask, istraining)
    outputs = tf.concat(tf.split(outputs, Config.num_header, axis=0), axis=2)
    return outputs


def Attention(Q, K, V, mask=False, istraining=True):
    scores = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    scores = scores / (K.get_shape().as_list()[-1] ** 0.5)
    if mask:
        diag_vals = tf.ones_like(scores[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(scores)[0], 1, 1])
        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        scores = tf.where(tf.equal(masks, 0), paddings, scores)
    p_attn = tf.nn.softmax(scores)
    p_attn = tf.layers.dropout(p_attn, rate=Config.dropout_rate,
                               training=tf.convert_to_tensor(istraining))
    return tf.matmul(p_attn, V)


def LayerNorm(x, sublayer):
    x = tf.add(x, sublayer)
    inputs = x
    epsilon = 1e-8
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
    # outputs =  tf.contrib.layers.layer_norm(x)
    return outputs


def PositionwiseFeedForward(inputs):
    # FFN(x) = max(0, x*W1+b1)*W2 + b2
    outputs = tf.layers.conv1d(inputs, filters=4 * Config.dim_model, kernel_size=1, activation=tf.nn.relu)
    outputs = tf.layers.conv1d(outputs, filters=Config.dim_model, kernel_size=1)
    return outputs


def EncoderLayer(inputs_embedding, istraining=True):
    multi = MultiHeadedAttention(inputs_embedding, inputs_embedding, inputs_embedding, istraining, False)
    multi_layernorm = LayerNorm(inputs_embedding, multi)
    fft = PositionwiseFeedForward(multi_layernorm)
    fft_layernorm = LayerNorm(multi_layernorm, fft)
    return fft_layernorm


def Encoder(inputs_embedding, istraining=True):
    encoder_outputs = inputs_embedding
    for i in range(Config.num_block):
        with tf.variable_scope("encoder_num_blocks_{}".format(i)):
            encoder_outputs = EncoderLayer(encoder_outputs, istraining)
    return encoder_outputs


def DecoderLayer(inputs_embedding, encoder_outputs, istraining=True):
    masked_multi = MultiHeadedAttention(inputs_embedding, inputs_embedding, inputs_embedding, istraining, True)
    masked_multi_layernorm = LayerNorm(inputs_embedding, masked_multi)
    multi = MultiHeadedAttention(masked_multi_layernorm, encoder_outputs, encoder_outputs, istraining, False)
    multi_layernorm = LayerNorm(masked_multi_layernorm, multi)
    fft = PositionwiseFeedForward(multi_layernorm)
    fft_layernorm = LayerNorm(multi_layernorm, fft)
    return fft_layernorm


def Decoder(inputs_embedding, encoder_outputs, istraining=True):
    decoder_outputs = inputs_embedding
    for i in range(Config.num_block):
        with tf.variable_scope("decoder_num_blocks_{}".format(i)):
            decoder_outputs = DecoderLayer(decoder_outputs, encoder_outputs, istraining)
    return decoder_outputs


def LabelSmoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)
