import os
import time
import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, config):
        self.config = config
        self.source_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='encode_input')
        self.source_length = tf.placeholder(tf.int32, [None], name='ec_length')
        self.target_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='decode_input')
        self.target_length = tf.placeholder(tf.int32, [None], name='target_length')
        self.target_seqs_label = tf.placeholder(tf.int32, [None, self.config.seq_length], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(3, dtype=tf.float32, trainable=False, name="global_loss")

        self.seq2seq()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def seq2seq(self):
        source_embedding = tf.get_variable('source_emb', [self.config.source_vocab_size, self.config.embedding_dim])
        target_embedding = tf.get_variable('target_emb', [self.config.target_vocab_size, self.config.embedding_dim])
        embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
        self.source_embedding = tf.concat([source_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
        self.target_embedding = tf.concat([target_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值

        embed_source_seqs = tf.nn.embedding_lookup(self.source_embedding, self.source_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
        embed_target_seqs = tf.nn.embedding_lookup(self.target_embedding, self.target_seqs)

        # 在词嵌入上进行dropout
        embed_source_seqs = tf.nn.dropout(embed_source_seqs, keep_prob=self.keep_prob)
        embed_target_seqs = tf.nn.dropout(embed_target_seqs, keep_prob=self.keep_prob)

        with tf.variable_scope("encoder"):
            # 定义rnn网络
            def get_en_cell(hidden_dim):
                # 创建单个lstm
                enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                return enc_base_cell
            self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([get_en_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            enc_output, self.enc_state= tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                            inputs=embed_source_seqs,
                                                            sequence_length=self.source_length,
                                                            # initial_state=self.initial_state1,  # 可有可无，自动为0状态
                                                            time_major=False,
                                                            dtype=tf.float32)
        with tf.variable_scope("decoder"):
            # 定义rnn网络
            def get_de_cell(hidden_dim):
                # 创建单个lstm
                dec_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                return dec_base_cell

            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_de_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            dec_output, self.dec_state = tf.nn.dynamic_rnn(self.dec_cell,
                                                           inputs=embed_target_seqs,
                                                           sequence_length=self.target_length,
                                                           initial_state=self.enc_state,  # 编码层的输出来初始化解码层的隐层状态
                                                           time_major=False,
                                                           dtype=tf.float32)

        # with tf.variable_scope("attention"):
        #     attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        #         self.config.hidden_dim, enc_output, memory_sequence_length=self.en_length)
        #     attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        #         self.dec_cell, attention_mechanism, attention_layer_size=self.config.hidden_dim)
        #     dec_output, self.dec_state = tf.nn.dynamic_rnn(
        #         attention_cell, inputs=embed_target_seqs, sequence_length=self.zh_length,
        #         time_major=False, dtype=tf.float32)

        with tf.name_scope("sorfmax_weights"):
            if self.config.share_emb_and_softmax:
                self.softmax_weight = tf.transpose(self.target_embedding)
            else:
                self.softmax_weight = tf.get_variable(
                    "weight", [self.config.hidden_dim, self.config.target_vocab_size + 1])
            self.softmax_bias = tf.get_variable("bias", [self.config.target_vocab_size + 1])

        with tf.name_scope("loss"):
            out_put = tf.reshape(dec_output, [-1, self.config.hidden_dim])
            logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.target_seqs_label,[-1]), logits=logits)
            label_weights = tf.sequence_mask(self.target_length, maxlen=tf.shape(self.target_seqs_label)[1], dtype=tf.float32)
            label_weights = tf.reshape(label_weights, [-1])
            self.mean_loss = tf.reduce_mean(loss*label_weights)

        with tf.name_scope("pres"):
            self.output_id = tf.argmax(logits, axis=1, output_type=tf.int32)[0]

        with tf.name_scope("optimize"):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)

    def train(self, batch_train_g, model_path):
        with self.session as sess:
            for batch_source, batch_source_len, batch_target, batch_target_len, batch_target_label in batch_train_g:
                start = time.time()
                feed = {self.source_seqs: batch_source,
                        self.source_length: batch_source_len,
                        self.target_seqs: batch_target,
                        self.target_length: batch_target_len,
                        self.target_seqs_label: batch_target_label,
                        self.keep_prob: self.config.train_keep_prob}
                _, mean_loss = sess.run([self.optim, self.mean_loss ], feed_dict=feed)
                end = time.time()

                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {}... '.format(mean_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    self.saver.save(sess, os.path.join(model_path, 'model-epoch'), global_step=self.global_step.eval()//self.config.save_every_n)
                if self.global_step.eval() >= self.config.max_steps:
                    break

    def test(self, test_g, model_path, zt):
        batch_source, batch_source_len = test_g
        feed = {self.source_seqs: batch_source,
                self.source_length: batch_source_len,
                self.keep_prob:1.0}
        enc_state = self.session.run(self.enc_state, feed_dict=feed)

        output_ids = []
        dec_state = enc_state
        dec_input, dec_len = zt.text_to_arr(['<s>',])  # decoder层初始输入
        dec_input = np.array([dec_input[:-1], ])
        dec_len = np.array([dec_len, ])
        for i in range(self.config.seq_length):  # 最多输出50长度，防止极端情况下死循环
            feed = {self.enc_state: dec_state,
                    self.target_seqs: dec_input,
                    self.target_length: dec_len,
                    self.keep_prob: 1.0}
            dec_state, output_id= self.session.run([self.dec_state, self.output_id], feed_dict=feed)

            char = zt.int_to_word(output_id)
            if char == '</s>':
                break
            output_ids.append(output_id)

            arr = [output_id]+[len(zt.vocab)] * (self.config.seq_length - 1)
            dec_input = np.array([arr, ])
        return output_ids


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
