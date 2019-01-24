import numpy as np
import tensorflow as tf
from data_util import *
from config import *
from tqdm import tqdm


class NerModel(object):
    def __init__(self, config):
        self.config = config

    def add_placeholders(self):
        # [batch_size]
        self.sentence_lengths = tf.placeholder(dtype=tf.int32,
                                               shape=[None],
                                               name="sentence_lengths")
        # [batch_size, max len of sentences]
        self.words2ids = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None],
                                        name="words2ids")
        # [batch_size, max len of sentences]
        self.tags2labels = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="tags2ids")
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32,
                                 shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels, sentence_lengths, dropout = None, lr = None):
        words_padded = padding_sentence(words, sentence_lengths, 0)
        labels_padded = padding_sentence(labels, sentence_lengths, 0)
        feed = {self.sentence_lengths: sentence_lengths,
                self.words2ids: words_padded,
                self.tags2labels: labels_padded}
        if dropout is not None:
            feed[self.dropout] = dropout
        else:
            feed[self.dropout] = 1.0
        if lr is not None:
            feed[self.lr] = lr
        return feed

    def embedding_layer(self):
        with tf.variable_scope("words_embedding"):
            if self.config.pre_embedding is None:
                params = tf.get_variable(name="lookup_params",
                                     dtype=tf.float32,
                                     shape=[self.config.num_words, self.config.word_dim])
            else:
                params = tf.Variable(self.config.pre_embedding,
                                     trainable=True,
                                     name='lookup_params',
                                     dtype=tf.float32)

            word_embeddings = tf.nn.embedding_lookup(params,
                                                     self.words2ids,
                                                     name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def bilstm_layer(self):
        with tf.variable_scope("bi_lstm"):
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_dim)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_dim)
            (output_fw, output_bw), final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                self.word_embeddings,
                sequence_length=self.sentence_lengths,
                dtype=tf.float32)
            lstm_outputs = tf.concat([output_fw, output_bw], axis=-1)
            self.lstm_outputs = tf.nn.dropout(lstm_outputs, self.dropout)

    def predict_layer(self):
        with tf.variable_scope("predict"):
            W = tf.get_variable("weight",
                                dtype=tf.float32,
                                shape=[2 * self.config.lstm_dim, self.config.num_tags])
            b = tf.get_variable("bias",
                                dtype=tf.float32,
                                shape=[self.config.num_tags])
            num_steps = tf.shape(self.lstm_outputs)[1]
            output = tf.reshape(self.lstm_outputs, [-1, 2 * self.config.lstm_dim])
            predict_tags = tf.matmul(output, W) + b
            self.logits = tf.reshape(predict_tags, [-1, num_steps, self.config.num_tags])

    def loss_layer(self):
        with tf.variable_scope("crf_loss"):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits,
                self.tags2labels,
                self.sentence_lengths)
            self.transition_params = transition_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)

    def train_operation(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = self.config.optimizer
            if optimizer == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(self.config.lr)
            elif optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
            elif optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.config.lr)
            elif optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.config.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(optimizer))

            if self.config.clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, global_norm = tf.clip_by_global_norm(grads, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs), global_step=self.global_step)
            else:
                self.train_op = optimizer.minimize(self.loss)

    def init(self):
        self.init = tf.global_variables_initializer()

    def build(self):
        self.add_placeholders()
        self.embedding_layer()
        self.bilstm_layer()
        self.predict_layer()
        self.loss_layer()
        self.train_operation()
        self.init()

    def run_epoch(self, sess, train_data, epoch,  test_data):
        self.config.num_batches = (self.config.num_sentences + self.config.batch_size - 1) // self.config.batch_size
        results = []
        train_loss = None
        print(self.config.num_batches)
        for i, (words, labels, length) in enumerate(
                tqdm(batches(train_data, self.config.batch_size), total=self.config.num_batches)):
            # for i, (words, labels, length) in enumerate(batches(train_data, self.config.batch_size)):

            # print(self.config.num_sentences)
            # print(train_data[0:20])
            # print(int(self.config.num_sentences/self.config.batch_size) +1)
            # for i in range(int(self.config.num_sentences/self.config.batch_size) +1):
            #     print(i*self.config.batch_size)
            #     print((i+1)*self.config.batch_size)
            #     words, labels = train_data[i*self.config.batch_size:(i+1)*self.config.batch_size]

            fd = self.get_feed_dict(words, labels, length, self.config.dropout, self.config.lr)
            # _, train_loss,summary, _step = sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict=fd)
            _, train_loss, _step = sess.run([self.train_op, self.loss, self.global_step],
                                            feed_dict=fd)
            # self.file_writer.add_summary(summary, _step)
            if i % 20 == 0:
                print("loss:{0}", train_loss)
            results.append([epoch, i, train_loss])
            # file = open(self.printfile, 'w')
            # for re in results:
            #     file.write(str(re)+'\n')
            # file.close()
            self.config.lr*=self.config.lr_decay
        res,result_label = self.run_evaluate(sess, test_data)

        if res['f1']>self.config.score or res['p']>self.config.corre :
            self.saver.save(sess, self.config.model_path, global_step=(epoch+1))
            self.config.score = res['f1']
            self.config.corre = res['p']

            import time
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 将指定格式的当前时间以字符串输出
            suffix = ".txt"
            newfile = './data/result/print/label' + t +'-'+str(epoch)+ suffix
            file = open(newfile, 'a')
            result_tags = get_processing_tag(result_label)
            for re in result_tags:
                file.write(str(re)+'\n')
            file.close()
        print(res)
        file1 = open(self.printresultfile, 'a')
        file1.write(str(res)+'\n')
        file1.close()
        print("eqpoch{0}--loss:{2}", epoch, train_loss)
        file = open(self.printfile, 'a')
        for re in results:
            file.write(str(re) + '\n')
        file.close()

    def run_evaluate(self, sess, test_data):
        self.config.num_batches = (self.config.num_sentences_test + self.config.batch_size - 1) // self.config.batch_size
        accs = []
        results = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        correct_per,correct_loc,correct_org = 0., 0., 0.
        total_correct_per,total_correct_loc, total_correct_org = 0., 0., 0.
        total_pred_per,total_pred_loc, total_pred_org = 0., 0., 0.
        for i, (words, labels, lengthes) in enumerate(
                tqdm(batches(test_data, self.config.batch_size), total=self.config.num_batches)):
            fd = self.get_feed_dict(words, labels, lengthes)
            _logits, _transition_params = sess.run([self.logits, self.transition_params], feed_dict=fd)
            viterbi_sequences = []
            # print(_logits)
            for loss, length in zip(_logits, lengthes):
                loss = loss[:length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(loss, _transition_params)
                viterbi_sequences += [viterbi_seq]
            for seq, label, viterbi_seq, length in zip(words,labels, viterbi_sequences, lengthes):
                label = label[:length]
                results.append(label)
                # print(label)
                viterbi_seq = viterbi_seq[:length]
                # print(viterbi_seq)
                accs += [a == b for (a, b) in zip(label, viterbi_seq)]
                label_chunks, label_chunks_devs = get_chunks(label, self.config.tag2id)
                label_chunks = set(label_chunks)

                viterbi_seq_chunks, viterbi_seq_chunks_devs = get_chunks(viterbi_seq, self.config.tag2id)
                viterbi_seq_chunks = set(viterbi_seq_chunks)

                correct_per += len(set(label_chunks_devs[0])&set(viterbi_seq_chunks_devs[0]))
                correct_loc += len(set(label_chunks_devs[1])&set(viterbi_seq_chunks_devs[1]))
                correct_org += len(set(label_chunks_devs[2])&set(viterbi_seq_chunks_devs[2]))

                total_pred_per += len(set(viterbi_seq_chunks_devs[0]))
                total_pred_loc += len(set(viterbi_seq_chunks_devs[1]))
                total_pred_org += len(set(viterbi_seq_chunks_devs[2]))

                total_correct_per += len(set(label_chunks_devs[0]))
                total_correct_loc += len(set(label_chunks_devs[1]))
                total_correct_org += len(set(label_chunks_devs[2]))

                correct_preds += len(label_chunks & viterbi_seq_chunks)
                total_preds += len(viterbi_seq_chunks)
                total_correct += len(label_chunks)

                # print(seq)
                # print(label)
                #print(label_chunks)
                #print(viterbi_seq_chunks)

        p,r,f1 = evalu(correct_preds,total_preds,total_correct)
        p_per,r_per,f1_per = evalu(correct_per,total_pred_per,total_correct_per)
        p_loc,r_loc,f1_loc = evalu(correct_loc,total_pred_loc,total_correct_loc)
        p_org,r_org,f1_org = evalu(correct_org,total_pred_org,total_correct_org)
        
        

        acc = np.mean(accs)
        print("acc:\t", 100 * acc, "f1:\t", 100 * f1, "p\t", 100*p, "r\t", 100*r)
        return {"acc": 100 * acc, "f1": 100 * f1, "p": 100*p, "r": 100*r,
            "f1_per": 100 * f1_per, "p_per": 100*p_per, "r_per": 100*r_per,
            "f1_loc": 100 * f1_loc, "p_loc": 100*p_loc, "r_loc": 100*r_loc,
            "f1_org": 100 * f1_org, "p_org": 100*p_org, "r_org": 100*r_org},results


    def train(self, train_data, test_data):
        # best_score = 0
        self.saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # self.sess = sess
            sess.run(self.init)

            import time
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 将指定格式的当前时间以字符串输出
            suffix = ".txt"
            newfile = './data/result/print/' + t + suffix
            newresultfile = './data/result/print/result_' + t + suffix
            self.printfile = newfile
            self.printresultfile = newresultfile

            file = open(newfile, 'a')
            file.close()

            file = open(newresultfile, 'a')
            file.close()

            for epoch in range(self.config.nepochs):
                print('The', epoch + 1, ' trainning...')
                self.run_epoch(sess, train_data, epoch,  test_data)

    def evaluate(self, test_data):
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            # saver.restore(sess, self.config.model_path)
            #saver = tf.train.import_meta_graph(self.config.meta_path)
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.model_path))
#            print(self.run_evaluate(sess, test_data))
