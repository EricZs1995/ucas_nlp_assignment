# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
from model import *
from tqdm import tqdm
from data_util import *
from nltk.translate.bleu_score import corpus_bleu


def train():
    ch_idxes, en_idxes, ches, enes, len_ch_vocab, len_en_vocab = preprocess_data(True)
    g = Model(True)
    print('chlen', len(ch_idxes))
    print('enlen', len(en_idxes))
    sv = tf.train.Supervisor(graph=g.graph, logdir=Config.save_models_dir, save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(Config.num_epoch):
            if sv.should_stop():
                break
            num_batch = (len(ch_idxes) + Config.batch_size) // Config.batch_size - 1
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                #    sess.run(g.train_op, feed_dict={g.ch:ch, g.en:en})
                sess.run(g.train_op)
            gs = sess.run(g.global_step)
            sv.saver.save(sess, Config.save_models_dir + '/model_epoch_%02d' % (epoch))
    print('training over...')


def test():
    print('test...')
    ch_idxes, en_idx2word, ches, enes, len_ch_vocab, len_en_vocab = preprocess_data(False)
    g = Model(False)
    num_batch = len(ch_idxes) // Config.batch_size
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(Config=tf.configProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Config.save_models_dir))
            print('restore')
            mname = open(Config.save_models_dir + '/checkpoint', 'r').read().split('"')[1]  # model     name
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                #                for i in range(len(ch_idxes) // Config.batch_size):
                for i in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    x = ch_idxes[i * Config.batch_size: (i + 1) * Config.batch_size]
                    sources = ches[i * Config.batch_size: (i + 1) * Config.batch_size]
                    targets = enes[i * Config.batch_size: (i + 1) * Config.batch_size]

                    preds = np.zeros((Config.batch_size, Config.max_length), np.int32)
                    for j in range(Config.max_length):
                        _preds = sess.run(g.preds, {g.ch: x, g.en: preds})
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):
                        got = " ".join(en_idx2word[idx] for idx in pred).split("</s>")[0].strip()
                        fout.write("SOURCE>> " + source + "\n")
                        fout.write("TARGET>> " + target + "\n")
                        fout.write("TRANSL>> " + got + "\n\n")
                        fout.flush()

                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    print('start...')
    if len(sys.argv) == 1:
        print('option')
    else:
        if sys.argv[1] == 'test':
            test()
        elif sys.argv[1] == 'train':
            train()
