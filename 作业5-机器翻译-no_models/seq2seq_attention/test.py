import os
import tensorflow as tf
import numpy as np
from data_util import TextConverter, batch_generator,Preprocess
from model import  Model
from config import Config
import codecs


def main(_):

    model_path = os.path.join('models', Config.file_name)

    et = TextConverter(text=None,save_dir='data/en_vocab.pkl', max_vocab=Config.source_vocab_size, seq_length = Config.seq_length+1)
    zt = TextConverter(text=None,save_dir='data/zh_vocab.pkl', max_vocab=Config.target_vocab_size, seq_length = Config.seq_length) 
    print('english vocab lens:',et.vocab_size)
    print('chinese vocab lens:',zt.vocab_size)

    en_arrs = et.get_en_arrs('data/en_pro.txt')
    zh_arrs = zt.get_en_arrs('data/zh_pro.txt')
    test_g = batch_generator( zh_arrs, en_arrs, Config.batch_size)
    model_path = os.path.join('models', Config.file_name)
    mname = open(model_path + '/checkpoint', 'r').read().split('"')[1]
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)
    

    pre =  Preprocess()
#    pre.clears()
    enes, zhes = pre.get_seq()
    len = 100
#    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/" + mname, "w", "utf-8") as fout:
        for i in range(len):
            zh = zhes[i] 
            zh = [i for i in zh]
            zh = zh[1:-1]
            en_arr, arr_len = zt.text_to_arr(zh)

            test_g = [np.array([en_arr,]), np.array([arr_len,])]
            output_ids = model.test(test_g, model_path, et)
            got = et.arr_to_text(output_ids)
            fout.write("SOURCE>> " + zhes[i][3:-4] +"\n")
            fout.write("TARGET>> " + enes[i] )
            fout.write("TRANSL>> " + got + "\n\n")
            fout.flush()
            print("SOURCE>> ",zhes[i])
            print("TARGET>> ",enes[i])
            print("TRANSL>> ",got)
        

if __name__ == '__main__':
    tf.app.run()
