import os
import tensorflow as tf
from data_util import TextConverter, batch_generator
from model import  Model
from config import Config

#from model2 import  Model,Config
#from model_attention import Model,Config

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    et = TextConverter(text=None,save_dir='data/en_vocab.pkl', max_vocab=Config.source_vocab_size, seq_length = Config.seq_length+1)
    zt = TextConverter(text=None,save_dir='data/zh_vocab.pkl', max_vocab=Config.target_vocab_size, seq_length = Config.seq_length)  # +1是因为，decoder层序列拆成input=[:-1]和label=[1:]
    print('english vocab lens:',et.vocab_size)
    print('chinese vocab lens:',zt.vocab_size)

    en_arrs = et.get_en_arrs('data/en_pro.txt')
    zh_arrs = zt.get_en_arrs('data/zh_pro.txt')

    train_g = batch_generator( zh_arrs, en_arrs, Config.batch_size)

    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    model.config.max_steps =1000*len(zh_arrs)// model.config.batch_size
    model.config.save_every_n = len(zh_arrs)// model.config.batch_size
    print(model.config.max_steps)
    print(model.config.save_every_n)
    print('start to training...')
    model.train(train_g, model_path)



if __name__ == '__main__':
    tf.app.run()
