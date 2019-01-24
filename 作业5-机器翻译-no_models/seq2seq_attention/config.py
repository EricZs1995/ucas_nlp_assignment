

class Config(object):
    source_data_file = './data/cn.txt'  # 翻译源语言文件
    target_data_file = './data/en.txt'  # 翻译目标语言文件
    save_models_dir = './models/'       # 模型存储目录
    save_model_path = 'seq2seq'         #保存模型文件
    evaluate_result_dir = './result/'   # 评估结果目录
    embedding_dim = 128            # 词向量维度
    seq_length = 50                # 序列长度
    target_vocab_size = 11000      # 词汇表达小
    source_vocab_size = 27000
    num_layers= 2                  # 隐藏层层数
    hidden_dim = 128               # 隐藏层神经元
    share_emb_and_softmax = False
    train_keep_prob = 0.8          # dropout
    learning_rate = 1e-3           # 学习率
    batch_size = 32                #
    log_every_n = 20               #
    save_every_n = 0               #
    max_steps = 20000              # 总迭代次数
