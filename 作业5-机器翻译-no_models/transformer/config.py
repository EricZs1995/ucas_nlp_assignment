# -*- coding: utf-8 -*-
class Config:

    source_data_file = './data/cn.txt'   # 翻译源语言文件
    target_data_file = './data/en.txt'   # 翻译目标语言文件
    save_models_dir = './models/'        # 模型存储目录
    evaluate_result_dir = './result/'    # 评估结果目录
    split = 0.95          # 数据集划分比例
    batch_size = 32       #
    lr = 0.0001           #
    lr_decay = 0.98       #
    dim_model = 512       # 单层模型单元数目
    num_block = 6         # encoder和decoder的block数目
    num_header = 8        # multiheader的header数目
    num_epoch = 25        # 训练轮数
    min_count = 2         # 词频最小阈值
    max_length = 32       # 语句最大长度
    dropout_rate = 0.2    #
