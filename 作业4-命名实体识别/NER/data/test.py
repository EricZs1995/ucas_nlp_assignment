from data_util import *
from config import *

if __name__ == "__main__":
    # a = {('PER', 18, 21), ('ORG', 0, 6), ('LOC', 22, 24), ('PER', 10, 13)}
    # b = {('ORG', 0, 6), ('LOC', 18, 20), ('LOC', 22, 24), ('PER', 10, 13)}
    #
    # print(len(a&b))

    label_chunks, label_chunks_devs = get_chunks(label, self.config.tag2id)
    # print(2/3)
    # print(2//3)
    #
    # print('a',1)
    # print("acc:\t", 100 , "\tf1:\t", 100 )
    # newfile = '../data/12345'
    # file = open(newfile,'a')
    # file.write('s')
    # file.write('a')
    # file.close()
    # file = open(newfile, 'a')
    # file.write('d')
    # file.close()

    # cwd = os.getcwd()
    # print(cwd)
    # import time
    # dir = '../data/result/print/'
    # t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  #将指定格式的当前时间以字符串输出
    # suffix = ".txt"
    # newfile= dir+t+suffix
    # file = open(newfile, 'w')




    # list = [1,2,3,4,5,10,2]

    # print(max(list))
    #
    # words = [[1,3,4,56,4],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7]]
    # for w in words:
    #     file.writelines(str(w))
    # words_pad = padding_sentence(words,list,-1)
    # print(words_pad)
    # for i in range(10):
    #     print(i)
    # print(list[:100])

    # t = [[1,3,4,56,4],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7],[3,3,5,6,7]]
    # train = predataset(words,t)
    # for i, (words, labels) in enumerate(batches(train, 2)):
    #     print(i)
    #     print((words,labels))
    #
    #
    #
    # print("dfs")
    # ss = [['d','s'],['a','d']]
    #
    #
    # print(zip(zip(*ss)))

    # d = dict()
    # # ss = ['s','a','b', 's', 'g', 't', 'a']
    # for idx, word in enumerate(ss):
    #     d[word] = idx
    # print(d)

# print(tags)
# print(tags_lens)

# for i in range(len(words_lens)):
# 	if(words_lens[i]!=tags_lens[i]):
# 		print('[0]error', i)