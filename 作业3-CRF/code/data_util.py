# coding = utf-8

import sys
import codecs

raw_data_filename = 'data_nr.txt'
traindata_filename = 'train_data.txt'
testdata_filename = 'test_data.txt'

def read_data():
#	raw_file = open(raw_data_filename,'r')
#	raw_data = raw_file.readlines()
#	raw_file.close()
	raw_data = []
	for line in  codecs.open(raw_data_filename,'r','utf-8'):
		raw_data.append(line)
	data_len = len(raw_data)
	train_data = raw_data[:int(data_len*0.8)]
	test_data = raw_data[int(data_len*0.8+1):]
	preprocess(train_data,traindata_filename)
	preprocess(test_data,testdata_filename)

def preprocess(data,filename):
	file = codecs.open(filename,'w','utf-8')
	i = 0
	for line in data:
		i += 1
		if i % 2000 == 0:
			print(i)
		wordstags = line.strip().split(' ')
		for wordtag in wordstags:
			wordtag = wordtag.strip()
			if wordtag == '':
				continue
			if '/nr' in wordtag:
				word = wordtag[:wordtag.rfind('/')]
				tag = wordtag[wordtag.rfind('/')+1:]
#				file.write(word+'\t'+tag+'\t'+'B-'+tag.upper()+'\n')
				file.write(word+'\t'+'B-'+tag.upper()+'\n')
			else:
#				file.write(wordtag+'\t'+'s'+'\t'+'O'+'\n')
				file.write(wordtag+'\t'+'O'+'\n')
		file.write('\n')
	file.flush()
			
if __name__ == '__main__':
	read_data()
