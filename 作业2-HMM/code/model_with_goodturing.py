#encoding=utf-8
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import codecs
import random

#inputfile='raw_data.txt'
inputfile = 'raw_data.txt'
outputfile = './output.txt'
avg = 100*1/102784

def HmmLearn(sequences,sequences_words,sequences_tags,worddict,tagdict,words2ids,tags2ids):
	emission = estimateEmissionProb(tagdict,sequences_tags, tags2ids)
	transition = estimateTransitionProb(tagdict, sequences_tags, tags2ids)
	observation = estimateObservationProb(sequences,worddict,tagdict, words2ids, tags2ids)
	return emission,transition,observation

def HmmPredict(sequences,uni,bi_tag,bi_tup,uni_p,bi_tag_p,bi_tup_p):
	opts = []
	opts_ids = []
	total = 0
	total_correct = 0
	k=0
	for words,tags in sequences:
#		opt_tag, opt_tagid = viterbi(emission,transition,observation,words,words2ids,tags2ids,ids2tags,tagdict)
		opt_tag = viterbi_smooth(uni,bi_tag,bi_tup,uni_p,bi_tag_p,bi_tup_p,words)

		opts.append(opt_tag)
		total += len(opt_tag)
		this_cor = 0
		for i in range(len(opt_tag)):
			if opt_tag[i] == tags[i]:
				this_cor += 1
		total_correct +=this_cor 

#		print(tags_id)
		print(opt_tag)
		print(tags)

		print('\tk',k,'\ttotal:\t',total,'\ttotal_coor:\t',total_correct,'\tlen:\t',len(opt_tag),'\tcorre:\t',this_cor,'\tacc:\t',total_correct/total)
		k +=1
	return opts,total,total_correct

def load_data(inputfile):
	sequences = []
	sequences_words = []
	sequences_tags = []
	data = []
	worddict1 = set()
	tagdict1 = set()
	worddict = []
	tagdict = []
	lines = []
	for line in codecs.open(inputfile,'r','utf-8'):
		lines.append(line)
	print('------------')
#	random.shuffle(lines)
	print('linesnum:',len(lines))
	for line in lines:
		words = []
		tags = []
		wordtag = []
		words_tags = line.strip().split(' ')
		for word_tag in words_tags:
			word_tag = word_tag.strip()
			word = word_tag[:word_tag.rfind('/')]
			tag = word_tag[word_tag.rfind('/')+1:]
			if word=='' and tag == '':
				continue
			if (tag != 'w' and word == '') or (tag == '》）。') or tag == '！”' or tag == '。”' :
				word = tag
				tag = 'w'
			if word not in worddict1:
				worddict.append(word)
				worddict1.add(word)
			if tag not in tagdict1:
				tagdict.append(tag)
				tagdict1.add(tag)
			words.append(word)
			tags.append(tag)
		sequences.append((words,tags))
		sequences_words.append(words)
		sequences_tags.append(tags)
	return sequences,sequences_words,sequences_tags,worddict,tagdict


def processdata(sequences,sequences_words,sequences_tags):
	ind_seq_split = int(len(sequences)*0.8)
	train_data = sequences[:ind_seq_split]
	train_seq_words = sequences_words[:ind_seq_split]
	train_seq_tags = sequences_tags[:ind_seq_split]
	test_data = sequences[ind_seq_split+1:]
	test_seq_words = sequences_words[ind_seq_split+1:]
	test_seq_tags = sequences_tags[ind_seq_split+1:]
	
	uni_dic, bi_tag_dic = stat_uni_bi(train_seq_tags)
	uni_unseen_dic, bi_tag_unseen_dic = stat_uni_bi(test_seq_tags)
	bi_tup_dic = stat_bi_tup(train_data)
	bi_tup_unseen_dic = stat_bi_tup(test_data)
	uni_unseen = uni_unseen_dic.keys()-uni_dic.keys()
	bi_tag_unseen = bi_tag_unseen_dic.keys()-bi_tag_dic.keys()
	bi_tup_unseen = bi_tup_unseen_dic.keys()-bi_tup_dic.keys()
	
	for uni in uni_unseen:
		uni_dic[uni] = 0
	for tag in bi_tag_unseen:
		bi_tag_dic[tag]=0
	for tup in bi_tup_unseen:
		bi_tup_dic[tup] = 0
#	def good_turing(dic,V_0,N):
	
	uni_p = good_turing(uni_dic,sum(uni_dic.values()))
	bi_tag_p = good_turing(bi_tag_dic,sum(bi_tag_dic.values()))
	bi_tup_p = good_turing(bi_tup_dic,sum(bi_tup_dic.values()))
	print(uni_p)
	print(bi_tup_p)
	return train_data,test_data, uni_p, bi_tag_p, bi_tup_p,uni_dic,bi_tag_dic,bi_tup_dic


def stat_bi_tup(sequences):
	bigram_dic = OrderedDict()
	for i,(words,tags) in enumerate(sequences):
		for j in range(len(tags)):
			tup = (tags[j],words[j])
			if tup not in bigram_dic:
				bigram_dic[tup] = 1
			else:
				bigram_dic[tup] += 1
	return bigram_dic


def stat_uni_bi(sequences):
	unigram_dic = OrderedDict()
	bigram_dic = OrderedDict()
	for seq in sequences:
		pre = '<start>'	
		for tag in seq:
			if tag in unigram_dic:
				unigram_dic[tag]+=1
			else:
				unigram_dic[tag]=1
			tup = (pre,tag)
			if tup in bigram_dic:
				bigram_dic[tup] += 1
			else:
				bigram_dic[tup] = 1
			pre = tag
		tup = (pre,'<end>')
		if tup in bigram_dic:
			bigram_dic[tup] += 1
		else:
			bigram_dic[tup] = 1
	unigram_dic['<start>'] = len(sequences)
	unigram_dic['<end>'] = len(sequences)
	return unigram_dic,bigram_dic
	
def cal_p(a,b,uni,bi, uni_p , bi_p):
	if (a,b) not in bi:
		fenzi = bi_p[0]
	else:
		fenzi = bi_p[bi[(a,b)]]
	if a not in uni:
		fenmu = uni_p[0]
	else:
		fenmu = uni_p[uni[a]]
	return fenzi/fenmu


def word_mapping(dictionary):
    ids2words = {i: v for i, v in enumerate(list(dictionary))}
    words2ids = {v: k for k, v in ids2words.items()}
    return words2ids, ids2words


def estimateEmissionProb(tagdict,sequences_tags, tags2ids):
	len_tag = len(tagdict)
	len_seqs = len(sequences_tags)
	emission = np.zeros((len_tag,1))
	emission_prob = np.zeros((len_tag,1))
	for i in range(len_seqs):
		emission[tags2ids[sequences_tags[i][0]]] += 1
	emission_sum =np.array(emission).sum()
	for i in range(len_tag):
		emission_prob[i] = 100*(emission[i]+1)/(emission_sum+len_seq)
	return emission_prob


def estimateTransitionProb(tagdict, sequences_tags, tags2ids):
	len_tag = len(tagdict)
	transition = np.zeros((len_tag,len_tag))
	transition_prob = np.zeros((len_tag,len_tag))
	for tags in sequences_tags:
		for i,tag in enumerate(tags):
			if i==0:
				continue
			transition[tags2ids[tags[i-1]]][tags2ids[tags[i]]] += 1
	trans_sum = np.array(transition).sum(axis = 1)
	for i in range(len_tag):
		if trans_sum[i]==0:
			for j in range(len_tag):
				transition_prob[i][j] = -1.0
		for j in range(len_tag):
			transition_prob[i][j] = -1.0 if (transition[i][j]==0) else 100* transition[i][j]/trans_sum[i]

	return transition_prob


def estimateObservationProb(sequences,worddict,tagdict, words2ids, tags2ids):
	len_word = len(worddict)
	len_tag = len(tagdict)
	observation = np.zeros((len_tag,len_word))
	observation_prob = np.zeros((len_tag,len_word))
	for (words,tags) in  sequences:
		for (word,tag) in zip(words,tags):
			observation[tags2ids[tag]][words2ids[word]] += 1
#	print('observation[10]',observation[20][:500])
	observation_sum  = np.array(observation).sum(axis = 1)
#	print('observationsum:',observation_sum)
	for i in range(len_tag):
		if observation_sum[i] == 0:
			for j in range(len_word):
				observation_prob[i][j] = -1.0
		for j in range(len_word):
			observation_prob[i][j] = -1.0 if (observation[i][j] ==0) else 100*observation[i][j]/observation_sum[i]
#	print('observation_pro[10]',observation_prob[20][:500])
	return observation_prob


def viterbi_smooth(uni,bi_tag,bi_tup,uni_p,bi_tag_p,bi_tup_p,sequence):
	tags = list(uni.keys())
	len_tag = len(tags)
	len_seq = len(sequence)

	viterbi_seq = np.zeros((len_tag,len_seq+1))
	viterbi_opt = np.zeros([len_tag,len_seq+1],dtype = np.int)

# cal_p(a,b,uni,bi, uni_p , bi_p):
	
	for i in range(len_tag):
		viterbi_seq[i][0] = cal_p('<start>',tags[i],uni,bi_tag,uni_p,bi_tag_p)*cal_p(tags[i],sequence[0],uni,bi_tup,uni_p,bi_tup_p)
		if viterbi_seq[i][0]<=0:
			viterbi_seq[i][0] = avg
		viterbi_opt[i][0] = -1
	for k,word in enumerate(sequence):
		if k ==0:
			continue
		for i in range(len_tag):
			for j in range(len_tag):
#				if viterbi_seq[j][k-1]<0 or transition[j][i]<0 or observation[i][wordid]<0:
#					new_value = avg
#				else:
#					new_value =  viterbi_seq[j][k-1]*transition[j][i]*observation[i][wordid]
				new_value = viterbi_seq[j][k-1]*cal_p(tags[j],tags[i],uni,bi_tag,uni_p,bi_tag_p)*cal_p(tags[i],word,uni,bi_tup,uni_p,bi_tup_p)
				if viterbi_seq[i][k] < new_value:
					viterbi_seq[i][k] = new_value
					viterbi_opt[i][k] = j
	max_prob = 0.0
	max_state = 0
	for i in range(len_tag):
		if max_prob < viterbi_seq[i][len_seq-1]:
			max_prob = viterbi_seq[i][len_seq-1]
			max_state = i
	stack = []
	k=len_seq-1
	print(viterbi_seq)
#	print(viterbi_opt)
#	print('viterbi:',viterbi_opt[0])
	while max_state != -1 :
		stack.append(max_state)	
		max_state = viterbi_opt[max_state][k]
		k -= 1
	opt_tagid = stack[::-1]
	opt_tag = []
	for tagid in opt_tagid:
		opt_tag.append(tags[tagid])
	return opt_tag


def viterbi(emission,transition,observation,sequence,words2ids,tags2ids,ids2tags,tagdict):
	len_tag = len(emission)
	len_seq = len(sequence)

	viterbi_seq = np.zeros((len_tag,len_seq+1))
	viterbi_opt = np.zeros([len_tag,len_seq+1],dtype = np.int)

	for i in range(len_tag):
		viterbi_seq[i][0] = emission[i]*observation[i][words2ids[sequence[0]]]
		if viterbi_seq[i][0]<=0:
			viterbi_seq[i][0] = avg
		viterbi_opt[i][0] = -1
	for k,word in enumerate(sequence):
		if k ==0:
			continue
		wordid = words2ids[word]
		for i in range(len_tag):
			for j in range(len_tag):
				if viterbi_seq[j][k-1]<0 or transition[j][i]<0 or observation[i][wordid]<0:
					new_value = avg
				else:
					new_value =  viterbi_seq[j][k-1]*transition[j][i]*observation[i][wordid]
				if viterbi_seq[i][k] < new_value:
					viterbi_seq[i][k] = new_value
					viterbi_opt[i][k] = j
	max_prob = 0.0
	max_state = 0
	for i in range(len_tag):
		if max_prob < viterbi_seq[i][len_seq-1]:
			max_prob = viterbi_seq[i][len_seq-1]
			max_state = i
	stack = []
	k=len_seq-1
#	print(viterbi_seq)
#	print(viterbi_opt)
#	print('viterbi:',viterbi_opt[0])
	while max_state != -1 :
		stack.append(max_state)	
		max_state = viterbi_opt[max_state][k]
		k -= 1
	opt_tagid = stack[::-1]
	opt_tag = []
	for tagid in opt_tagid:
		opt_tag.append(ids2tags[tagid])
	return opt_tag,opt_tagid

def writefile(filename,vectors):
	# file = codecs.open(filename,'w','utf-8')
	np.savetxt(filename, vectors, delimiter=' ')
	# for vector in vectors:
		
def good_turing(dic,N):
	bucket = {}
	max_r = 0
	for tup in dic:
		count = dic[tup]
		if count > max_r:
			max_r = count
		if count not in bucket:
			bucket[count] = 1
		else:
			bucket[count] += 1
	r_dic = np.zeros([max_r+2])
	p_dic = np.zeros([max_r+2])
	for r in range(1,max_r+1):
		if r in bucket:
			if r+1 in bucket:
				r_dic[r] = (r+1)*bucket[r+1]/bucket[r]
			else:
				r_dic[r] = r
			p_dic[r] = r_dic[r]/N
	p_dic[0] = bucket[1]/(N*bucket[0])
	p_sum = sum(p_dic)
	for i in range(max_r):
		if p_dic[i]!=0:
			p_dic[i] /=p_sum
	return p_dic


if __name__ == '__main__':
	print('loading data...')
	sequences,sequences_words,sequences_tags,worddict,tagdict = load_data(inputfile)
	words2ids, ids2words = word_mapping(worddict)
	tags2ids, ids2tags = word_mapping(tagdict)
	print(len(worddict))
	print(len(tagdict))
	train_data,test_data, uni_p, bi_tag_p, bi_tup_p ,uni_dic,bi_tag_dic,bi_tup_dic= processdata(sequences,sequences_words,sequences_tags)

#	len_8 = int(len(sequences_tags)*0.8)
#	print(len_8)
	print('learn...')
	print('predict...')
#	pre_tags ,total,total_correct = HmmPredict(sequences[len_8+1:],sequences_words[len_8+1:],sequences_tags[len_8+1:],worddict,tagdict,words2ids,tags2ids,emission,transition,observation)
	pre_tags ,total,total_correct = HmmPredict(test_data,uni_dic,bi_tag_dic,bi_tup_dic,uni_p,bi_tag_p,bi_tup_p)
#	pre_tags ,opts_ids,total,total_correct = HmmPredict(sequences[:100],sequences_words[:100],sequences_tags[:100],worddict,tagdict,words2ids,tags2ids,emission,transition,observation)
	print(pre_tags)
	print('acc:\t', total_correct/total)
	print('over...')

