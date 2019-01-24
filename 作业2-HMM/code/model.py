#encoding=utf-8
import tensorflow as tf
import numpy as np
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
#	transition = []
#	observation = []
	return emission,transition,observation
	# print(emission)
	# print(transition)
	# print(observation)

def HmmPredict(sequences,sequences_words,sequences_tags,worddict,tagdict,words2ids,tags2ids,emission,transition,observation):
	opts = []
	opts_ids = []
	total = 0
	total_correct = 0
	k=0
	for words,tags in sequences:
		opt_tag, opt_tagid = viterbi(emission,transition,observation,words,words2ids,tags2ids,ids2tags,tagdict)
		opts.append(opt_tag)
		opts_ids.append(opt_tagid)
		total += len(opt_tag)
#		total_correct += len(list(opt_tag) & list(tags))
#		total_correct += sum(np.array(opt_tag),np.array(tags))		
		tags_id = []
		for tag in tags:
			tags_id.append(tags2ids[tag])
		
		this_cor = sum(np.array(opt_tagid)==np.array(tags_id))		
		total_correct +=this_cor 

#		print(tags_id)
#		print(opt_tag)
#		print(tags)

		print('\tk',k,'\ttotal:\t',total,'\ttotal_coor:\t',total_correct,'\tlen:\t',len(opt_tag),'\tcorre:\t',this_cor,'\tacc:\t',total_correct/total)
#		if k%100 == 0:
#			print('k\t',k,'total:\t',total,'total_coor:\t',total_correct,'len:\t',len(opt_tag),'corre:\t',this_cor)
#			print('opt:',opt_tag)
#			print('tag:',tags)
		k +=1
	return opts,opts_ids,total,total_correct

def load_data(inputfile):
	sequences = []
	sequences_words = []
	sequences_tags = []
	data = []
	worddict1 = set()
	tagdict1 = set()
	worddict = []
	tagdict = []
#	worddict.add('<start>')
#	tagdict.add('0')
#	worddict.add('<end>')
#	tagdict.add('-1')
	lines = []
	for line in codecs.open(inputfile,'r','utf-8'):
		lines.append(line)
	random.shuffle(lines)
	print('linesnum:',len(lines))
	for line in lines:
		words = []
		tags = []
		wordtag = []
#		words.append('<start>')
#		tags.append('0')
		# wordtag.append(('<start>','0'))
		words_tags = line.strip().split(' ')
#		print(words_tags)
#		print(line)
		for word_tag in words_tags:
#			word,tag = word_tag.strip().split('/')[:3]
			word_tag = word_tag.strip()
			word = word_tag[:word_tag.rfind('/')]
			tag = word_tag[word_tag.rfind('/')+1:]
			if word=='' and tag == '':
#				print('line:',line)
#				print('wordtags:',words_tags)
#				print('word:',word,' tag:',tag)
#				print('words:',words[-5:])
#				print('tags:',tags[-5:])
#				print('---------------------------------------------')
				continue
			if (tag != 'w' and word == '') or (  tag in ['》）。', '！”','。”']     )  :
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
			# wordtag.append((word,tag))
#		words.append('<end>')
#		tags.append('-1')
		# wordtag.append(('<end>','-1'))
		sequences.append((words,tags))
#	sequences = random.shuffle(sequences)
#	for i,(words,tags) in enumerate(sequences): 
		sequences_words.append(words)
		sequences_tags.append(tags)
	return sequences,sequences_words,sequences_tags,worddict,tagdict

def word_mapping(dictionary):
    ids2words = {i: v for i, v in enumerate(list(dictionary))}
    words2ids = {v: k for k, v in ids2words.items()}
    return words2ids, ids2words


def estimateEmissionProb(tagdict,sequences_tags, tags2ids):
	len_tag = len(tagdict)
	len_seqs = len(sequences_tags)
	emission = np.zeros((len_tag,1))
#	print(len(emission))
#	print(len(emission[0]))
	emission_prob = np.zeros((len_tag,1))
	for i in range(len_seqs):
		emission[tags2ids[sequences_tags[i][0]]] += 1
	emission_sum =np.array(emission).sum()
#	print(emission_sum)
#	print(emission)
	for i in range(len_tag):
		emission_prob[i] = -1.0 if (emission[i]==0) else 100*emission[i]/emission_sum
#	print(emission_prob)
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


def viterbi(emission,transition,observation,sequence,words2ids,tags2ids,ids2tags,tagdict):
	len_tag = len(emission)
	len_seq = len(sequence)

	viterbi_seq = np.zeros((len_tag,len_seq+1))
	viterbi_opt = np.zeros([len_tag,len_seq+1],dtype = np.int)

	#1:初始化，计算t=0
	for i in range(len_tag):
		viterbi_seq[i][0] = emission[i]*observation[i][words2ids[sequence[0]]]
		if viterbi_seq[i][0]<=0:
			viterbi_seq[i][0] = avg
		viterbi_opt[i][0] = -1
	#2:迭代
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
	#3:终止
	max_prob = 0.0
	max_state = 0
	for i in range(len_tag):
		if max_prob < viterbi_seq[i][len_seq-1]:
			max_prob = viterbi_seq[i][len_seq-1]
			max_state = i
	#4:回溯
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
		

if __name__ == '__main__':
	print('loading data...')
	sequences,sequences_words,sequences_tags,worddict,tagdict = load_data(inputfile)
	words2ids, ids2words = word_mapping(worddict)
	tags2ids, ids2tags = word_mapping(tagdict)
#	print(words2ids[worddict[0]])
	print(worddict)
	print(len(worddict))
	print(tagdict)
	print(len(tagdict))
#	print('tag:',tagdict)
#	print('tag2id',tags2ids)
#	print('id2tag',ids2tags)
	len_8 = int(len(sequences_tags)*0.8)
#	print(len_8)
	print('learn...')
	emission,transition,observation = HmmLearn(sequences[:len_8],sequences_words[:len_8],sequences_tags[:len_8],worddict,tagdict,words2ids,tags2ids)
#	emission,transition,observation = HmmLearn(sequences,sequences_words,sequences_tags,worddict,tagdict,words2ids,tags2ids)
	print('writing...')
	writefile(outputfile,emission)
	writefile(outputfile,transition)
	writefile(outputfile,observation)
#	print(emission)
#	print(transition)
#	print(observation)
	print('predict...')
	pre_tags ,opts_ids,total,total_correct = HmmPredict(sequences[len_8+1:],sequences_words[len_8+1:],sequences_tags[len_8+1:],worddict,tagdict,words2ids,tags2ids,emission,transition,observation)
#	pre_tags ,opts_ids,total,total_correct = HmmPredict(sequences[:100],sequences_words[:100],sequences_tags[:100],worddict,tagdict,words2ids,tags2ids,emission,transition,observation)
	print(pre_tags)
	print('acc:\t', total_correct/total)
	print('over...')

