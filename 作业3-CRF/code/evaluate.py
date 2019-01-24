import codecs

tag_filename = 'test.result'

def evaluate():
	total = 0
	total_correct = 0
	total_tag = 0
	predict_total = 0
	predict_correct = 0
	for line in codecs.open(tag_filename,'r','utf-8'):
		line  = line.strip()
		if line == '':
			continue
		words = line.split('\t')
		total += 1
		if words[-2] == words[-1]:
			total_correct += 1
		if words[-2] == 'B-NR':
			total_tag += 1
		if words[-1] == 'B-NR':
			predict_total += 1
		if words[-1] == words[-2] and words[-1]=='B-NR':
			predict_correct += 1
	print('total:\t',total,'total_correct:\t',total_correct,'total_tag:\t',total_tag,'predict_total:\t',predict_total,'predict_correct:\t',predict_correct)	
	accuracy = total_correct/total
	precision = predict_correct/predict_total
	recall=predict_correct/total_tag
	f1 =2*precision*recall/(precision+recall) 
	print('Accuracy:\t',accuracy ,'f1:\t',f1,'precision:\t',precision,'recall:\t',recall)


if __name__ == '__main__':
	evaluate()
