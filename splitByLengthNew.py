import sys,os
import json
import re

train_file_positive='data/positive_matrix.tsv.translated'
train_file_negative='data/negative_matrix.tsv.translated'

dev_file = 'data/dev_matrix.tsv.translated'
test_file = 'data/test_matrix.tsv.translated'

#for now read the existing vocab->int mapping
vocab_file = '../VSMKBC/data/proc/domain.tokenString'
with open(vocab_file) as vocab:
	vocab2int = json.load(vocab)

#..from David's featureExtraction.py
num = re.compile("\d")
def normalize(string):
	string = re.sub(num,"#NUM",string)
	return string 

pad_token = -1
#train
#count the max path
max_length = -1
train_files = [train_file_positive,train_file_negative]
for train_counter,train_file in enumerate(train_files):
	with open(train_file) as train:
		for entity_count, line in enumerate(train): #each entity pair
			split = line.split('\t')
			if len(split) !=3:
				continue
			split = split[2].split('###')
			for each_path in split:
				each_path = each_path.rstrip()
				length = 0
				relation_types = each_path.split('-')
				for relation in relation_types: #every node in the path
					relation = relation.rstrip()
					if len(relation) !=0:
						length = length + 1
				if length > max_length:
					max_length = length
print 'Max length is '+str(max_length)
output_dir = 'data/train'

labels = [1,-1]
output_file = output_dir+'/'+'train.txt'
#delete all files in out_dir
for f in os.listdir(output_dir):	
	
	if os.path.exists(output_dir+'/'+f):
		print f
		os.remove(output_dir+'/'+f)
# if os.path.exists(output_file):
# 	os.remove(output_file)
for train_counter,train_file in enumerate(train_files):
	label = labels[train_counter]
	with open(train_file) as train:
		for entity_count, line in enumerate(train): #each entity pair
			all_paths=[]
			split = line.split('\t')
			if len(split) !=3:
				continue
			e1 = split[0].rstrip()
			e2 = split[1].rstrip()
			split = split[2].split('###') #all the paths linking e1 and e2
			# max_length = -1
			# length_list = [] #stores the len of each path
			# for each_path in split: #each path of the entity pair
			# 	each_path = each_path.rstrip()
			# 	length = 0
			# 	relation_types = each_path.split('-')
			# 	for relation in relation_types: #every node in the path
			# 		relation = relation.rstrip()
			# 		if len(relation) !=0:
			# 			length = length + 1
			# 	if length > max_length:
			# 		max_length = length
			key_error_count = 0
			for each_path in split: #now pad them
				each_path = each_path.rstrip()
				try:
					relation_types = map(lambda r:vocab2int['domain'][normalize(r.rstrip())],[x for x in each_path.split('-') if len(x.rstrip())!=0])
					length = len(relation_types)
					pad_length = max_length - length
					for i in xrange(pad_length):
						relation_types.insert(0,pad_token)
					all_paths.append(relation_types)
				except KeyError:
					key_error_count = key_error_count + 1
					continue			
			output_str = str(label)+'\t'
			path_counter = 0
			for path in all_paths:					
				counter = 0
				for relation_type in path:
					if counter == 0:
						output_str = output_str+str(relation_type)
					else:
						output_str = output_str+' '+str(relation_type)
					counter = counter + 1
				if path_counter < len(all_paths)-1:
					if len(output_str)!=0:
						output_str = output_str+';'
				path_counter = path_counter + 1
			# with open(output_file,'a') as out:	
			# out.write(output_str+'\n')
			if path_counter!=0:
				output_file_with_pathlen = output_file+'.'+str(path_counter)+'.int'
				with open(output_file_with_pathlen,'a') as out:
					out.write(output_str+'\n')
			entity_count = entity_count + 1
			if entity_count % 100 == 0:
				print 'Processed '+str(entity_count)+' entity pairs'			 
			#print str(key_error_count) +'keys were not found'			
