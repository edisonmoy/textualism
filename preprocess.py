import numpy as np
import gensim as gensim
import pandas as pd
import json as json
import fnmatch
from gensim.test.utils import datapath
import spacy
import re
import time as tm
import nltk
import boto3



#Stemming Data
from nltk.stem import PorterStemmer

#Lemmatizing Data
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.corpus import brown



porter = PorterStemmer()

# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')#"en", disable=["parser","tagger","ner","textcat"])

def remove_stopwords_from_sentence(sentence, stop_word_list=None):
	if stop_word_list is None: stop_word_list = set(stopwords.words('english')).pop

	#Removes Stopwords from a sentence-array and lowercases all words
	sentence=[word for word in sentence if word not in stop_word_list]
	return sentence


def remove_stopwords(list_or_list_of_lists, stop_word_list=None):
	 is_this_list_of_lists = type(list_or_list_of_lists[0]) == list
	 if not is_this_list_of_lists: return remove_stopwords_from_sentence(list_or_list_of_lists, stop_word_list)
	 else: return [remove_stopwords_from_sentence(lst, stop_word_list) for lst in list_or_list_of_lists]


def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)



def construct_sentence_arr_from_text(text_string):
	"""
		Takes 'text_string' and produces a list of lists where each nested list
		contains the words of a sentence.

	"""
	# Replace every number by '#' think like the unknown character from the youtube tutorial.
	string_tester = re.sub('\d', '#',text_string)

	# Use 'nlp' library to make string into a sentence.
	intro_doc     = nlp(string_tester)
	sentences     = list(intro_doc.sents)

	sentence_arr  = []

	# Create list of lists where each nested list has words of a sentence.
	for i, sentence in enumerate(sentences):

		sen_text = sentence.text
		sen_text = sen_text.replace("\n", " ")
		sen_text=porter.stem(sen_text)
		sen_list = sen_text.split(' ')  # make list of words.

		# The initial text can contain "\n\n" which cause empty '' elements.
		# Remove these.
		sen_list = [element for element in sen_list if element != '']

		# Removes remove all occurances of hashtags.
		# This has the consequence of removing all numbers and references to cases.
		sen_list = [element for element in sen_list if '#' not in element]

		# If the length of the sentence is less than 5 don't use it.
		# TODO: this might be nicer to have later as it could break the attempt at reconstructing sentences.
		if len(sen_list) < 5: continue
		# The library might split sentences wrong, e.g.
		# 'Hi my name is \n Alex." instead of "Hi my name is Alex.".
		# To fix this, we look at the last sentence in 'sentence_arr' and check
		# that the last word contains a punctuation, i.e., ['.', '?', '!'].
		if len(sentence_arr) > 0:

			last_word      = sentence_arr[-1][-1]
			last_character = last_word[-1]
			if not last_character in ['.', '?', '!']:
				sentence_arr[-1] += sen_list

			else:
				sentence_arr.append(sen_list)
		else:
				sentence_arr.append(sen_list)

	# Tokenizer; handle conjungation e.g. ran/running/etc.
	return sentence_arr



#def clean_sentence_arr
def construct_word_arr(sentence_arr):
	sentence_arr_helper=[]
	lemmatizer = WordNetLemmatizer()
	for sentence in sentence_arr:
		sentence_tokens=word_tokenize(sentence)

		#sentence_tokens=nltk.pos_tag(sentence_tokens)
		sentence_lemma=[]
		for word, tag in nltk.pos_tag(sentence_tokens):
			if (word in ["'m","'s"]) or (len(word)==1) or (len(word)==2 and word[-1]=="."):
				continue
			wntag = tag[0].lower()
			if wntag in ['a', 'r', 'n', 'v']:
				lemma = lemmatizer.lemmatize(word, wntag)
				sentence_lemma.append(lemma)

			elif wntag not in ['.',',',':',')','('] and word not in ["''","``","'","[","]","§"]:
				if word[0] in ".!?)()[]/,'":
					word=word[1:]

				if word[-1] in ".!?)()[]/,'-+":
					word=word[:-1]

				if len(word)>1:
					sentence_lemma.append(word)

		sentence_arr_helper.append(sentence_lemma)
	return sentence_arr_helper

def print_common_words(list_of_sentences,n_words=10):
	#Takes lists of lists that contains
	#sentences with the nested list containing lemmatized,stemmed and tokenized
	#versions of a text corpus
	meta_arr = [item for sublist in list_of_sentences for item in sublist]
	word_freq = Counter(meta_arr)
	common_words = word_freq.most_common(n_words)
	print (common_words)



path="./Justia_Cases/"
for year in range(19,2020):
	print("Start processing year {}").format(year)
	distros_dict = json.load( open(path+str(year)+'_justia.json', 'r')  )# data[2000]
	sentence_arr = []
	i_max=30
	t1=tm.time()
	for i, case in enumerate(distros_dict):  # This died at 25 because there was a 0.
		print("%i Year: \r[%i / %i]"%(year,i+1, len(distros_dict)))
		case = case['written_opinion']
		if case!=None:
			for index_opinion in range(len(case)):

				if case != None:
					string_tester       =  case[index_opinion]['justia_text']
					sentence_arr_helper =  construct_sentence_arr_from_text(string_tester)

					text_arr_helper=[]
					for sen in sentence_arr_helper:
						sen_helper=' '.join(word for word in sen)
						text_arr_helper.append(sen_helper)
					helper_arr	    =  construct_word_arr(text_arr_helper)
					sentence_arr        += helper_arr
		else:
			print ("NONE-Case")
	t2=tm.time()
	np.save("cleaned_data/Sentence_Array_Justia_"+str(year)+".npy",sentence_arr)
	print("------------------")

# Stop instance at end of script
# ec2 = boto3.resource('ec2')
# ec2.Instance('i-0d87649fb25bd8193').stop()



"""
#sentence_lst = remove_stopwords( sentence_arr )
sentence_lst=sentence_arr
sentence_arr=np.array(sentence_lst)
np.save("Sentence_Array_File_2000",sentence_arr)
print_common_words(sentence_lst)
Model= Word2Vec(sentence_lst,min_count=5,size=200)

Model.save("word2vec.model")
"""

"""
print (len(brown.words()))
sentences = brown.sents()
Model= Word2Vec(sentences,min_count=5,size=200)
Model.save("Brown_Model_Small_Size.model")

Model= Word2Vec(sentences,min_count=5,size=200)
"""

print (construct_word_arr(["I'm writing this ''corpora'', and succeeding sets; in a written fashion. Yes this is me!"]))

#test_sentences=random.choices(sent, k=)

#Model=Word2Vec.load("word2vec.model")
#print (Model.predict_output_word(context_words_list=['The','judge', 'make', 'his', 'decision','week'], topn=10))
#Find a small model
