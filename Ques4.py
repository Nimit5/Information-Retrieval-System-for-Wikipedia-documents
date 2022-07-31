import os
import sys
import pickle
import re
import math
import csv 
import string
import unidecode
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import PlaintextCorpusReader


# Function for text cleaning and preprocessing

def text_preprocess(text):
    text=text.lower()
    
    # Removing new lines and tabs
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    
    # Removing_extra whitespace
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')

    # Converting unicode data into ASCII characters. 
    text = unidecode.unidecode(text)
    
    # Removing links
    remove_https = re.sub(r'http\S+', '', text)
    text = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    
    # Removing Code
    text=re.sub(r' {[^}]*}','',text) 
    
    # Removing wikipedia references 
    text = re.sub("\[[0-9]+\]", ' ', text)
    
    # Removing punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Remove_whitespace
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')

    # Removing stopwords
    stoplist = stopwords.words('english') 
    stoplist = set(stoplist)
    text = repr(text)
    
    # Text without stopwords
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    
    # Convert list of tokens_without_stopwords to String type.
    words_string = ' '.join(No_StopWords)    
    
    tokens_words = nltk.word_tokenize(words_string)
    ps = PorterStemmer()
    
    word_list = nltk.word_tokenize(words_string)
    text = ' '.join([ps.stem(w) for w in word_list])
    
    return text 


# Root folder where the cleaned text files to be located
corpus_root = 'english-corpora-processed'


# Read the list of files
filelists = PlaintextCorpusReader(corpus_root, '.*')

# List down the IDs of the files read from the local storage
doc_list=filelists.fileids()


corpus_size=len(filelists.fileids())


# loading pickels file of posting list
with open('posting_dict.pkl', 'rb') as f:
    loaded_posting_dict = pickle.load(f)

# loading pickels file of list containing document length
with open('doc_len_list.pkl', 'rb') as f:
    loaded_doc_len_list = pickle.load(f)


# loading pickels file of global matrix
with open('global_matrix.pkl', 'rb') as f:
    loaded_global_matrix = pickle.load(f)


# loading pickels file of list containing normalised document length
with open('doc_normalised_length.pkl', 'rb') as f:
    loaded_doc_normalised_length=pickle.load(f)


unique_words_all = len(loaded_posting_dict)

# dictionary to store queries
queries={}

# taking input from command line
query_file_name=sys.argv[1]

with open(query_file_name, 'rb') as f2:
    Lines = f2.readlines()
    for line in Lines:
        l1=line.strip()
        str1 = l1.decode('UTF-8') 
        temp=str1.split("\t")
        queries[temp[0]]=temp[1]


# preprocessing and cleaning of query

for item in queries:
    query=text_preprocess(queries[item])
    query=word_tokenize(query[1:-1])
    queries[item]=query


# Boolean System: 
# input - list of terms present in a query 
# output- list of all relevant docId

def boolean_query(query):
    
    boolean_result = []
    binary_array_per_word=[]
    binary_array_all_words=[]
    
    for term in query:
        if term in loaded_posting_dict:
            binary_array_per_word=[0]*corpus_size
            temp_dict=loaded_posting_dict[term]
            
            for item in temp_dict:
                temp2=doc_list.index((str(item)+".txt"))
                binary_array_per_word[temp2]=1
            
            binary_array_all_words.append(binary_array_per_word)
        else:
            print(word," not found")
    
    for j in range(len(query)-1):
        word1=binary_array_all_words[0]
        word2=binary_array_all_words[1]

        bitwise_op = [w1 & w2 for (w1,w2) in zip(word1,word2)]
            
        binary_array_all_words.remove(word1)
        binary_array_all_words.remove(word2)
        binary_array_all_words.insert(0,bitwise_op)
  
    res = binary_array_all_words[0]
    cnt = 0
    
    for item in res:
        if item==1:
            boolean_result.append(doc_list[cnt])
        cnt = cnt+1
    
    return boolean_result


# Writing output into Qrel's file

rows=[]
for item in queries:
    
    bool_op=boolean_query(queries[item])[:5]
    
    for res in bool_op:
        rows.append([item,1,res.replace('.txt',''),1])
         
    # writing to csv file 
    with open('Boolean_Qrel.txt', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)  
        
        # writing the data rows 
        csvwriter.writerows(rows)


# TF-IDF System: 
# input - list of terms present in a query 
# output- list of all relevant docId in decreasing order of score 


def tf_idf_score(query):
    len_query=len(query)
    tf_query={}
    arr=np.zeros((len_query+1,corpus_size))
    for i in range(len_query):
        try:
            tf_query[query[i]]+=1
        except:
            tf_query[query[i]]=1
            
        temp20 = loaded_posting_dict[query[i]]
        
        for item in temp20:
            temp=item
            temp2=doc_list.index((str(temp)+".txt"))
            arr[i][temp2]=(temp20[item])
            arr[-1][temp2]+=arr[i][temp2]**2
    
    
    # Processing Query
    query_square=0
    for item in tf_query:
        
        tf_query[item]=tf_query[item]*(np.log(corpus_size/len(loaded_posting_dict[item]))) #need to handle this case when item is not in dict
        query_square+=tf_query[item]**2
    

    q_arr=np.zeros(len_query)
    
    y=0
    for z in tf_query:
        q_arr[y]=tf_query[z]
        y+=1
    
    # Dot product
    
    doc_score={}
    for i in range(corpus_size):
        s=0
        for j in range(len_query):
            s+=(arr[j][i]*q_arr[j])
        doc_score[doc_list[i]]=s
    doc_score = {k: v for k, v in sorted(doc_score.items(), key=lambda x: x[1],reverse=True)}
    return doc_score


# Writing output into Qrel's file

rows1=[]

for item in queries:
    
    tf_idf_op=list(tf_idf_score(queries[item]).keys())[:5]
    
    for res in tf_idf_op:
        rows1.append([item,1,res.replace('.txt',''),1])
    
    # writing to csv file 
    with open('TfIdf_Qrel.txt', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)  
        
        # writing the data rows 
        csvwriter.writerows(rows1)


# BM25 System: 
# input - list of terms present in a query 
# output- list of all relevant docId in decreasing order of score 

def BM25(query,k1=1.2,b=0.75):
    
    bm25_scores = {}
    #calculating avg_doc_len value
    avg_doc_len_ = sum(loaded_doc_len_list) / corpus_size
    
    for index in range(corpus_size):
        scr = 0.0
        doc_len = loaded_doc_len_list[index]
        term_frequency = loaded_global_matrix[index]

        for term in query:
            if term not in term_frequency:
                continue
            
            count=len(loaded_posting_dict[term])
            freq = term_frequency[term]
            idf_val=np.log(1 + (corpus_size - count + 0.5) / (count + 0.5))
            num = idf_val * freq * (k1 + 1)
            deno = freq + k1 * (1 - b + b * doc_len / avg_doc_len_)
            scr += (num/deno)
        
        bm25_scores[doc_list[index]]=scr
    
    final_ans=dict(sorted(bm25_scores.items(), key=lambda item: item[1],reverse=True))
    return final_ans


# Writing output into Qrel's file

rows2=[]
for item in queries:
    
    bm25_op=list(BM25(queries[item]).keys())[:5]
    for res in bm25_op:
        rows2.append([item,1,res.replace('.txt',''),1])
    
    # writing to csv file 
    with open('BM25_Qrel.txt', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)  
        
        # writing the data rows 
        csvwriter.writerows(rows2)

