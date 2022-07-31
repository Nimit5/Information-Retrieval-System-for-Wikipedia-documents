import os
import re
import string
import nltk
import unidecode
from nltk import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader


# Root folder where the text files are located
corpus_root = 'english-corpora'

# Read the list of files
filelists = PlaintextCorpusReader(corpus_root, '.*')

os.mkdir('english-corpora-processed') 
# Root folder where the cleaned files are to be located
corpus_root_cleaned = 'english-corpora-processed'


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


for file in filelists.fileids():
    s=corpus_root+"/"+str(file)
    s2=corpus_root_cleaned+"/"+str(file)
    with open(s2, 'w') as f2,open(s, 'r') as f1:
        text = f1.read()
        text=text_preprocess(text)
        f2.writelines(text)
    
