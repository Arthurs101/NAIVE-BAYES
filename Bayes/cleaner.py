import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk import punkt

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#clean the data of the message
# no url, no email address, no numbers ,no symbols
def proccess_message(contents):
    # nltk.download('punkt')
    ps = PorterStemmer()

    contents = contents.lower()
    contents = re.sub(r'<[^<>]+>', ' ', contents)
    contents = re.sub(r'[0-9]+', 'number', contents)
    contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', contents)
    contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', contents)
    contents = re.sub(r'[$]+', 'dollar', contents)
    words = word_tokenize(contents)
    
    for i in range(len(words)):
        words[i] = re.sub(r'[^a-zA-Z0-9]', '', words[i])
        words[i] = ps.stem(words[i])
        
    words = [word for word in words if len(word) >= 1]
    
    return words

def learn_vocabulary(messages):
    vocab = dict()
    for message in messages:
        for word in message:
            if word in vocab.keys():
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab

def getFeatureVector(word_indices, vocab_length):
    feature_vec = np.zeros(vocab_length)
    
    for i in word_indices:
        feature_vec[i] = 1
        
    return feature_vec

def getIndices(email, vocabulary):
    def getKey(dictionary, val):
        for key, value in dictionary.items():
            if value == val:
                return key
        
    word_indices = set()
    
    for word in email:
        if word in vocabulary.values():
            word_indices.add(getKey(vocabulary, word))
    
    return word_indices