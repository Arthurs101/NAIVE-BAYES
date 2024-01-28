import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#clean the data of the message
# no url, no email address, no numbers ,no symbols
def proccess_message(message):
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