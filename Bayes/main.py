import pandas as pd
from cleaner import *
from sklearn.feature_extraction.text import CountVectorizer
from model import NaiveBayes
import os

#read the file using pandas
df = pd.read_csv(os.path.join(os.getcwd(), "Bayes", "entrenamiento.txt"),delimiter="\t", header=None , names=["status","message"])
features_vector = CountVectorizer()

messages = df["message"].tolist()
messages = list(map(lambda x: proccess_message(x), messages))
#extract all the words and its amount
vocabulary = learn_vocabulary(messages)
#clean the messages

#calculate the train index
train_index = int(len(df)*0.8)
vocab_length = len(list(vocabulary.keys()))
X = messages
Y = df["status"].tolist()
agent = NaiveBayes(vocabulary)
Xt = X[:train_index]
Yt = Y[:train_index]
Xt = np.array(Xt, dtype=object)
Yt = np.array(Yt, dtype=object)
agent.fitModel(Xt, Yt)
# print(X) 

