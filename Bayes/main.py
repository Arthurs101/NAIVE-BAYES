import pandas as pd
from cleaner import *
from sklearn.feature_extraction.text import CountVectorizer
from model import NaiveBayes


#read the file using pandas
df = pd.read_csv("entrenamiento.txt",delimiter="\t", header=None , names=["status","message"])
features_vector = CountVectorizer()
#extract the features as a CountVectorizer
messages = df["message"].tolist()
messages = proccess_message(messages)
#clean the messages

#calculate the train index
train_index = int(len(df)*0.8)

X_train = X.loc[0:train_index]
Y_train = df["status"].loc[0:train_index]

X_train = X_train.values
Y_train = Y_train.values

agent = NaiveBayes()

# agent.fitModel(X_train, Y_train)
# print(X) 

