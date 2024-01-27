import numpy as np

class NaiveBayes(object):
    def __init__(self):
        pass
    def fitModel(self,X :np.ndarray , Y: np.ndarray , K : int = 1):
        '''
        X,Y must be a numpy array
        X is the independent variable
        Y is the dependent variable ( the result )
        K: laplace smoothing value ( mnust be int , default 1)
        '''
        data , features = X.shape()  #features : columns, data: rows , both being just the amount
        self.classes  = np.unique(Y) #classes : clasifications of data "SPAM", "HAM" , "SCAM", etc...

        #calculate the probabilities of the classes P(Yn)
        self.Py = np.zeros(len(self.classes), dtype=np.float)

        #fit the model

        for index, clas in enumerate(self.classes): 
            #coicidences of the class
            C_coincidences = X[ Y == clas]

            #apply laplace smoothering with the prob of the class (K)
            
            self.Py[index] = ( C_coincidences.shape[0] + K )  /  ( float(data) + K )