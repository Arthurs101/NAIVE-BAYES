import numpy as np

class NaiveBayes(object):
    def __init__(self):
        #dictionari for the features given Y, ex : P("sandwitch" | "spam")
        self.factsPs = {}
    def fitModel(self,X :np.ndarray , Y: np.ndarray , K : int = 1):
        '''
        X,Y must be a numpy array
        X is the independent variable
        Y is the dependent variable ( the result )
        K: laplace smoothing value ( mnust be int , default 1)
        '''
        data , features = X.shape  #features : columns, data: rows , both being just the amount
        self.classes  = np.unique(Y) #classes : clasifications of data "SPAM", "HAM" , "SCAM", etc...

        #calculate the probabilities of the classes P(Yn)
        self.Py = np.zeros(len(self.classes), dtype=float)

        #fit the model

        for index, clas in enumerate(self.classes):  
            #calculate the prior probability of the classes P(Yn) 
            coincidences = int(np.sum(Y == clas))
            self.Py[index] = ( coincidences + K)  / (Y.size + K*self.classes.size)
            
            #calculate the P(Xn|Y) for the classes
            X_c = X[Y ==  clas]  #only the X's where the Y is equal to the actual class
