import numpy as np

class NaiveBayes(object): 
    def __init__(self,vocab):
        #dictionari for the features given Y, ex : P("sandwitch" | "spam")
        self.factsPs = []
        self.vocabulary = vocab
    def fitModel(self,X :np.ndarray , Y: np.ndarray , K : int = 1):
        '''
        X,Y must be a vocabulary
        X is the independent variable
        Y is the dependent variable ( the result )
        K: laplace smoothing value ( mnust be int , default 1)
        '''
        self.classes  = np.unique(Y) #classes : clasifications of data "SPAM", "HAM" , "SCAM", etc...

        #calculate the probabilities of the classes P(Yn)
        self.Py = np.zeros(len(self.classes), dtype=float)

        #fit the model

        for index, clas in enumerate(self.classes):  
            #calculate the prior probability of the classes P(Yn) 
            coincidences = int(np.sum(Y == clas))
            self.Py[index] = ( coincidences + K)  / (len(Y) + K*self.classes.size)
            
            #calculate the P(Xn|Y) for the classes
            X_c = X[Y ==  clas]  #only the X's with the same index where the Y is equal to the actual class
            
            #get the values of all the P(Xi1|class)
            class_amount_vocab = {}
            
            for Xi in X_c:
                for word in Xi:
                    if word in class_amount_vocab.keys():
                        class_amount_vocab[word] += 1
                    else: 
                        class_amount_vocab[word] = 1

            #calculate all the P(Xi1|class) with laplace smoothing
            #get all the observations
            class_obs = sum(class_amount_vocab.values())
            #calculate the probabilities
            PX_c = {}
            for key in self.vocabulary.keys(): 
                if key not in class_amount_vocab.keys():
                    PX_c[key] = (K) / (class_obs + K*len(self.vocabulary.keys()))
                else:
                    PX_c[key] = (class_amount_vocab[key] + K) / (class_obs + K*len(self.vocabulary.keys()))
            
            #add the probabilities to the class 
            self.factsPs.append(PX_c)

    def predict(self,X :np.ndarray):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,x :np.ndarray):
        #probabilities of all the classes given a x input
        #returns the class with highest probability
        posterior = []
        for index, clas in enumerate(self.classes):
            probs = self.factsPs[index]
            pX_I = self.Py[index]
            for word in x:
                pX_I *= probs[word]
            posterior.append(pX_I)
        return self.classes[np.max(posterior)]
                


