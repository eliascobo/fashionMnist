#Elias Daniel Cobo Medvedsky
#Comodoro Rivadavia
#30/09/2019

import numpy as np

class modelFashionMnist:
    def __init__(self, bins=256, model_type="A"):
        self.bins = bins
        self.model_type=model_type
        
    def fit(self, images, labels):
        self.probability_ = list()
        self.priori_ = list()
        for cat in range(len(set(labels))):
            hist = self.gethist(images, labels, cat) 
            self.probability_.append(hist[0])
            self.priori_.append(hist[1])
        self.probability_ = np.array(self.probability_).T
        self.priori_ =np.array(self.priori_)
        
    def gethist(self, images, labels, category):
        priori = (labels==category).sum()/len(labels) #para que a funcion sea mas generica
        hist = np.histogram(images[labels==category].flatten(),
                            bins=self.bins, 
                            range=[0,256], 
                            density=True)
        '''1 log probabilidades, 2 log priori'''
        return np.log(hist[0]), np.log(priori)
    
    def predict(self, images):
        self.hists_ = list()
        for image in images:
            self.hists_.append(np.histogram(image.flatten(), 
                                bins=self.bins, 
                                range=[0,256], 
                                density=False)[0])
            
        self.hists_ = np.array(self.hists_)
        self.dot_ = np.dot(self.hists_,self.probability_) + self.priori_
            
        return np.argmax(self.dot_, axis=1)

    
    
    def score(self, images,labels):
        predictions = self.predict(images)
        compare = np.equal(predictions, labels)
        nonzero = np.count_nonzero(compare)
        return nonzero/len(labels)