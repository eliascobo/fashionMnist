#Elias Daniel Cobo Medvedsky
#Comodoro Rivadavia
#01/10/2019

import numpy as np
from scipy.stats import norm

class gaussModelFashion:
       
    def __init__(self,e=1e-8):
        self.e = e
        
    def fit(self, images, labels):
        self.means = list() #np.zeros(11)
        self.stds = list() #np.zeros(11)
        self.cantclass = len(set(labels))

        for i in range(self.cantclass):
            means = np.mean(images[labels==i])
            stds = np.std(images[labels==i]) + self.e 
            self.means.append(means)
            self.stds.append(stds)
            
    def predict(self, images):
        sumcat = list()
        for i in range(self.cantclass):
            #funcion de densidad de probabilidad (PDF)
            pdf = norm(self.means[i], self.stds[i])
            sumcat.append(np.log(pdf.pdf(images)).sum(axis=1)) # valor que tiene cada imagen
        sumcat = np.array(sumcat)
        #print(sumcat.shape) # 10 categorias x n cant de imagenes
        return np.argmax(sumcat, axis=0) #devuelve a que clase pertence la imagen
                
    def predictFor1(self, image):
        sumcat = list()
        for i in range(self.cantclass):
            #funcion de densidad de probabilidad (PDF)
            pdf = norm(self.means[i], self.stds[i])
            sumcat.append(np.log(pdf.pdf(image.flatten())).sum())  
        return np.argmax(sumcat) #devuelve a que clase pertence la imagen
    
    def score(self, images,labels):
        predictions = self.predict(images)
        compare = np.equal(predictions, labels)
        nonzero = np.count_nonzero(compare)
        return nonzero/len(labels)