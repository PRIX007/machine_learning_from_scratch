# -*- coding: utf-8 -*-
"""
@author: Priyanshu Kumawat
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
class LinearSVM:
    def __init__(self,C=1):
        self.C=1
    
    def _objective(self,margins):
        return 0.5*self.w.dot(self.w)+self.C*np.maximum(0,1-margins).sum()
    
    def fit(self,X,Y,lr=1e-5,n_iters=400):
        N,D=X.shape
        self.N=N
        self.w=np.random.randn(D)
        self.b=0
        
        #gradient loss 
        losses=[]
        for _ in range(n_iters):
            margins=self._decision_function(X)
            loss= self._objective(margins)
            losses.append(loss)
            
            idx=np.where(margins<1)[0]
            grad_w=self.w-self.C*Y[idx].dot(X[idx])
            self.w-= lr*grad_w
            grad_b=-self.C*Y[idx].sum()
            self.b-=lr*grad_b
            
        self.support_=np.where((Y*self._decision_function(X)))[0]
        print("num SVs:",len(self.support_))
        print("w:",self.w)
        print("b:",self.b)
        
        plt.plot(losses)
        plt.title("LOSS per ITERATION")
        plt.show()
        
    def _decision_function(self,X):
        return X.dot(self.w)+self.b
    
    def predict(self,X):
        return np.sign(self._decision_function(X))
    
    def score(self,X,Y):
         P=self.predict(X)
         return np.mean(Y==P)
  
def medical():
    data=load_breast_cancer()
    X,Y =data.data,data.target
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.33)
    return Xtrain,Xtest,Ytrain,Ytest,1e-3,200

 

#main program

Xtrain,Xtest,Ytrain,Ytest,lr,n_iters=medical()
#target variable be 0 replaced by -1
Ytrain[Ytrain==0]=-1
Ytest[Ytest==0]=-1

scaler=StandardScaler()
Xtrain=scaler.fit_transform(Xtrain)
Xtest=scaler.transform(Xtest)

model=LinearSVM(C=1.0)
model.fit(Xtrain,Ytrain,lr=lr,n_iters=n_iters)
print("our built model performance \n ")
print("train",model.score(Xtrain,Ytrain))
print("test",model.score(Xtest,Ytest))


data=load_breast_cancer()
X1,Y1 =data.data,data.target
Xtrain1,Xtest1,Ytrain1,Ytest1=train_test_split(X1,Y1,test_size=0.33)
inbuiltsvm=SVC(kernel='linear',C=1.0)
inbuiltsvm.fit(Xtrain1,Ytrain1)
P1=inbuiltsvm.predict(Xtrain1)
print("inbuilt model performance \n")
print("train",np.mean(Ytrain1==P1))
P2=inbuiltsvm.predict(Xtest1)
print("test",np.mean(Ytest1==P2))
#we can see that our built svm accuracy is appreciable in train (less than 
#linearSVC) but fortunately equalent in test case 