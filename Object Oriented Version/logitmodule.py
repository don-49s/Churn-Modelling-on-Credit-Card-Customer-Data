#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression #log regression

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score, fbeta_score
from sklearn.metrics import log_loss
from sklearn import metrics

from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC

from IPython.display import Image  
from six import StringIO  
import graphviz
import pydot 

import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[2]:


class stats_logit:
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = sm.Logit(self.y_train,self.x_train).fit()
        self.yhat = self.model.predict(self.x_test)
        self.predictions = list(map(round,self.yhat))
        
        self.accuracy_score = accuracy_score(self.y_test,self.predictions)
        self.recall_score = recall_score(self.y_test,self.predictions)
        self.f2_score = fbeta_score(self.y_test, self.predictions, beta = 2.0)
#         self.train_accuracy = self.model.score(self.x_train, self.y_train)
#         self.test_accuracy = self.model.score(self.x_test,self.y_test)
        
#         self.train_pred_proba = self.model.predict_proba(self.x_train)
#         self.test_pred_proba = self.model.predict_proba(self.x_test)
#         self.train_log_loss = log_loss(self.y_train,self.train_pred_proba)
#         self.test_log_loss = log_loss(self.y_test, self.test_pred_proba)
        
    def classification_report(self):
        print(classification_report(self.y_test, self.predictions))
    
    def confusion_matrix(self):
        print('Confusion Matrix\n',confusion_matrix(self.y_test,self.predictions))
        
    def KPI_summary(self):
        print('accuracy score: {}'.format(self.accuracy_score))
        print('recall score: {}'.format(self.recall_score))
        print('f2 score: {}'.format(self.f2_score))
#         print('training accuracy: {}'.format(self.train_accuracy))
#         print('testing accuracy: {}'.format(self.test_accuracy))
        
#     def log_loss(self):
#         print('training log loss: {}'.format(self.train_log_loss))
#         print('testing log loss: {}'.format(self.test_log_loss))
        
    def summary(self):
        print(self.model.summary())
        

class sk_logit:
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = LogisticRegression(C=4.281332398719396, solver='newton-cg')
        self.model.fit(self.x_train,self.y_train)
        self.predictions = self.model.predict(self.x_test)
        
        self.accuracy = accuracy_score(self.y_test,self.predictions)
        self.recall = recall_score(self.y_test,self.predictions)
        self.precision = precision_score(self.y_test, self.predictions)
        self.f2_score = fbeta_score(self.y_test,self.predictions, beta=2.0)
        self.training_accuracy = self.model.score(self.x_train,self.y_train)
        self.test_accuracy = self.model.score(self.x_test,self.y_test)
        
        ## predicted probabilities on training data
        self.train_pred_proba = self.model.predict_proba(self.x_train)
        
        ## predicted probabilities on test data
        self.test_pred_proba = self.model.predict_proba(self.x_test)
        
    def classification_report(self):
        print(classification_report(self.y_test, self.predictions))
    
    def confusion_matrix(self):
        print('Confusion Matrix\n',confusion_matrix(self.y_test,self.predictions))
        plot_confusion_matrix(self.model,self.x_test,self.y_test,normalize='true',cmap='bwr')
        plt.show()
        
    def KPI_summary(self):
        print('accuracy score = ',round(self.accuracy*100,2),'%')
        print('recall = ',round(self.recall*100,2),'%')
        print('precision = ',round(self.precision*100,2),'%')
        print('f2_score = ', round(self.f2_score*100,2),'%')
        print('training accuracy = ',round(self.training_accuracy*100,2),'%')
        print('testing accuracy = ',round(self.test_accuracy*100,2),'%')

    def predictive_probabilities(self):
        print('training prediction probability is ',self.train_pred_proba)
        print('testing prediction probability is ',self.test_pred_proba)
    
    def log_loss(self):
        print('log loss on training data is ',log_loss(self.y_train, self.train_pred_proba))
        print('log loss on testing data is ',log_loss(self.y_test,self.test_pred_proba))
        
        
class sk_logit_balanced:
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = LogisticRegression(C=4.281332398719396, solver='newton-cg', class_weight='balanced')
        self.model.fit(self.x_train,self.y_train)
        self.predictions = self.model.predict(self.x_test)
        
        self.accuracy = accuracy_score(self.y_test,self.predictions)
        self.recall = recall_score(self.y_test,self.predictions)
        self.precision = precision_score(self.y_test, self.predictions)
        self.f2_score = fbeta_score(self.y_test,self.predictions, beta=2.0)
        self.training_accuracy = self.model.score(self.x_train,self.y_train)
        self.test_accuracy = self.model.score(self.x_test,self.y_test)
        
        ## predicted probabilities on training data
        self.train_pred_proba = self.model.predict_proba(self.x_train)
        
        ## predicted probabilities on test data
        self.test_pred_proba = self.model.predict_proba(self.x_test)
        
    def classification_report(self):
        print(classification_report(self.y_test, self.predictions))
    
    def confusion_matrix(self):
        print('Confusion Matrix\n',confusion_matrix(self.y_test,self.predictions))
        plot_confusion_matrix(self.model,self.x_test,self.y_test,normalize='true',cmap='bwr')
        plt.show()
        
    def KPI_summary(self):
        print('accuracy score = ',round(self.accuracy*100,2),'%')
        print('recall = ',round(self.recall*100,2),'%')
        print('precision = ',round(self.precision*100,2),'%')
        print('f2_score = ', round(self.f2_score*100,2),'%')
        print('training accuracy = ',round(self.training_accuracy*100,2),'%')
        print('testing accuracy = ',round(self.test_accuracy*100,2),'%')

    def predictive_probabilities(self):
        print('training prediction probability is ',self.train_pred_proba)
        print('testing prediction probability is ',self.test_pred_proba)
    
    def log_loss(self):
        print('log loss on training data is ',log_loss(self.y_train, self.train_pred_proba))
        print('log loss on testing data is ',log_loss(self.y_test,self.test_pred_proba))
        
if __name__ == "__main__":
    print("logitmodule.py is being run directly")
    
else:
    print("logitmodule.py is being imported into another module")
    


# In[ ]:




