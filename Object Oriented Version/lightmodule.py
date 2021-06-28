#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from lightgbm import LGBMClassifier


# In[3]:


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


# In[ ]:





# In[10]:


class lgbm:
    
    model = LGBMClassifier(application='binary', max_depth=5, min_data_in_leaf=60, n_estimators=150, num_leaves=100)
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model.fit(self.x_train,self.y_train)
        
        self.predictions = self.model.predict(self.x_test)
 
        self.accuracy = accuracy_score(self.y_test,self.predictions)
        self.recall = recall_score(self.y_test,self.predictions)
        self.precision = precision_score(self.y_test, self.predictions)
        self.f2_score = fbeta_score(self.y_test,self.predictions, beta=2.0)
        self.training_accuracy = self.model.score(self.x_train,self.y_train)
        self.test_accuracy = self.model.score(self.x_test,self.y_test)
        
        ## predicted probabilities on training data
        self.pred_proba_train = self.model.predict_proba(self.x_train)
        
        ## predicted probabilities on test data
        self.pred_proba_test = self.model.predict_proba(self.x_test)
        
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
        print('pred_proba for training data:',self.pred_proba_train)
        print('pred_proba for test data:',self.pred_proba_test)
    
    def log_loss_training(self):
        print('log loss on training data:',log_loss(self.y_train, self.pred_proba_train))
    
    def log_loss_testing(self):
        print('log loss on test data:', log_loss(self.y_test, self.pred_proba_test))
    
    def feature_importance(self):
        self.df = pd.DataFrame({"Feature_names":self.x_train.columns,'Importances':self.model.feature_importances_})
        self.df1 = self.df.sort_values(by='Importances',ascending=False).reset_index()
        print(self.df1,'\n')
        self.df2 = self.df1[:10].sort_values(by='Importances',ascending=True)
        plt.figure(figsize=(6,6))
        plt.barh(self.df2['Feature_names'],self.df2['Importances'])
        plt.show()
        plt.tight_layout()
        print('\n')
        
class lgbm_unbalanced:
    
    model = LGBMClassifier(application='binary', max_depth=5, min_data_in_leaf=60, n_estimators=150, num_leaves=100,
                          is_unbalance=True)
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model.fit(self.x_train,self.y_train)
        
        self.predictions = self.model.predict(self.x_test)

        self.accuracy = accuracy_score(self.y_test,self.predictions)
        self.recall = recall_score(self.y_test,self.predictions)
        self.precision = precision_score(self.y_test, self.predictions)
        self.f2_score = fbeta_score(self.y_test,self.predictions, beta=2.0)
        self.training_accuracy = self.model.score(self.x_train,self.y_train)
        self.test_accuracy = self.model.score(self.x_test,self.y_test)
        
        ## predicted probabilities on training data
        self.pred_proba_train = self.model.predict_proba(self.x_train)
        
        ## predicted probabilities on test data
        self.pred_proba_test = self.model.predict_proba(self.x_test)
        
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
        print('pred_proba for training data:',self.pred_proba_train)
        print('pred_proba for test data:',self.pred_proba_test)
    
    def log_loss_training(self):
        print('log loss on training data:',log_loss(self.y_train, self.pred_proba_train))
    
    def log_loss_testing(self):
        print('log loss on test data:', log_loss(self.y_test, self.pred_proba_test))
    
    def feature_importance(self):
        self.df = pd.DataFrame({"Feature_names":self.x_train.columns,'Importances':self.model.feature_importances_})
        self.df1 = self.df.sort_values(by='Importances',ascending=False).reset_index()
        print(self.df1,'\n')
        self.df2 = self.df1[:10].sort_values(by='Importances',ascending=True)
        plt.figure(figsize=(6,6))
        plt.barh(self.df2['Feature_names'],self.df2['Importances'])
        plt.show()
        plt.tight_layout()
        print('\n')
        
if __name__ == "__main__":
    print("lightmodule.py is being run directly")
    
else:
    print("lightmodule.py is being imported into another module")


# In[ ]:




