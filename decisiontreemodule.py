#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier #decision tree
from sklearn import tree

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,fbeta_score
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


# In[2]:


class decision_tree():
    
    model = DecisionTreeClassifier(criterion = 'entropy',max_depth=8, min_samples_leaf=20)
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
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
    
    def feature_importance(self):
        df = pd.DataFrame({'Feature_names':self.x_train.columns, 'Importances':self.model.feature_importances_})
        df = df.sort_values(by='Importances',ascending=True).reset_index()
        df2 = df.sort_values(by='Importances',ascending=True)
        plt.figure(figsize=(8,6))
        plt.barh(df2['Feature_names'],df2['Importances'])
        df2 = df.sort_values(by='Importances',ascending=True)
        df2.sort_values(by='Importances',ascending=False,inplace=True)
        df2['cumsum'] = df2['Importances'].cumsum(axis=0)
        df2 = df2.reset_index().drop(['level_0','index'],axis=1)
        print(df2)
        plt.figure(figsize= (8,6))
        plt.plot(df2['Feature_names'],df2['cumsum'],linestyle='dashed',marker='+')
        plt.xticks(rotation=80)
        plt.margins(0)
        plt.title('cumulative line chart')
        plt.xlabel('feature names')
        plt.ylabel('importances')
        plt.show()
        
    def plot_tree(self):
        plt.figure(figsize=(20,12))
        tree.plot_tree(self.model, filled=True, rounded=True,class_names=['Existing Customer','Attrited Customer'],
                 feature_names=self.x_train.columns)


class decision_tree_rebalanced():
    
    model = DecisionTreeClassifier(criterion = 'entropy',max_depth=8, min_samples_leaf=20, class_weight='balanced')
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
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
    
    def feature_importance(self):
        df = pd.DataFrame({'Feature_names':self.x_train.columns, 'Importances':self.model.feature_importances_})
        df = df.sort_values(by='Importances',ascending=True).reset_index()
        df2 = df.sort_values(by='Importances',ascending=True)
        plt.figure(figsize=(8,6))
        plt.barh(df2['Feature_names'],df2['Importances'])
        df2 = df.sort_values(by='Importances',ascending=True)
        df2.sort_values(by='Importances',ascending=False,inplace=True)
        df2['cumsum'] = df2['Importances'].cumsum(axis=0)
        df2 = df2.reset_index().drop(['level_0','index'],axis=1)
        print(df2)
        plt.figure(figsize= (8,6))
        plt.plot(df2['Feature_names'],df2['cumsum'],linestyle='dashed',marker='+')
        plt.xticks(rotation=80)
        plt.margins(0)
        plt.title('cumulative line chart')
        plt.xlabel('feature names')
        plt.ylabel('importances')
        plt.show()
        
    def plot_tree(self):
        plt.figure(figsize=(20,12))
        tree.plot_tree(self.model, filled=True, rounded=True,class_names=['Existing Customer','Attrited Customer'],
                 feature_names=self.x_train.columns)
        
        
        
if __name__ == "__main__":
    print("decisiontreemodule.py is being run directly")
    
else:
    print("decisiontreemodule.py is being imported into another module")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




