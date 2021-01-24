#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
from collections import Counter 
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#reading file
trainingSet=pd.read_csv("trainingSet.csv",delimiter=',') 
testSet=pd.read_csv("testSet.csv",delimiter=',') 


# In[13]:


class  classifier():

    ''' code for deploying naive bayes classifier '''
    
    def __init__(self,train,test):
        self.trainingSet=train
        self.testSet=test
    
    #nbc function gives us sampling of the train data set
    def nbc(self,tfrac):
        self.trainingSample = self.trainingSet.sample(frac=tfrac,random_state=47)
    
    def probability_prior(self,response):
        
        #calculate total number of positive and negative decisions resp.
        total_yes=np.sum([self.trainingSample[str(response)]==1])
        total_no=np.sum([self.trainingSample[str(response)]==0])
        
        #calculate priors of yes and no respectively(P(yes),P(no))
        prior_yes=total_yes/(total_yes+total_no)
        prior_no=total_no/(total_yes+total_no)
        
        return total_yes,total_no,prior_yes,prior_no
        #return prior_yes,prior_no
        
        
    def train(self):
        
        #get the column names list 
        columnList=list(self.trainingSample.columns)
        
        condProb_yes={}
        condProb_no={}
          
        total_yes,total_no,prior_yes,prior_no=self.probability_prior("decision")
        
        for name in columnList:
            
            #Find distinct values in each column:
            uniqueList=list(np.sort(self.trainingSample[name].unique()))
            
            #Store P(x_i|yes),P(x_i|no)
            condProb_yes[name]={}
            condProb_no[name]={}
            
            #Finding prob of each distinct values:
            for num in uniqueList:
                condProb_yes[name][num]= len(self.trainingSample[name][ (self.trainingSample[name]==num) & (self.trainingSample["decision"]==1)]) / total_yes
                
                condProb_no[name][num]= len(self.trainingSample[name][ (self.trainingSample[name]==num) & (self.trainingSample["decision"]==0)]) / total_no
            
            
            #Laplace smoothing:
            for i in uniqueList:
                if condProb_yes[name][i]==0:
                    for num in uniqueList:
                        condProb_yes[name][num]= (len(self.trainingSample[name][ (self.trainingSample[name]==num) & (self.trainingSample["decision"]==1)]) + 1)/ (total_yes+len(uniqueList))
                    break
                
            for i in uniqueList:
                if condProb_no[name][i]==0:
                    for num in uniqueList:
                        condProb_no[name][num]= (len(self.trainingSample[name][ (self.trainingSample[name]==num) & (self.trainingSample["decision"]==0)]) + 1)/ (total_no+len(uniqueList))
                    break
                        
            
        return condProb_yes , condProb_no   
    
    
    def predict(self,dataset,res):
        if dataset=="testSet":dataset=self.testSet
        elif dataset=="trainingSet":dataset=self.trainingSet
        else:dataset=self.trainingSample
            
        condProb_yes , condProb_no=self.train()    
        
        total_yes,total_no,prior_yes,prior_no=self.probability_prior("decision")
            
        decision=[]
        
        #get the column names list 
        columnList=list(dataset.columns)[:-1]
        
        for row in range(len(dataset)):
            
            #Initialise likelihood_yes,likelihood_no to 1
            like_yes=1
            like_no=1
            
            #Take each row for prediction
            row_data=dataset.iloc[row][:-1]
            
            #Calculate likelihood
            for name in columnList:
                
                #If the observation is unseen:
                
                if row_data[name] not in (condProb_yes[name].keys()):
                    condProb_yes[name][(row_data[name])]=1/(total_yes+len(list(np.sort(dataset[name].unique()))))
                
                if row_data[name] not in (condProb_no[name].keys()):
                    condProb_no[name][(row_data[name])]=1/(total_no+len(list(np.sort(dataset[name].unique()))))
                
                #Capital phi calculation
                
                like_yes=(like_yes)*(condProb_yes[name][(row_data[name])])
                like_no=(like_no)*(condProb_no[name][(row_data[name])])
           
            
            #Multiplying prior values
            
            like_yes=like_yes*prior_yes
            like_no=like_no*prior_no
            
            decision_yes=like_yes/(like_yes+like_no)
            decision_no=like_no/(like_yes+like_no)
            
            
            #Taking argmax
            if decision_yes>decision_no:
                decision.append(1)
            else:
                decision.append(0)
            
            
            
        result=np.mean(decision==dataset["decision"])
        
        if (res=="test"):
            print("Testing Accuracy: {}".format(np.round(result*100),2))
        else:
            print("Training Accuracy: {}".format(np.round(result*100),2))
            
        return result
    


# In[14]:


Naive_bayes=classifier(trainingSet,testSet)
Naive_bayes.nbc(1)


# In[15]:


result=Naive_bayes.predict("trainingSample","train")
result=Naive_bayes.predict("testSet","test")
