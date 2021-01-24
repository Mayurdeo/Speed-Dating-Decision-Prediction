#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
import copy
import random
pd.options.mode.chained_assignment = None 


# In[2]:


#Current version doesn'tr support
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# In[3]:


# #Load the data set
# testSet=pd.read_csv("testSet.csv",delimiter=',')
# trainingSet=pd.read_csv("trainingSet.csv",delimiter=',')


# In[4]:


# #reading file

trainingSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension : eg.trainingSet.csv ")),delimiter=',') 
testSet=pd.read_csv("{}".format(input("Enter the Test Dataset with csv extension :eg.testSet.csv ")),delimiter=',')   


# In[5]:


class DecisionTree():

    ''' code for a decision tree '''
    
    def __init__(self,trainingSet,testSet,maxDepth=8,exampleLimit=50,decisionVar='decision',vectorised=False):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.maxDepth=maxDepth
        self.exampleLimit=exampleLimit
        self.decisionVar=decisionVar
        self.vectorised=vectorised

    def labelPurity(self,data):

        """This function is to check whether the decision column has any impurities"""

        decisionColumn = data[:, -1]
        uniqueClass = np.unique(decisionColumn)
        if len(uniqueClass) == 1:
            return True
        else:
            return False

    def majorityClass(self,data):

        decisionColumn = data[:, -1]
        uniqueClass, counts = np.unique(decisionColumn, return_counts=True)
        index = counts.argmax()
        label = uniqueClass[index]   
        return label

    def attributeSplit(self,data):

        splits = {}
        _, col = data.shape
        for ind in range(col - 1):         
            values = data[:, ind]
            uniqueList = np.unique(values)    
            splits[ind] = uniqueList   
        return splits

    def dataSplit(self,data, splitCol, splitValue):

        splitColValues = data[:, splitCol]
        dataLeft= data[splitColValues == splitValue]
        dataRight = data[splitColValues != splitValue]    
        return dataLeft, dataRight

    def giniIndex(self,data):

        decisionColumn=data[:,-1]
        _, count = np.unique(decisionColumn, return_counts=True)
        prob = count / count.sum()
        gini=1-np.sum(prob**2)       
        return gini

    def giniGain(self,dataLeft, dataRight):

        n=len(dataLeft)+len(dataRight)
        giniLeft=self.giniIndex(dataLeft)
        giniRight=self.giniIndex(dataRight)
        giniGain=(len(dataLeft)/n)*giniLeft + (len(dataRight)/n)*giniRight
        return giniGain

    def bestSplit(self,data, attrSplits):

        beforeGini = np.inf
        for column_index in attrSplits:
            for value in attrSplits[column_index]:
                dataLeft, dataRight = self.dataSplit(data, splitCol=column_index, splitValue=value)
                afterGini = self.giniGain(dataLeft, dataRight)
                if afterGini <= beforeGini:
                    beforeGini = afterGini
                    bestSplitColumn = column_index
                    bestSplitValue = value   
        return bestSplitColumn, bestSplitValue

    def tree(self,examples, counter=0):

        if counter == 0:
            global featureNames
            featureNames = examples.columns
            data = examples.values
        else:
            data = examples           
        if (self.labelPurity(data)) or (len(data) <self.exampleLimit) or (counter == self.maxDepth):
            leaf = self.majorityClass(data)     
            return leaf
        else:    
            counter += 1
            attrSplits = self.attributeSplit(data)
            split_column, split_value = self.bestSplit(data, attrSplits)
            dataLeft, dataRight = self.dataSplit(data, split_column, split_value)
            if len(dataLeft) == 0 or len(dataRight) == 0:
                leaf = majorityClass(data)
                return leaf
            name = featureNames[split_column]
            query = "{} = {}".format(name, split_value)
            sub_tree = {query: []}
            false = self.tree(dataLeft, counter)
            true = self.tree(dataRight, counter)
            if false == true:
                sub_tree = false
            else:
                sub_tree[query].append(false)
                sub_tree[query].append(true)

            return sub_tree
        
    def predict(self,example, tree):
        query = list(tree.keys())[0]
        name, cp, value = query.split(" ")
        if str(example[name]) == value:
            pred = tree[query][0]
        else:
            pred = tree[query][1]
        if not isinstance(pred, dict):
            return pred
        else:
            root = pred
            return self.predict(example, root)

    def accuracy(self,example, root,key):
        prediction = example.apply(self.predict, args=(root,), axis=1)
        boolean = prediction == example["decision"]
        acc = boolean.mean()   
        acc = np.round(acc*100,decimals=2)
        
        if key=="train":
            print("Training Accuracy DT : {}".format(acc))
        elif key=="test":
            print("Testing Accuracy DT : {}".format(acc))
            
        return acc
            
    def bootstrapPredict(self,example,root):
        prediction = example.apply(self.predict, args=(root,), axis=1)
        return np.array(prediction)
    
    def bootstrapAccuracy(self,example,prediction,key,model=None):
        boolean = prediction == example["decision"]
        acc = boolean.mean()   
        acc = np.round(acc*100,decimals=2)
        
        if model=="bagging":
            if key=="train":
                print("Training Accuracy BT : {}".format(acc))
            elif key=="test":
                print("Testing Accuracy BT : {}".format(acc))
                
        if model=="forests":
            if key=="train":
                print("Training Accuracy RF : {}".format(acc))
            elif key=="test":
                print("Testing Accuracy RF : {}".format(acc))       

        return acc


# In[6]:


class RandomForests(DecisionTree):

    ''' code for random forest '''
    def __init__(self,trainingSet,testSet,maxDepth=8,exampleLimit=50,decisionVar='decision',vectorised=False):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.maxDepth=maxDepth
        self.exampleLimit=exampleLimit
        self.decisionVar=decisionVar
        self.vectorised=vectorised
        
    def randomSplitSplits(self,data):
        randomSplits = {}
        _, nCol = data.shape
        columnInd = list(range(nCol - 1))  
        rnSubspace=int(np.sqrt(nCol))
        if rnSubspace and rnSubspace <= len(columnInd):
            columnInd = random.sample(population=columnInd, k=rnSubspace)
        for column_index in columnInd:          
            values = data[:, column_index]
            unique_values = np.unique(values)
            randomSplits[column_index] = unique_values
        return randomSplits
    
    def forests(self,examples, counter=0):

        if counter == 0:
            global featureNames
            featureNames = examples.columns
            data = examples.values
        else:
            data = examples           
        if (self.labelPurity(data)) or (len(data) <self.exampleLimit) or (counter == self.maxDepth):
            leaf = self.majorityClass(data)     
            return leaf
        else:    
            counter += 1
            attrSplits = self.randomSplitSplits(data)
            split_column, split_value = self.bestSplit(data, attrSplits)
            dataLeft, dataRight = self.dataSplit(data, split_column, split_value)
            if len(dataLeft) == 0 or len(dataRight) == 0:
                leaf = self.majorityClass(data)
                return leaf
            name = featureNames[split_column]
            query = "{} = {}".format(name, split_value)
            sub_tree = {query: []}
            false = self.forests(dataLeft, counter)
            true = self.forests(dataRight, counter)
            if false == true:
                sub_tree = false
            else:
                sub_tree[query].append(false)
                sub_tree[query].append(true)

            return sub_tree


# In[7]:


#To find mode value in a dataframe

def modeValue(x):
    a, b = Counter(x).most_common(1)[0]
    return pd.Series([a, b])


# In[8]:


def decisionTree(trainingSet,testSet):
    DT=DecisionTree(trainingSet,testSet,8,50,"decision",vectorised=True)
    tree=DT.tree(trainingSet)
    DT.accuracy(trainingSet,root=tree,key="train")
    DT.accuracy(testSet,root=tree,key="test")


# In[9]:



def bagging(trainingSet,testSet):
    
 
    trees=[]
    predictTrain=pd.DataFrame()
    predictTest=pd.DataFrame()
    stopCriteria=30
    
    for index in range(stopCriteria):
        examples=trainingSet.sample(frac=1,replace=True)
        Bagging=DecisionTree(examples,testSet,8,50,"decision",vectorised=True)
        tree=Bagging.tree(examples)
        trees.append(tree)
    
    for count,tree in enumerate(trees):
        predTrain=Bagging.bootstrapPredict(trainingSet,tree)
        predTest=Bagging.bootstrapPredict(testSet,tree)
        predictTrain[str(count)]=predTrain
        predictTest[str(count)]=predTest

        
    predictTrain[['frequent','freq_count']] = predictTrain.apply(modeValue, axis=1)
    predictTest[['frequent','freq_count']] = predictTest.apply(modeValue, axis=1)
    
    
    Bagging.bootstrapAccuracy(trainingSet,(predictTrain["frequent"]).values,key="train",model="bagging")
    Bagging.bootstrapAccuracy(testSet,(predictTest["frequent"]).values,key="test",model="bagging")
        


# In[10]:


def randomForests(trainingSet,testSet):
        forests=[]
        rf_predictTrain=pd.DataFrame()
        rf_predictTest=pd.DataFrame()
        stopCriteria=30

        for index in range(stopCriteria):
            examples=trainingSet.sample(frac=1,replace=True)
            RF=RandomForests(examples,testSet,8,50,"decision",vectorised=True)
            rf_tree=RF.forests(examples)
            forests.append(rf_tree)

        for count,trea in enumerate(forests):
            rf_predTrain=RF.bootstrapPredict(trainingSet,trea)
            rf_predTest=RF.bootstrapPredict(testSet,trea)
            rf_predictTrain[str(count)]=rf_predTrain
            rf_predictTest[str(count)]=rf_predTest


        rf_predictTrain[['frequent','freq_count']] = rf_predictTrain.apply(modeValue, axis=1)
        rf_predictTest[['frequent','freq_count']] = rf_predictTest.apply(modeValue, axis=1)


        RF.bootstrapAccuracy(trainingSet,(rf_predictTrain["frequent"]).values,key="train",model="forests")
        RF.bootstrapAccuracy(testSet,(rf_predictTest["frequent"]).values,key="test",model="forests")


# In[11]:


for i in range(3):

    modelIdx=int(input("Enter model index : eg.1 or 2 or 3"))

    if modelIdx==1:
        decisionTree(trainingSet,testSet)
    elif modelIdx==2:
        bagging(trainingSet,testSet)
    elif modelIdx==3:
        randomForests(trainingSet,testSet)
    else:
        print("Invalid Input")