#!/usr/bin/env python
# coding: utf-8

# In[ ]:


trainingSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension : eg.trainingSet.csv ")),delimiter=',') 
testSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension :eg.testSet.csv ")),delimiter=',')


class Support_Vector_Machines():

    ''' code for support vector machines '''
    
    def __init__(self,trainingSet,testSet,stepSize,maxIterations,threshold,lamda):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.stepSize=stepSize
        self.maxIterations=maxIterations
        self.threshold=threshold
        self.lamda=lamda
    
    def addIntercept(self,dataframe):
        dataframe.insert(loc=0,column="Intercept",value=1)
        return dataframe 
    
    def labelSplit(self,dataframe,keyword):
        dataFrameFeatures=dataframe.drop(columns=keyword)
        dataFrameLabels=dataframe[keyword]
        return dataFrameFeatures,dataFrameLabels
       
    def svm(self,trainingSet,testSet):
        trainingSet=self.trainingSet
        testSet=self.testSet
        
        #Seperate Features from labels
        trainFeatures,trainLabels=self.labelSplit(trainingSet,"decision")
        testFeatures,testLabels=self.labelSplit(testSet,"decision")
      
        #Add intercept to the features
        self.addIntercept(trainFeatures)
        self.addIntercept(testFeatures)
        
        #Map labels to (-1,1)
        trainLabels[trainLabels==0]=-1
        testLabels[testLabels==0]=-1
        
        #Initialize weight's
        w=np.zeros(trainFeatures.shape[1],dtype="int")
                
        #Initialize count
        count=0
        
        while count<self.maxIterations:
            #Compute z=w^T*x_i using broadcasting technique
            w_T=np.transpose(w)
            
            #Broadcast to find y_estimate
            y_hat=np.sum(w_T*trainFeatures,axis=1)
            
            #Map y_estimates to [-1,1]
            y_hat[y_hat>0]=1
            y_hat[y_hat<=0]=-1
            
            hinge=y_hat*trainLabels
            hinge[hinge>=1]=0
            
            #Calculate delta:
            delta=np.sum(trainLabels*np.transpose(trainFeatures),axis=1)
 
            #Add regularization
            reg=self.lamda*w
            
            #Calculate gradient
            gradient=(1/trainFeatures.shape[0])*(reg-delta)
                 
            #Updating weights
            
            w_new=w-(self.stepSize*gradient)
            diff=np.sqrt(np.sum((w_new-w)**2))
            
            if  (diff < self.threshold):
                break
            else:
                count+=1
                w=w_new
                continue

        trainDecision=np.sum(w*trainFeatures,axis=1)
        trainDecision[trainDecision>0]=1
        trainDecision[trainDecision<=0]=-1

        testDecision=np.sum(w*testFeatures,axis=1)
        testDecision[testDecision>0]=1
        testDecision[testDecision<=0]=-1

        trainResult=np.mean(trainDecision==trainLabels)*100
        testResult=np.mean(testDecision==testLabels)*100

        print("Training Accuracy SVM: {}".format(np.round(trainResult,2)))
        print("Testing Accuracy SVM: {}".format(np.round(testResult,2)))
        
        return 


# In[ ]:


classifierSvm=Support_Vector_Machines(trainingSet,testSet,0.01,500,1*np.exp(-6),0.01)
