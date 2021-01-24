#reading file

trainingSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension : eg.trainingSet.csv ")),delimiter=',') 
testSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension :eg.testSet.csv ")),delimiter=',')

class Logistic_Regression():
    
    ''' code for deploying Logistic Regression '''

    def __init__(self,trainingSet,testSet,stepSize,maxIterations,threshold,lamda):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.stepSize=stepSize
        self.maxIterations=maxIterations
        self.threshold=threshold
        self.lamda=lamda
    
    def sigmoid(self,z):
        estimate=1/(1+np.exp(-z))
        return estimate 
        
    def addIntercept(self,dataframe):
        dataframe.insert(loc=0,column="Intercept",value=1)
        return dataframe
    
    def labelSplit(self,dataframe,keyword):
        dataFrameFeatures=dataframe.drop(columns=keyword)
        dataFrameLabels=dataframe[keyword]
        return dataFrameFeatures,dataFrameLabels
     
    def tr(self,trainingSet,testSet):
        trainingSet=self.trainingSet
        testSet=self.testSet
        
        #Seperate Features from labels
        trainFeatures,trainLabels=self.labelSplit(trainingSet,"decision")
        testFeatures,testLabels=self.labelSplit(testSet,"decision")
      
        #Add intercept to the features
        self.addIntercept(trainFeatures)
        self.addIntercept(testFeatures)
        
        #Initialize weight's
        w=np.zeros(trainFeatures.shape[1],dtype="int")
                
        #Initialize count
        count=0
        
        while count<self.maxIterations:
            #Compute z=w^T*x_i using broadcasting technique
            w_T=np.transpose(w)
            y_hat=self.sigmoid(np.sum(w_T*trainFeatures,axis=1))
            
            #Calculating gradient
            gradient=np.sum(np.array(-trainLabels+y_hat)*np.transpose(trainFeatures),axis=1)+(self.lamda*w)
            
            #Updating weights
            w_new=w-(self.stepSize*gradient)
            diff=np.sqrt(np.sum((w_new-w)**2))
            
            if  (diff < self.threshold):
                break
            else:
                count+=1
                w=w_new
                continue
   
        trainDecision=self.sigmoid(np.sum(w*trainFeatures,axis=1))
        trainDecision[trainDecision>0.5]=1
        trainDecision[trainDecision<=0.5]=0
        
        testDecision=self.sigmoid(np.sum(w*testFeatures,axis=1))
        testDecision[testDecision>0.5]=1
        testDecision[testDecision<=0.5]=0
        
        trainResult=np.mean(trainDecision==trainLabels)*100
        testResult=np.mean(testDecision==testLabels)*100
        
        print("Training Accuracy LR: {}".format(np.round(trainResult,2)))
        print("Testing Accuracy LR: {}".format(np.round(testResult,2)))
        
        return gradient


# In[ ]:


classifierLm=Logistic_Regression(trainingSet,testSet,0.01,500,1*np.exp(-6),0.01)