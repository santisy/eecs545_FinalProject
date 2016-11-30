# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:40:58 2016
Nov. 28 Basic PMF Main script

This implements basic PMF for predicting user ratings
Use five fold data for cross validation
@author: Cheng Ouyang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import gc
from MFNov29Helper import*
import os

if __name__ == "__main__":
    # parameter initialization
    featureDim = 7 # dimension of feature vector
    np.random.seed(0) # make controllable initialization
    lambdaU = 0.02 # lambda for user matrix regularization
    lambdaP = 0.02 # lambda for item matrix regularization
    step = 0.001 # iteration step 
    # select training data and testing data
    numFolds = 5 # number of folds of data
    testIdx = 1 #index of testing data
    iterNum = 2 # number of iterations
 
    # data file path, and read
    categoryName = 'shopping'
    dataPath = "F:\\EECS545Proj\\Yelp_Dataset\\test_" + categoryName + "-TBD\\test_by_distance\\"    
    dataTerms = ['userID','itemID','subcategory','rating','time','userLocation','itemLocation','distance']
    now = datetime.datetime.now()        
    resultPath = "F:\\EECS545Proj\\results\\" + str(now.strftime('%m%d%H%M')) + "\\"           
    os.mkdir(resultPath)
    # read the raw data into pandas dataframe
    flag = 1
    for i in range(1,3):
        dataFID = dataPath + 'data' + str(i) + '.txt'
        if i!= testIdx:
            if flag == 1:
                 trainContent = pd.read_csv(dataFID, delimiter = ';')
                 trainContent.columns = dataTerms
                 flag = 0
            else:
                 content = pd.read_csv(dataFID, delimiter = ';')
                 content.columns = dataTerms
                 trainContent = pd.concat([trainContent,content])
        else:
            testContent = pd.read_csv(dataFID, delimiter = ';')
            testContent.columns = dataTerms
    
    # training user ID List generation 
    userList = pd.unique(trainContent['userID']).tolist()
    userNum = len(userList)
    # training item ID generation
    itemList = pd.unique(trainContent['itemID']).tolist()
    itemNum = len(itemList)
    # training rating matrix construction
    gc.enable()
    gc.collect()
    RUI = np.zeros([len(userList),len(itemList)], dtype = np.float16)
    for (idx,row) in trainContent.iterrows():
        RUI[userList.index(row['userID']),itemList.index(row['itemID'])] = np.float16(row['rating'])
    print "RUI has been constructed!"
    
    # U,P initialization
    U = np.random.randn(len(userList),featureDim)
    P = np.random.randn(len(itemList),featureDim)
    # iteration
    numRating = np.sum(np.sign(RUI))
    for i in range(1,iterNum): #use a more rational way, update step by step 
    # took 220 seconds for one iteration. Try to accelerate by parallel
        error = 0
        startTime = time.time()
        for j in range(0,userNum):
            for k in range(0,itemNum):
                if RUI[j][k] != 0:
                    # update P,U
                    (Uu,Pi,res) = upUpdate(U[j,np.newaxis],P[k,np.newaxis],RUI[j][k],lambdaU,lambdaP)
                    U[j] = U[j] - step*Uu
                    P[k] = P[k] - step*Pi
                    error += res/numRating
        iterTime = time.time() - startTime
        print "I have finished one iteration!"
        print "the MAE for objective function is", error
        print "take", str(iterTime), "seconds"
        # save intermediate results
        recordCSV(P, resultPath + 'P%s.csv' %i) 
        recordCSV(U, resultPath + 'U%s.csv' %i)  
        
    numTest = testContent['userID'].count()
    MSE = 0 # testing error MSE
    MAE = 0 # testing error MAE
    numInvalid = 0 # number of invalid testing data
    count = 0
    for (idx,row) in testContent.iterrows():
        count += 1            
        testRUI = row['rating']
        try:
            U_hat = U[userList.index(row['userID']),np.newaxis]
            P_hat = P[itemList.index(row['itemID']),np.newaxis]
            RUI_hat = float(np.dot(U_hat,P_hat.T))
            MSE += ((testRUI - RUI_hat)**2)
            MAE += abs(testRUI - RUI_hat)
        except:
            numInvalid += 1
            print "invalid testing data detected!"
            
        if count/1000 == 0:
            print str((count/numTest)*100) + "%"
    MSE = np.sqrt(MSE/(numTest - numInvalid))
    MAE = MAE/(numTest - numInvalid)
    line1 = "Number of iterations: " + str(i)
    line2 = "MSE for testing data: " + str(MSE)
    line3 = "MAE for testing data: " + str(MAE)
    print line1
    print line2
    print line3
    resultLogFID = resultPath + "log.txt"
    f = open(resultLogFID, 'wb')
    f.writelines([line1,'\n',line2,'\n',line3])
    f.writelines(["feature demention: ",str(featureDim),'\n',"lambdaU: ",str(lambdaU),'\n',"lambdaP: ",str(lambdaP),'\n',"step: ",str(step)])
    f.close()
        
        
                
            
    
    
        
                    
    
                    
                


