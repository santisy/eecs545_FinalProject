# -*- coding: utf-8 -*-
"""

Basic PMF without any bias with yelp dataset
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
    featureDim = 10 # dimension of feature vector
    np.random.seed(0) # make controllable initialization
    lambdaU = 0.001 # lambda for user matrix regularization 0.6 0.8, feature dim 10?
    lambdaP = 0.0001 # lambda for item matrix regularization
    step = 0.0004 # iteration step 
    # select training data and testing data
    numFolds = 5 # number of folds of data
    testIdx = 1 #index of testing data
    # iterNum = 250 # number of iterations
    log = [] # iteration log
    maxIterNum = 200 # conrolling number of iterations
 
    # data file path, and read
    method = "Basic MF"
    dataset = "Yelp-shopping"
    categoryName = 'shopping'
    dataPath = "F:\\EECS545Proj\\Yelp_Dataset\\test_" + categoryName + "-TBD\\test_by_distance\\"    
    dataTerms = ['userID','itemID','subcategory','rating','time','userLocation','itemLocation','distance']
    now = datetime.datetime.now()        
    resultPath = "F:\\EECS545Proj\\results\\" + str(now.strftime('%m%d%H%M%S')) + method + dataset + "\\"           
    os.mkdir(resultPath)
    f = open(resultPath + 'parameters.txt','wb')
    f.writelines(["feature demention: ",str(featureDim),'\n',"lambdaU: ",str(lambdaU),'\n',"lambdaP: ",str(lambdaP),'\n',"step: ",str(step),'\n',"number of folds: ",str(numFolds),'\n',"test data index: ",str(testIdx)])
    f.close()
    # read the raw data into pandas dataframe
    flag = 1
    for i in range(1,numFolds + 1):
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
    ratingBias = trainContent['rating'].mean()# calculate rating bias

    # training rating matrix construction
    gc.enable()
    gc.collect()
    RUI = np.zeros([len(userList),len(itemList)], dtype = np.float16)
    for (idx,row) in trainContent.iterrows():
        RUI[userList.index(row['userID']),itemList.index(row['itemID'])] = np.float16(row['rating'])
    print "RUI has been constructed!"
    uLst = RUI.nonzero()[0].tolist()
    iLst = RUI.nonzero()[1].tolist()
    numRating = len(uLst)
    ratingList = zip(uLst,iLst)
    # U,P,BU,BI initialization
    U = np.random.randn(len(userList),featureDim)
    P = np.random.randn(len(itemList),featureDim)
    # this remains to be discussed.

    # iteration
    gc.collect()
    iterNum = 0
    errorOld = 100
    errorNew = 90
    while(errorOld - errorNew >= step):
        iterNum += 1
                
    # took 5 seconds for one teration now. Excited!
        errorOld = errorNew
        errorNew = 0
        count = 0 # for debug
        startTime = time.time()
        for j,k in ratingList:
            #count += 1
            try:
                RUI[j][k] != 0
            except:
                print"RUI == 0, there must be something wrong"
            # calculate gradient
            (Uu,Pi,res) = rupUpdate(ratingBias,U[j,np.newaxis],P[k,np.newaxis],RUI[j][k],lambdaU,lambdaP)
            U[j] = U[j] - step*Uu
            P[k] = P[k] - step*Pi
            errorNew += res/numRating

        iterTime = time.time() - startTime
        line1 =  str(iterNum)+ " iterations have finished!"
        line2 =  "the training error (RMSE) is " + str(float(np.sqrt(errorNew)))
        line3 =  "took " + str(iterTime) + "seconds"
        print line1,line2,line3
        log.append(line1)
        log.append(line2)
        log.append(line3)
        if iterNum >= maxIterNum:
            break
        
        # save intermediate results
        
    numTest = testContent['userID'].count()
    RMSE = 0 # testing error RMSE
    MAE = 0 # testing error MAE
    numInvalid = 0 # number of invalid testing data
    count = 0
    for (idx,row) in testContent.iterrows():
        count += 1            
        testRUI = row['rating']
        try:
            U_hat = U[userList.index(row['userID']),np.newaxis]
            P_hat = P[itemList.index(row['itemID']),np.newaxis]
            RUI_hat = float(np.dot(U_hat,P_hat.T)) + ratingBias
            RMSE += ((testRUI - RUI_hat)**2)
            MAE += abs(testRUI - RUI_hat)
        except:
            numInvalid += 1

    RMSE = np.sqrt(RMSE/(numTest - numInvalid))
    MAE = MAE/(numTest - numInvalid)
    line1 = "Number of iterations: " + str(iterNum)
    line2 = "RMSE for testing data: " + str(RMSE)
    line3 = "MAE for testing data: " + str(MAE)
    line4 = "there are" + str(numInvalid) + " invalid data out of " + str(numTest) + " testing data"
    print line1
    print line2
    print line3
    print line4
    resultLogFID = resultPath + "log.txt"
    f = open(resultLogFID, 'wb')
    f.writelines(log)
    f.writelines([line1,'\n',line2,'\n',line3,'\n',line4])
    f.close()
    recordCSV(P, resultPath + 'P%s.csv' %iterNum) 
    recordCSV(U, resultPath + 'U%s.csv' %iterNum)
        
        
                
            
    
    
        
                    
    
                    
                


