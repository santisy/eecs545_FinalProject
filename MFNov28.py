# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:40:58 2016
Nov. 28 Basic PMF Main script

This implements basic PMF for predicting user ratings
Use five fold data for cross validation
@author: Cheng Ouyang
"""
import numpy as np
import cudamat as cm
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import gc
from MFNov29Helper import*
import os

if __name__ == "__main__":
    # parameter initialization
    featureDim = 20 # dimension of feature vector
    np.random.seed(0) # make controllable initialization
    lambdaU = 1 # lambda for user matrix regularization
    lambdaP = 1 # lambda for item matrix regularization
    step = 0.0002 # iteration step
    # select training data and testing data
    numFolds = 5 # number of folds of data
    testIdx = 1 #index of testing data
    iterNum = 100 # number of iterations
 
    # data file path, and read
    categoryName = 'Shopping'
    dataPath = os.path.expanduser('~/Documents/Yelp_Dataset/test_') + categoryName + "-TMC/test_by_distance/"
    dataTerms = ['userID','itemID','subcategory','rating','time','userLocation','itemLocation','distance']
    now = datetime.datetime.now()        
    resultPath = os.path.expanduser('~/Documents/results/') + str(now.strftime('%m%d%H%M')) + '/'
    if not os.path.isdir(resultPath):
        os.mkdir(resultPath)
    # read the raw data into pandas dataframe
    flag = 1
    for i in range(1, numFolds+1):
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
    itemNum = len(itemList) # training rating matrix construction
    gc.enable()
    gc.collect()
    RUI = np.zeros([len(userList),len(itemList)], dtype = np.float16)
    for (idx,row) in trainContent.iterrows():
        RUI[userList.index(row['userID']),itemList.index(row['itemID'])] = np.float16(row['rating'])
    print("RUI has been constructed!")
    uLst = RUI.nonzero()[0].tolist()
    iLst = RUI.nonzero()[1].tolist()
    numRating = len(uLst)
    ratingList = tuple(zip(uLst, iLst))

    # U,P initialization
    U = np.random.randn(len(userList),featureDim)
    P = np.random.randn(len(itemList),featureDim)
    # iteration
    # numRating = np.sum(np.sign(RUI))
    gc.collect()
    for i in range(1, iterNum):  # use a more rational way, update step by step
        # took 5 seconds for one teration now. Excited!
        error = 0
        count = 0  # for debug
        startTime = time.time()
        print(100)
        for j, k in ratingList:
            # count += 1
            try:
                RUI[j][k] != 0
            except:
                print("RUI == 0, there must be something wrong")
            (Uu, Pi, res) = upUpdate(U[j, np.newaxis], P[k, np.newaxis], RUI[j][k], lambdaU, lambdaP)
            U[j] = U[j] - step * Uu
            P[k] = P[k] - step * Pi
            error += res / numRating
        iterTime = time.time() - startTime
        print("I have finished one iteration!")
        print("the MAE for objective function is", error)
        print("take", str(iterTime), "seconds")
        # save intermediate results
        #recordCSV(P, resultPath + 'P%s.csv' %i)
        #recordCSV(U, resultPath + 'U%s.csv' %i)
        
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
            #print("invalid testing data detected!")
            
        if count/1000 == 0:
            print(str((count/numTest)*100) + "%")
    MSE = np.sqrt(MSE/(numTest - numInvalid))
    MAE = MAE/(numTest - numInvalid)
    line1 = "Number of iterations: " + str(i)
    line2 = "MSE for testing data: " + str(MSE)
    line3 = "MAE for testing data: " + str(MAE)
    print(line1)
    print(line2)
    print(line3)
    resultLogFID = resultPath + "log.txt"
    f = open(resultLogFID, 'w')
    f.write(line1 + '\n' + line2 + '\n' + line3 + '\n')
    f.write("feature demention: " + str(featureDim) + '\n' +  "lambdaU: " + str(lambdaU) + '\n' + "lambdaP: " + str(lambdaP) + '\n' + "step: " + str(step) + '\n')
    f.close()
        
        
                
            
    
    
        

