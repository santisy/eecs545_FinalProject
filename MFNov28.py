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
import threading
import Queue
import scipy.sparse as sps
import time
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
    testIdx = 1 #index of testing
    iterNum = 5 # number of iterations
    
    # data file path, and read
    categoryName = 'shopping'
    dataPath = "F:\\EECS545Proj\\Yelp_Dataset\\test_" + categoryName + "-TBD\\test_by_distance\\"
    
    dataTerms = ['userID','itemID','subcategory','rating','time','userLocation','itemLocation','distance']
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
                    (Uu,Pi,res) = upUpdate(U[j,np.newaxis],P[k,np.newaxis],RUI[j][k],lambdaU,lambdaP)
                    U[j] = U[j] - step*Uu
                    P[k] = P[k] - step*Pi
                    error += res/numRating
            print j
        iterTime = time.time() - startTime
        print "I have finished one iteration!"
        print "the MAE is", error
        print "take", str(iterTime), "seconds"
        PFID = dataPath + 'P%s.txt' %i    
        PWriter = open(PFID,'w+')
        PWriter.writelines(P)
        PWriter.close()
        UFID = dataPath + 'U%s.txt' %i    
        UWriter = open(UFID,'w+')
        UWriter.writelines(U)
        UWriter.close()
    # at least it worked
    # the probl
                    
    
                    
                
    
    
    # testing user ID reading 
    # testing ID reading
    # testing rating data reading
    # testing rating data construction
    
    # error calculation
    # save result 
 

