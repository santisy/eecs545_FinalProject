# -*- coding: utf-8 -*-
"""
Social constrained version of PMF in Yelp dataset
至少试一下，或者用constrained MF
Modified on Mon Nov 28
Modified on Sat Nov. 30
Demenstrating that PMF can easily be extended with prior knowledges and constrains or bias
Modified on Wed Nov 30, optimized version, with rating bias
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
    lambdaU = 1 # lambda for user matrix regularization 0.6 0.8, feature dim 10?
    lambdaP = 1 # lambda for item matrix regularization
    lambdaBu = 10 # lambda for user bias
    lambdaBi = 10 #lambda for item bias
    beta = 0.5 # influence of friends on users
    step = 0.0004 # iteration step 
    # select training data and testing data
    numFolds = 5 # number of folds of data
    testIdx = 1 #index of testing data
    # iterNum = 250 # number of iterations
    log = [] # iteration log
    maxIterNum = 300 # conrolling number of iterations
        
    # data file path, and read
    mode = 1 # 1 for debugging, SUV and RUI are directly read from saved file 
    categoryName = 'shopping'
    method = "PMF Socially Constrained"
    dataset = "Yelp-shopping"
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
    
    friendCircleFID = dataPath + 'friendCircle10.csv'
    friendCircle = pd.read_csv(friendCircleFID)
    friendCircle.columns = ['idx','mainUser','friend']
    numRatingList = trainContent.groupby('userID').count().itemID
    numRatingList.columns = ['userID','numRating']
        
    # training user ID List generation 
    userList = pd.unique(trainContent['userID']).tolist()
    userNum = len(userList)
    # training item ID generation
    itemList = pd.unique(trainContent['itemID']).tolist()
    itemNum = len(itemList)
    ratingBias = trainContent['rating'].mean()# calculate rating bias
        
    # training rating and friend matrix construction
    gc.enable()
    gc.collect()
    if mode != 1: # oridinary mode
        RUI = np.zeros([len(userList),len(itemList)], dtype = np.float16)
        SUV = np.zeros([len(userList),len(itemList)], dtype = np.float16)
        for (idx,row) in trainContent.iterrows():
            RUI[userList.index(row['userID']),itemList.index(row['itemID'])] = np.float16(row['rating'])
        print "RUI has been constructed!"
        np.save(dataPath +"RUI",RUI)
        # now construct the SUV matrix, row is main user    
        for (idx,row) in friendCircle.iterrows():
            try:
                SUV[userList.index(row.mainUser.split('\r')[0]),userList.index(row.friend.split('\r')[0])] = np.float16(numRatingList.loc[numRatingList.index == row.mainUser.split('\r')[0]].values[0])
            except:
                pass
        print "SUV has been constructed!"        
        # calculate social circle & influence matrix
        for i in range(0,userNum):
            currFriend = np.nonzero(SUV[i]) # friends of each user
            sumExperience = np.sum(SUV[i][currFriend])
            SUV[i][currFriend] = SUV[i][currFriend]/sumExperience # normalized experience of each user
        print "social circle experience calculated!"
        np.save(dataPath+"experienced SUV",SUV)
    else: #quick mode where SUV is given
        SUV = np.load(dataPath+"experienced SUV.npy")
        RUI = np.load(dataPath+"RUI.npy")
        
    uLst = RUI.nonzero()[0].tolist()
    iLst = RUI.nonzero()[1].tolist()
    numRating = len(uLst)
    ratingList = zip(uLst,iLst)
    ratingListLen = len(ratingList)
    # U,P,BU,BI initialization
    U = np.random.randn(len(userList),featureDim)
    P = np.random.randn(len(itemList),featureDim)
    # this remains to be discussed.
    BU = np.random.randn(userNum)
    BI = np.random.randn(itemNum)
    #BU = np.zeros([userNum,1])
    #BI = np.zeros([itemNum,1])
    # iteration
    gc.collect()
    iterNum = 0
    errorOld = 100
    errorNew = 90
    
    while(errorOld - errorNew >= step):
        iterNum += 1
                
    # took 23 seconds for each iteration
        errorOld = errorNew
        errorNew = 0
        count = 0 # for debug
        startTime = time.time()
        
        for idx,(j,k) in enumerate(ratingList):
            try:
                RUI[j][k] != 0
            except:
                print"RUI == 0, there must be something wrong"
            # calculate gradient
            (bu,bi,Uu,Pi,res) = biasedUpdate(BU[j],BI[k],ratingBias,U[j,np.newaxis],P[k,np.newaxis],RUI[j][k],lambdaU,lambdaP,lambdaBu,lambdaBi)
            # updating            
            U[j] = U[j] - step*Uu
            P[k] = P[k] - step*Pi
            BU[j] = BU[j] - step*bu
            BI[k] = BI[k] - step*bi
            # updating 
            errorNew += res/numRating
            # i don't think that will work
            if (idx < ratingListLen-1) and (ratingList[idx + 1][0] != j): # it's time to update Uu according to social circle
                friends = np.nonzero(SUV[j])[0] # find the friends
                term1 = beta*(U[j] - np.dot(SUV[j][friends],U[friends]))
                friendsSubLst = list(friends)
                term2 = np.zeros([1,featureDim])
                for friend in friendsSubLst:
                    friendfriends = np.nonzero(SUV[friend])[0] # feature for friend's friend
                    term2 += SUV[friend][j] * (U[friend] - np.dot(SUV[friend][friendfriends],U[friendfriends])) 
                Uu = beta*(term1 - term2)
                U[j] = U[j] - step*Uu
                
        iterTime = time.time() - startTime
        line1 =  str(iterNum)+ " iterations have finished!"
        line2 =  "the training error (RMSE) is " + str(float(np.sqrt(errorNew)))
        line3 =  "took " + str(iterTime) + "seconds"
        print line1,line2,line3
        log.append(line1)
        log.append(line2)
        log.append(line3)
        if iterNum>=maxIterNum:
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
            bu_hat = BU[userList.index(row['userID'])]
            bi_hat = BI[itemList.index(row['itemID'])]
            RUI_hat = float(np.dot(U_hat,P_hat.T)) + ratingBias + bu_hat + bi_hat
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
    recordCSV(BU,resultPath + 'BU%s.csv'%iterNum)
    recordCSV(BI,resultPath + 'BI%s.csv'%iterNum)
        
        
                
            
    
    
        
                    
    
                    
                


