# -*- coding: utf-8 -*-
"""
Nov. 28 Basic PMF Helper Files

This script contains the helper functions used for basic PMF 
Created on Tue Nov 29 14:24:4 2016

@author: Scarecrow
"""
# this function updates the uth row of user feature matrix U, for the vectorized gradient update
import numpy as np
def userGradientUpdate(RU, P, Uu, lambda1):

    RU_hat = np.dot(Uu,P)*np.sign(RU)
    grad = np.dot((RU_hat - RU),P) + lambda1*Uu
    return grad
        
# this function updates the ith row of item feature matrix P, for the vectorized gradient update
def itemGradientUpdate(RI, U, Pi, lambda2):
    
    RI_hat = np.dot(Pi,U)*np.sign(RI)
    grad = np.dot((RI_hat - RI),U) + lambda2*Pi
    return grad

# this function is the basic element-wise updating function
def upUpdate(Uu,Pi,Rui,lambda1,lambda2):
    Rui_hat = np.dot(Uu,Pi.T)
    if abs(Rui_hat)>= 40:
        print "overshoot! There must be something wrong with the gradient updating!"
    diff = Rui_hat - Rui
    Uu = diff*Pi + lambda1*Uu
    Pi = diff*Uu + lambda2*Pi
    res = diff**2
    return Uu,Pi,res
    
def recordCSV(matrix, FID):
    try:
        with open(FID,'wb') as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
            f.close()
            print "written successful: FID ",FID
    except:
        print "cannot write file: FID ",FID

def parallelUpdate(RUI,U,P,j,k,lambda1,lambda2,step,error,numRating,lock):
     if RUI[j][k] != 0:
         Rui_hat = np.dot(U[j,np.newaxis],P[k,np.newaxis].T)
         res = Rui_hat - RUI[j][k]
         #lock.acquire()
         if abs(Rui_hat)>= 40:
             print "overshoot! There must be something wrong with the gradient updating!"
         Uu = res*P[k] + lambda1*U[j]
         Pi = res*U[j] + lambda2*P[k]
         U[j] = U[j] - step*Uu
         P[k] = P[k] - step*Pi
         #lock.release()
         
         




