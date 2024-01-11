import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
import random
import openpyxl
import pandas as pd
from sympy.plotting import plot3d
from IPython.display import display
from sympy import Symbol
import scipy.io
from mat4py import loadmat
import keras
from keras.callbacks import History,EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Reshape
import time
from keras.models import load_model


 
start = time.time()
Results_cross = []

def findAllFile_cross(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_cross.append(mat)
                yield fullname

#build temp list for crossing
IKlist = []
IAlist = []
GVlist = []
EPlist = []


IK_array = np.empty((6,101,3))
IK_tay = np.empty((3,101))
IK_toay = np.empty((6,3,101))


IA_array = np.empty((4,101,1))
IA_tay = np.empty((1,101))
IA_toay = np.empty((4,1,101))

GV_array = np.empty((1,101,1))
GV_tay = np.empty((1,101))
GV_toay = np.empty((1,1,101))

#取出跨越障礙高度共有leading、trailing的toe跟heel共四個值
#EP_array = np.empty((1,1))
#EP_toay = np.empty((1,4))


Feature = np.empty((23,101))
#Feature_obstacles = np.empty((23,101))


def getfeature(value):
    
    #EP_array = np.empty((1))
    o = Results_cross[value]['Results']['EP']
    EPlist.append(o)
    
    EP_array = np.array(EPlist[value]['LeadingToeClearance'])
    '''
    EP_array[0,:] = np.array(EPlist[value]['LeadingHeelClearance'])
    EP_array[1,:] = np.array(EPlist[value]['LeadingToeClearance'])
    EP_array[2,:] = np.array(EPlist[value]['TrailingHeelClearance'])
    EP_array[3,:] = np.array(EPlist[value]['TrailingToeClearance'])
    EP_toay = np.transpose(EP_array)
    '''
    #print(EP_array)
    #Feature[22,:] = GV_toay[0,0,:] 
    Feature_obstacles1 = EP_array

    
    x = Results_cross[value]['Results']['IK']
    IKlist.append(x)
    
    IK_array[0,:,:] = np.array(IKlist[value]['LeadingAnkle'])
    IK_array[1,:,:] = np.array(IKlist[value]['LeadingHip'])
    IK_array[2,:,:] = np.array(IKlist[value]['LeadingKnee'])
    IK_array[3,:,:] = np.array(IKlist[value]['TrailingAnkle'])
    IK_array[4,:,:] = np.array(IKlist[value]['TrailingHip'])
    IK_array[5,:,:] = np.array(IKlist[value]['TrailingKnee'])
    
    
   
    for i in range(IK_toay.shape[0]):  
        IK_tay = np.transpose(IK_array[i])
        IK_toay[i,:,:] = IK_tay
   

    k = 0
    for i in range(IK_toay.shape[0]):
        for j in range(IK_toay.shape[1]):
            Feature[k,:] = IK_toay[i,j,:] #k = 3*i+j
            k += 1
    
    y = Results_cross[value]['Results']['IA']
    IAlist.append(y)
    IA_array[0,:,:] = np.array(IAlist[value]['FrontalIA'])
    IA_array[1,:,:] = np.array(IAlist[value]['FrontalRCIA'])
    IA_array[2,:,:] = np.array(IAlist[value]['SagittalIA'])
    IA_array[3,:,:] = np.array(IAlist[value]['SagittalRCIA'])
    
    for i in range(IA_array.shape[0]):
      IA_tay = np.transpose(IA_array[i])
      IA_toay[i,:,:] = IA_tay
      

    for i in range(IA_toay.shape[0]):
        for j in range(IA_toay.shape[1]):
            Feature[k,:] = IA_toay[i,j,:] #k = 3*i+j
            k += 1
    
    z = Results_cross[value]['Results']['SelfDefinedVariables']
    GVlist.append(z)
    GV_array[0,:,:] = np.array(GVlist[value]['GaitVelocity'])
    GV_tay = np.transpose(GV_array[0])
    GV_toay[0,:,:] = GV_tay
    Feature[22,:] = GV_toay[0,0,:]
    
    
   
    
    
    
        
    return Feature,Feature_obstacles1
    




    
Feature_cross = np.empty((216,23,101))
#Feature_walking = np.empty((162,23,101))
Feature_obstacles = np.empty((216))
FEATUREccc = np.empty((23,101))
#FEATUREwww = np.empty((23,101))
#Feature_obs = np.empty((1))
#YY_cross = []
#YY_walking = []


def main():
    base = r"./Results_cross/"
    #base1 = r"./Results_walking/"
   
    for i in findAllFile_cross(base):
        print(i)
    '''    
    for j in findAllFile_walking(base1):
        print(j)
    '''
    
    
    for i in range(len(Results_cross)):
        FEATUREccc,Feature_obs= getfeature(i)
        Feature_cross[i,:,:] = FEATUREccc
        Feature_obstacles[i] = Feature_obs
        

    
    X_normalized = (Feature_cross - Feature_cross.mean()) / Feature_cross.std()
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Feature_obstacles, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    #train=0.6，val=0.2，test=0.2
    
    
    
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[2] , X_train.shape[1])
    X_val =X_val.reshape(X_val.shape[0],X_val.shape[2] , X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[2] , X_test.shape[1])


    model = load_model('./LSTM model performance/Regression/Regression.h5')
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mean_squared_error', optimizer= opt)