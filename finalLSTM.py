import os
import numpy as np
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
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split

Results_cross = []
Results_walking = []

def findAllFile_cross(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_cross.append(mat)
                print(len( Results_cross))
                yield fullname

def findAllFile_walking(base1):
    for root, ds, fs in os.walk(base1):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_walking.append(mat)
                yield fullname
#build temp list for crossing
IKlist = []
IAlist = []
GVlist = []

IK_array = np.empty((6,101,3))
IK_tay = np.empty((3,101))
IK_toay = np.empty((6,3,101))


IA_array = np.empty((4,101,1))
IA_tay = np.empty((1,101))
IA_toay = np.empty((4,1,101))

GV_array = np.empty((1,101,1))
GV_tay = np.empty((1,101))
GV_toay = np.empty((1,1,101))
Feature = np.empty((23,101))
#-----------------------------------------------
#build temp list for walking
IKlist_w = []
IAlist_w = []
GVlist_w = []

IK_array_w = np.empty((6,101,3))
IK_tay_w = np.empty((3,101))
IK_toay_w = np.empty((6,3,101))


IA_array_w = np.empty((4,101,1))
IA_tay_w = np.empty((1,101))
IA_toay_w = np.empty((4,1,101))

GV_array_w = np.empty((1,101,1))
GV_tay_w = np.empty((1,101))
GV_toay_w = np.empty((1,1,101))

Feature = np.empty((23,101))
#Feature_w = np.empty((23,101))

def getfeature(value,posture):
    if posture =='cross':
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
    
        
    elif posture =='walk':
        x = Results_walking[value]['Results']['IK']
        IKlist_w.append(x)
        
        IK_array_w[0,:,:] = np.array(IKlist_w[value]['LeadingAnkle'])
        IK_array_w[1,:,:] = np.array(IKlist_w[value]['LeadingHip'])
        IK_array_w[2,:,:] = np.array(IKlist_w[value]['LeadingKnee'])
        IK_array_w[3,:,:] = np.array(IKlist_w[value]['TrailingAnkle'])
        IK_array_w[4,:,:] = np.array(IKlist_w[value]['TrailingHip'])
        IK_array_w[5,:,:] = np.array(IKlist_w[value]['TrailingKnee'])
        
        
       
        for i in range(IK_toay_w.shape[0]):  
            IK_tay_w = np.transpose(IK_array_w[i])
            IK_toay_w[i,:,:] = IK_tay_w
       
    
        k = 0
        for i in range(IK_toay_w.shape[0]):
            for j in range(IK_toay_w.shape[1]):
                Feature[k,:] = IK_toay_w[i,j,:] #k = 3*i+j
                k += 1
            
        y = Results_walking[value]['Results']['IA']
        IAlist_w.append(y)
        IA_array_w[0,:,:] = np.array(IAlist_w[value]['FrontalIA'])
        IA_array_w[1,:,:] = np.array(IAlist_w[value]['FrontalRCIA'])
        IA_array_w[2,:,:] = np.array(IAlist_w[value]['SagittalIA'])
        IA_array_w[3,:,:] = np.array(IAlist_w[value]['SagittalRCIA'])
        
        for i in range(IA_array_w.shape[0]):
          IA_tay_w = np.transpose(IA_array_w[i])
          IA_toay_w[i,:,:] = IA_tay_w
        
        for i in range(IA_toay_w.shape[0]):
            for j in range(IA_toay_w.shape[1]):
                Feature[k,:] = IA_toay_w[i,j,:] #k = 3*i+j
                k += 1
        
        z = Results_walking[value]['Results']['SelfDefinedVariables']
        GVlist_w.append(z)
        GV_array_w[0,:,:] = np.array(GVlist_w[value]['GaitVelocity'])
        GV_tay_w = np.transpose(GV_array_w[0])
        GV_toay_w[0,:,:] = GV_tay_w
        Feature[22,:] = GV_toay_w[0,0,:]

    
    
    return Feature
    




    
Feature_cross = np.empty((162,23,101))
Feature_walking = np.empty((162,23,101))
FEATUREccc = np.empty((23,101))
FEATUREwww = np.empty((23,101))
YY_cross = []
YY_walking = []
#Y_cross = np.empty((45,1,1))
#Y_walking = np.empty((45,1,1))
#Feature_totally = np.empty((90,23,101))

def main():
    base = r"./Results_cross/"
    base1 = r"./Results_walking/"
   
    for i in findAllFile_cross(base):
        print(i)
    for j in findAllFile_walking(base1):
        print(j)
    
    
    
    for i in range(len(Results_cross)):
        FEATUREccc= getfeature(i,'cross')
        Feature_cross[i,:,:] = FEATUREccc
        
        #print(i,FEATUREccc)
        #print('')
    
    for i in range(len(Results_walking)):
        FEATUREwww= getfeature(i,'walk')
        Feature_walking[i,:,:] = FEATUREwww
        
        #print(i,FEATUREwww)
        #print('')
    Feature_totally = np.concatenate((Feature_cross, Feature_walking),axis = 0)
    
    for i in range(54):
        x = 1
        YY_cross.append(x)
    Y_cross = np.array(YY_cross)
    
    for i in range(54):
        x = 0
        YY_walking.append(x)
    Y_walking = np.array(YY_walking)
    
    Y_totally = np.concatenate((Y_cross, Y_walking),axis = 0)
    
    X_normalized = (Feature_totally - Feature_totally.mean()) / Feature_totally.std()
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_totally, test_size=0.2, random_state=42)
    
    
    
    
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[2] , X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[2] , X_test.shape[1])
    
    
    #%% self
    
    class SelfAttention(keras.layers.Layer):
        def __init__(self, attention_units):
            super(SelfAttention, self).__init__()
            self.attention_units = attention_units

        def build(self, input_shape):
            self.W = self.add_weight(
                shape=(input_shape[-1], self.attention_units),
                initializer="glorot_uniform",
                trainable=True
            )
            self.b = self.add_weight(
                shape=(self.attention_units,),
                initializer="zeros",
                trainable=True
            )
            self.V = self.add_weight(
                shape=(self.attention_units, 1),
                initializer="glorot_uniform",
                trainable=True
            )
    
        def call(self, inputs):
            score = keras.activations.tanh(keras.backend.dot(inputs, self.W) + self.b)
            attention_weights = keras.activations.softmax(keras.backend.dot(score, self.V), axis=1)
            attended_input = inputs * attention_weights
            return attended_input
    #%% end
    
    '''
    #建立LSTM模型 训练
    d = 0.01
    model = Sequential()
    model.add(Bidirectional(GRU(64, input_shape=(101,23))))
    model.add(Dropout(d))
    model.add(SelfAttention(64))
    #model.add(Dense(32,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    #earlystopping
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=2, mode='auto', restore_best_weights=True)
    model.fit(X_train, y_train, nb_epoch = 1000, batch_size = 32,validation_data=(X_test, y_test),callbacks=[monitor]) #训练模型1000次
    
    
    #劃出迭代曲線
    pd.DataFrame(model.history.history).plot()
    #在训练集上的拟合结果
    y_train_predict=model.predict(X_train)
    y_train_predict=y_train_predict[:,0]
    y_train_predict>0.5
    y_train_predict=[int(i) for i in y_train_predict>0.5]
    y_train_predict=np.array(y_train_predict)
    from sklearn import metrics
    print("精確度等指標：")
    print(metrics.classification_report(y_train,y_train_predict))
    print("混淆矩陣：") 
    print(metrics.confusion_matrix(y_train,y_train_predict))
    
    
    #在测试集上的拟合结果
    y_test_predict=model.predict(X_test)
    y_test_predict=y_test_predict[:,0]
    y_test_predict>0.5
    y_test_predict=[int(i) for i in y_test_predict>0.5]
    y_test_predict=np.array(y_test_predict)
    from sklearn import metrics
    print("精確度等指標：")
    print(metrics.classification_report(y_test,y_test_predict))
    print("混淆矩陣：")
    print(metrics.confusion_matrix(y_test,y_test_predict))
    '''
    '''
    model = Sequential()
    model.add(LSTM(64, input_shape=(101,23)))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 64
    epochs = 10
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    loss, accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)

    
    ### Create the Stacked LSTM model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(23,101)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()
    '''
    '''
    input_length = len(X_train[0])
    model = Sequential()
    model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print ('Compiling...')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    '''              
    #print ('Fitting model...')
    #hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_split = 0.1, verbose = 1)
    #model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)
    
    '''
    X_train = np.ramdom.shuffle(Feature_totally)
    Y_train =  Y_totally
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[2] , X_train.shape[1])
    Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1] , 1)
    '''
    
    return X_train, X_test, y_train, y_test,Feature_totally
      
if __name__ == '__main__':
    #Feature_totally = main()
    X_train, X_test, y_train, y_test ,Feature_totally= main()



        