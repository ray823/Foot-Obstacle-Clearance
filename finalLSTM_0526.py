'''
use the regression to predict the obstacle height

'''
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

 
start = time.time()
Results_cross = []
#Results_walking = []

def findAllFile_cross(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_cross.append(mat)
                yield fullname
'''
def findAllFile_walking(base1):
    for root, ds, fs in os.walk(base1):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_walking.append(mat)
                yield fullname
'''
#build temp list for crossing
IKlist = []
IAlist = []
GVlist = []
EPlist = []
EP_OBSlist = []
AP_01list = []
AP_02list = []
AP_03list = []
AP_04list = []
AP_05list = []
AP_06list = []


AP_01_array = np.empty((1,101))
AP_02_array = np.empty((1,101))
AP_03_array = np.empty((1,101))
AP_04_array = np.empty((1,101))
AP_05_array = np.empty((1,101))
AP_06_array = np.empty((1,101))

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


Feature = np.empty((29,101))
#Feature_obstacles = np.empty((23,101))


def getfeature(value):
    '''
    #EP_array = np.empty((1))
    o = Results_cross[value]['Results']['EP']
    EPlist.append(o)
    
    EP_array = np.array(EPlist[value]['LeadingToeClearance'])
    
    EP_array[0,:] = np.array(EPlist[value]['LeadingHeelClearance'])
    EP_array[1,:] = np.array(EPlist[value]['LeadingToeClearance'])
    EP_array[2,:] = np.array(EPlist[value]['TrailingHeelClearance'])
    EP_array[3,:] = np.array(EPlist[value]['TrailingToeClearance'])
    EP_toay = np.transpose(EP_array)
    
    #print(EP_array)
    #Feature[22,:] = GV_toay[0,0,:] 
    Feature_obstacles1 = EP_array
    '''
    
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
    
    #從EP取出障礙物高度並與Clearance相加取得toe離地高度
    A = Results_cross[value]['Results']['EP']
    EP_OBSlist.append(A)
    
    EP_OBS_array = np.array(EP_OBSlist[value]['OBS']+EP_OBSlist[value]['LeadingToeClearance'])
    Feature_obstacles1 = EP_OBS_array
    
    
    #取出AP單一值資料並變成序列資料
    B = Results_cross[value]['Results']['AP']['BodyHeight']
    AP_01list.append(B)
    for i in range(101):
        AP_01_array[:,i] = np.array(AP_01list[value])
        Feature[23,:] = AP_01_array[:,i]
    
    C = Results_cross[value]['Results']['AP']['BW']
    AP_02list.append(C)
    for i in range(101):
        AP_02_array[:,i] = np.array(AP_02list[value])
        Feature[24,:] = AP_02_array[:,i]
    
    D = Results_cross[value]['Results']['AP']['FootLength']
    AP_03list.append(D)
    for i in range(101):
        AP_03_array[:,i] = np.array(AP_03list[value])
        Feature[25,:] = AP_03_array[:,i]
        
    E = Results_cross[value]['Results']['AP']['HipWidth']
    AP_04list.append(E)
    for i in range(101):
        AP_04_array[:,i] = np.array(AP_04list[value])
        Feature[26,:] = AP_04_array[:,i]
    
    F = Results_cross[value]['Results']['AP']['ShankLength']
    AP_05list.append(F)
    for i in range(101):
        AP_05_array[:,i] = np.array(AP_05list[value])
        Feature[27,:] = AP_05_array[:,i]
    
    G = Results_cross[value]['Results']['AP']['ThighLength']
    AP_06list.append(G)
    for i in range(101):
        AP_06_array[:,i] = np.array(AP_06list[value])
        Feature[28,:] = AP_06_array[:,i]
   
    
    
    
        
    return Feature,Feature_obstacles1




    
Feature_cross = np.empty((204,29,101))
#Feature_walking = np.empty((162,23,101))
Feature_obstacles = np.empty((204))
FEATUREccc = np.empty((29,101))

#FEATUREap = np.empty((1))
#Feature_ap = np.empty((204,1))
#FEATUREwww = np.empty((23,101))
#Feature_obs = np.empty((1))
#YY_cross = []
#YY_walking = []


def main():
    base = r"./Results_cross/"
    #base1 = r"./Results_walking/"
   
    for i in findAllFile_cross(base):
        print(i)
   
    
    
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
    
    #%% self attention
    
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
    # returns a compiled model
    # identical to the previous one
    #model = load_model('my_model.h5')
    #建立LSTM模型 训练
    model = Sequential()
    input_shape=(101,23)
    model.add(Bidirectional(GRU(64, input_shape=(101,23), return_sequences=False)))
    model.add(SelfAttention(64))
    model.add(Dense(32, activation='relu'))
    model.add(SelfAttention(32))
    model.add(Dense(32, activation='relu'))
    model.add(SelfAttention(32))
    model.add(Dense(16, activation='relu'))
    model.add(SelfAttention(16))
    model.add(Dense(16, activation='relu'))
    model.add(SelfAttention(16))
    model.add(Dense(8, activation='relu'))
    model.add(SelfAttention(8))
    model.add(Dense(8, activation='relu'))
    model.add(SelfAttention(8))
    model.add(Dense(4, activation='relu'))
    model.add(SelfAttention(4))
    model.add(Dense(4, activation='relu'))
    model.add(SelfAttention(4))
    model.add(Dense(2, activation='relu'))
    model.add(SelfAttention(2))
    model.add(Dense(2, activation='relu'))
    model.add(SelfAttention(2))
    model.add(Dense(1, activation='linear'))
    
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mean_squared_error', optimizer= opt)
    
    
    model.fit(X_train, y_train, epochs=10, batch_size=32,validation_data=(X_val, y_val))


    
    #d = 0.01
    #model = Sequential()
    #input_shape=X_train.shape
    #model.add(LSTM(64, input_shape=(101,23) ,return_sequences = True))
    #model.add(Dropout(d))
    #model.add(SelfAttention(64))
    #model.add(LSTM(32,return_sequences=True))
    #model.add(Dense(32,init='uniform',activation='relu'))        
    #model.add(Dense(1,init='uniform',activation='linear'))
    
    #model.add(Dense(1))
    #舊的model.compile
    #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'])
    #opt = keras.optimizers.Adam(learning_rate=0.0001)
    #model.compile(loss='mse', optimizer= opt,metrics=['accuracy'])
    # Give a summary
    #model.summary()
    
    model.build(input_shape)
    print(model.summary())
    with open('./LSTM model performance/Regression_heel_test/modelsummary.txt', 'w') as f:

        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    #earlystopping
    start_fitting = time.time()
    #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=2, mode='auto', restore_best_weights=True)
    #model.fit(X_train, y_train, nb_epoch = 1000, batch_size = 32,validation_data=(X_val, y_val),callbacks=[monitor]) #训练模型1000次
    #epochxxx =  monitor.stopped_epoch+1   #len(model.history.history['loss'])
    #print(epochxxx)    
    fitting_model_time = time.time()
    print('Training Model took: ', fitting_model_time - start_fitting)

    #保存模型和權重
    saved = model.save('./LSTM model performance/Regression_heel_test/Regression.h5')    
    model.save_weights('./LSTM model performance/Regression_heel_test/Regression_weights.h5')  # to store
    '''
    #load模型和權重
    # returns a compiled model
    # identical to the previous one
    #model = load_model('my_model.h5')
    #model.load_weights('my_model_weights.h5')  # to load
    #%% plot loss and epoch relation
    '''
    loss_train = model.history.history['loss']
    loss_val = model.history.history['val_loss']
    epochs = range(epochxxx)
    fig = plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig('./LSTM model performance/Regression/train_Validation_loss.png')
    
    
    loss_train = model.history.history['accuracy']
    loss_val = model.history.history['val_accuracy']
    epochs = range(epochxxx)
    fig1 = plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig1.savefig('./LSTM model performance/Regression/train_Validation_accuracy.png')
    #%% end
    '''
    '''
    #劃出迭代曲線
    pd.DataFrame(model.history.history).plot()
    #在训练集上的拟合结果
    #plt.figure() 
    
    y_train_predict=model.predict(X_train)
    y_train_predict=y_train_predict[:,0]
    '''
    '''
    draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
    draw.iloc[100:150,0].plot(figsize=(12,6))
    draw.iloc[100:150,1].plot(figsize=(12,6))
    plt.legend(('real', 'predict'),fontsize='15')
    plt.title("Train Data",fontsize='30') #添加标题
    #plt.show()
    #fig.savefig('./LSTM model performance/Regression_heel_test/train_Validation_loss.png')
    #展示在训练集上的表现 
    '''
    '''
    epocha=range(len(y_train))
    epochb=range(len(y_train_predict))
    plt.plot(epocha, y_train, 'g', label='real')
    plt.plot(epochb, y_train_predict, 'b', label='predict')
    plt.legend(('real', 'predict'),fontsize='15')
    plt.title("Train Data",fontsize='30') #添加标题
    plt.show()
    
    #在測試集上的擬合结果
    #plt.figure() 
    y_test_predict=model.predict(X_test)
    y_test_predict=y_test_predict[:,0]
    
    draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
    draw.iloc[200:250,0].plot(figsize=(12,6))
    draw.iloc[200:250,1].plot(figsize=(12,6))
    plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
    plt.title("Test Data",fontsize='30') #添加标题
    '''
    '''
    epochc=range(len(y_test))
    epochd=range(len(y_test_predict))
    
    plt.plot(epochc, y_test, 'g', label='real')
    plt.plot(epochd, y_test_predict, 'b', label='predict')
    plt.title("Test Data",fontsize='30') #添加标题
    plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
    plt.show()
    '''
    
    
    #fig1.savefig('./LSTM model performance/Regression_heel_test/test_loss.png')
    #展示在测试集上的表现 
    '''
    #输出结果
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import math
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    print('训练集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_train_predict, y_train))
    print(mean_squared_error(y_train_predict, y_train) )
    print(mape(y_train_predict, y_train) )
    print('测试集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_test_predict, y_test))
    print(mean_squared_error(y_test_predict, y_test) )
    print(mape(y_test_predict,  y_test) )
    y_var_test=y_test[1:]-y_test[:len(y_test)-1]
    y_var_predict=y_test_predict[1:]-y_test_predict[:len(y_test_predict)-1]
    txt=np.zeros(len(y_var_test))
    for i in range(len(y_var_test-1)):
        txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
    result=sum(txt)/len(txt)
    print('预测涨跌正确:',result)
    print('训练时间（秒）:',54.56)
    
    
    end = time.time()
    print('TOTAL time spent', end-start)
    '''
    return Feature_cross,Feature_obstacles, X_train, y_train, X_val, y_val, X_test, y_test
    
      
if __name__ == '__main__':
    #Feature_totally = main()
    Feature_cross,Feature_obstacles, X_train, y_train, X_val, y_val, X_test, y_test= main()
    #Feature_totally,X_train, y_train, X_val, y_val, X_test, y_test,epochxxx= main()



        