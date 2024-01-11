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




''' By 尚宸
path = r"./Results_cross"
dirs = os.listdir( path )
print(dirs)
count = 0
aaa = []
for i in range (len(dirs)):
    #print(path+'/'+dirs[i])
    dirs_E = os.listdir( path + '/' + dirs[i] )
    for j in range (len(dirs_E)):
        #print(path + '/' + dirs[i] + '/' + dirs_E[j])
        dirs_OC = os.listdir( path + '/' + dirs[i] + '/' +  dirs_E[j])
        for k in range (len(dirs_OC)):
            print(path + '/' + dirs[i] + '/' + dirs_E[j] + '/'+ dirs_OC[k] + '/Results.mat' )
            x = path + '/' + dirs[i] + '/' + dirs_E[j] + '/'+ dirs_OC[k] + '/Results.mat' 
            #print(len(x))
            count += 1
            
            mat = loadmat(x) 
            a.append(mat)
print(count)
'''
#data = loadmat('./Results_cross/E01/OC10/Trial40/Results.mat')

Results_cross = []
Results_walking = []

def findAllFile_cross(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_cross.append(mat)
                yield fullname

def findAllFile_walking(base1):
    for root, ds, fs in os.walk(base1):
        for f in fs:
            if f.endswith('.mat'):
                fullname = os.path.join(root, f)
                mat = loadmat(fullname) 
                Results_walking.append(mat)
                yield fullname
#build temp list
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

#Feature = np.empty((45,23,101))
#FAFA = np.empty((18,101))
Feature = np.empty((23,101))
def getfeature(value):
    
    
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
    '''
    k = 0
    for l in range(IK_toay.shape[0]):  #45人
        for i in range(IK_toay.shape[1]): #6個特徵
            for j in range(IK_toay.shape[2]):  #xyz(3)
                FAFA[3*i+j,:] = IK_toay[l][i,j,:] #k = 3*i+j
                Feature[l,:,:] = FAFA[3*i+j,:] #k = 3*i+j
                k += 1
    '''

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
    
    
    return Feature
    
    
    
    
    
    
    
    
    
    


''' 
c = []
d = []
e = []
IK_array = np.empty((6,101,3))
IK_tay = np.empty((3,101))
IK_toay = np.empty((6,3,101))
IK_tot = np.empty((18,101))
F1 = np.empty((23,101))

IA_array = np.empty((4,101,1))
IA_tay = np.empty((1,101))
IA_toay = np.empty((4,1,101))

GS_array = np.empty((1,101,1))
GS_tay = np.empty((1,101))
GS_toay = np.empty((1,1,101))
#IK_all = 
'''
Fwwwww = np.empty((45,23,101))
FEATURESSS = np.empty((23,101))


def main():
    base = r"./Results_cross/"
    base1 = r"./Results_walking/"
    #Ftot = np.empty((3,23,101))
    #Ftot = np.empty((45,23,101))
    #Ftot = np.empty((len(Results_cross),23,101))
    #Ftot = np.empty((45,23,101))
    for i in findAllFile_cross(base):
        print(i)
    for j in findAllFile_walking(base1):
        print(j)
    
    #Ftot[0,:,:]= getfeature(0)
    #Ftot= getfeature(1)
    #print(Ftot.shape)
    #Ftot[2,:,:]= getfeature(2)
    #print(Ftot)
    #Ftot = np.empty((len(Results_cross),23,101))
    #print(Ftot.shape)
    
    #k = 0
    #print(len(Results_cross))
    
    
    #Ftot= getfeature(1)
    #print(Ftot)
    
    
    
    for i in range(len(Results_cross)):
        FEATURESSS= getfeature(i)
        Fwwwww[i,:,:] = FEATURESSS
        
        print(i,FEATURESSS)
        print('')
        #k+=1
        #print(FEA0)
    #print(Ftot.shape[0])
    
    
    '''
    df = pd.DataFrame(Ftot[0])
    df.to_csv('AAA.csv')     
    df = pd.DataFrame(Ftot[1])
    df.to_csv('BBB.csv')
    df = pd.DataFrame(Ftot[2])
    df.to_csv('CCC.csv')
    
    #ex1_answer = {'Ftot':Ftot}
    #np.save('ex1_answer.npy', ex1_answer)
    
    
    df = pd.DataFrame(Ftot)
    df.to_csv('BBB.csv')     
    '''
    
    
    
    
    '''    
    x = Results_cross[0]['Results']['IK']
    c.append(x)
    #print(c)
    #for i in range(len(c[0])):
    IK_array[0,:,:] = np.array(c[0]['LeadingAnkle'])
    IK_array[1,:,:] = np.array(c[0]['LeadingHip'])
    IK_array[2,:,:] = np.array(c[0]['LeadingKnee'])
    IK_array[3,:,:] = np.array(c[0]['TrailingAnkle'])
    IK_array[4,:,:] = np.array(c[0]['TrailingHip'])
    IK_array[5,:,:] = np.array(c[0]['TrailingKnee'])
    #IK_array = np.array(c[0][i])
    #print(IK_array)
    #print(IK_array.shape)
    for i in range(IK_toay.shape[0]):
        IK_tay = np.transpose(IK_array[i])
        IK_toay[i,:,:] = IK_tay

    #print(IK_toay.shape[0])
    IK_tot = np.empty((18,101))
    #IK_tot[0:3,:] = IK_toay[0]
    k = 0
    for i in range(IK_toay.shape[0]):
        for j in range(IK_toay.shape[1]):
            F1[k,:] = IK_toay[i,j,:] #k = 3*i+j
            k += 1
            
 
    #IK_tot=np.reshape(IK_toay,(18,101))
    #IK_tot = IK_toay.reshape((18, 101))
    y = Results_cross[0]['Results']['IA']
    d.append(y)
    print(IA_array.shape)
    IA_array[0,:,:] = np.array(d[0]['FrontalIA'])
    IA_array[1,:,:] = np.array(d[0]['FrontalRCIA'])
    IA_array[2,:,:] = np.array(d[0]['SagittalIA'])
    IA_array[3,:,:] = np.array(d[0]['SagittalRCIA'])
    
    for i in range(IA_array.shape[0]):
      IA_tay = np.transpose(IA_array[i])
      IA_toay[i,:,:] = IA_tay
    
    for i in range(IA_toay.shape[0]):
        for j in range(IA_toay.shape[1]):
            F1[k,:] = IA_toay[i,j,:] #k = 3*i+j
            k += 1
    print(F1)
    #F1_TR1[0:4,:] = IA_array
    z = Results_cross[0]['Results']['SelfDefinedVariables']
    e.append(z)
    GS_array[0,:,:] = np.array(e[0]['GaitVelocity'])
    GS_tay = np.transpose(GS_array[0])
    GS_toay[0,:,:] = GS_tay
    F1[22,:] = GS_toay[0,0,:]
    print(F1) 
    '''
    
    '''
    for t in range(len(Results_cross)):
        x = Results_cross[t]['Results']['IK']['LeadingAnkle']
        print(x)
        c.append(x)
    
    my_array = np.array(c)
    print(my_array.shape)
    '''
        
    '''    
    for t in range(len(a)):
        x = a[t]['Results']['STP']
        print(x)
        c.append(x)
    print(c)
    df = pd.DataFrame(c)
    df.to_csv('myfile.csv')      
    '''    
  
if __name__ == '__main__':
    main()




        