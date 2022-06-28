
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:38:21 2021

@author: afsaneh
"""
import os,sys,datetime
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
import numpy as np 
import pandas as pd
import operator
import itertools as it

from functools import partial
import matplotlib as mpl
import statistics 
import itertools as it
import pickle

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.scipy.linalg as jsla
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import inspect
import glob
import time


path='/Users/afsaneh/Desktop/Final/for_Github'

#current_path=os.path.abspath(os.path.dirname(path))
#sys.path.append(os.path.join(current_path,'..','utils'))

from utils import *
from cal_alpha import cal_alpha_opt_post

#% generate random state
#extra 2keys for choosing train_test data subsets
seed1=547610
seed2=240650
seed3=35400
seed4=1872304
num_var=6
sd_lst= [5949, 7422, 4388, 2807, 5654, 5518, 1816, 1102, 9886, 1656, 4379,
       2029, 8455, 4987, 4259, 2533, 9783, 7987, 1009, 2297] #np.random.choice(10000,20)

'''
key1 = random.PRNGKey(seed1)
key1, *subkeys1 = random.split(key1, num_var+1)

# extra terms for ACE
key2 = random.PRNGKey(seed2)
key2, *subkeys2 = random.split(key2, num_var+1+2*5)

'''
# verbos 200
key3 = random.PRNGKey(seed3)
key3, *subkeys3 = random.split(key3, 2)


# for ACE
key4 = random.PRNGKey(seed4)
key4, *subkeys4 = random.split(key4, 1000*num_var)


ewh_dict={}

#mix_w=[0.8, 0.2]

'''
 
for sd in sd_lst:#[:1]:
        #print(sd)
        #data = np.load('main_seed%s_std.npz' %sd)
        data=np.load('/Users/afsaneh/Documents/UCL/Thesis/KPV/AG/Final/Final/main_seed%s_std.npz' %sd)
        #do_cal = np.load('do_A_seed%s_std.npz' %sd)
        do_cal=np.load('/Users/afsaneh/Documents/UCL/Thesis/KPV/AG/Final/Final/do_A_seed%s_std.npz' %sd)
        keep_x=False
        
        #samp_size=[500,1000,1500, 2000]
        #samp_size=[100,200,300, 500]#,800,1000,1500,2000]
            
        '''
do_cal=np.load('/Users/afsaneh/Documents/UCL/TKPVF/edu/IM/do_A_edu_IM_80_seed100.npz')
keep_x=False
for f in glob.glob('/Users/afsaneh/Documents/UCL/TKPVF/edu/IM/main_edu_IM*.npz'):
        
        
        
        
        #print(f)
        data = np.load(f)    
        
        
        data_dict={}
        for i in data.files:
            data_dict[i]= data[i]
        
        lst_train= ['train_y', 'train_a',  'train_z', 'train_w', 'train_u']
        lst_test= ['test_y', 'test_a',  'test_z', 'test_w', 'test_u']
        
        train =[]
        test= []
        
        for i in  data_dict.keys():
            if i in lst_train:             
                    for k in range(data_dict[i].shape[1]):
                        train.append(data_dict[i][:,k])
            if i in lst_test:             
                    for k in range(data_dict[i].shape[1]):
                        test.append(data_dict[i][:,k])
     
        
        
        
        
        
        keep_x=False
        preprocess_var=True
        
        
        
        start_time = time.time()
        
        def range_var(arr):
               try:
                   ran_var=arr.shape[1]
               except IndexError:
                   ran_var=1          
               return ran_var
              
        def create_lst(str_v,n_str=1):
            lst_arr=[]   
            if n_str==1: 
                lst_arr.append(str_v)
            else: 
                for count in range(1, n_str+1): 
                    lst_arr.append(str_v+str(count))
           
            return lst_arr
        
        
        def x_inclusion(data, keep_x=True):   ## Is X included in the model ?
            if keep_x:
                try: 
                    X = data['train_x']#[:n] #ndarray
                    X2= data['test_x']
                    lst_x =  create_lst('X', range_var(X))
                except KeyError:
                    X=[]
                    X2=[]
                    lst_x=[]
            else: 
                X=[]
                X2=[]
                lst_x=[]        
            return X,X2, lst_x
        
           
        A = data['train_a']#[:n])
        Y = data['train_y']#[:n]
        U = data['train_u']#[:n] #ndarray
        Z = data['train_z']#[:n]
        W = data['train_w']#[:n]
        X =  x_inclusion(data, keep_x)[0]
        A2= data['test_a']
        Y2= data['test_y']
        U2= data['test_u']
        Z2= data['test_z']
        W2= data['test_w']
        X2= x_inclusion(data, keep_x)[1]    
        
        lst_a =  create_lst('A', range_var(A))
        lst_y =  create_lst('Y', range_var(Y))
        lst_w =  create_lst('W', range_var(W))
        lst_z =  create_lst('Z', range_var(Z))
        lst_u =  create_lst('U', range_var(U))
        lst_x =  x_inclusion(data, keep_x)[2] 
        
        int_lst=[]
        
        lst_O=lst_a+lst_y+lst_z+lst_w+lst_x
        lst_O_exp=lst_a+lst_z+lst_w+lst_x
        lst_all=lst_u+lst_a+lst_y+lst_z+lst_w+lst_x
        
        lst_O_w=lst_a+lst_y+lst_z+lst_w+lst_x
        lst_O_exp_w=lst_a+lst_z+lst_w+lst_x
        
        
        do_A=do_cal['do_A']
        gt_EY_do_A=do_cal['gt_EY_do_A']
        #plt.scatter(do_A,gt_EY_do_A),plt.scatter(A,Y)
        
        
                
        if keep_x: 
            O_train_val=np.append (np.append (np.append((jnp.array([A,Y]).T.squeeze()),Z, axis=1), W, axis=1), X, axis=1)
            O_test=np.append (np.append (np.append((jnp.array([A2,Y2]).T.squeeze()),Z2, axis=1), W2, axis=1), X2, axis=1)
            
        else:
            O_train_val=np.append (np.append((jnp.array([A,Y]).T.squeeze()),Z, axis=1), W, axis=1)
            O_test= np.append (np.append((jnp.array([A2,Y2]).T.squeeze()),Z2, axis=1), W2, axis=1)
        

        if preprocess_var:
            O_train_val, Std_Scale=standardise(O_train_val)
            O_test=standardise(O_test)[0]
            do_A_s = (do_A-A.mean())/A.std()#[0].inverse_transform(do_A)
            #gt_EY_do_A = (gt_EY_do_A-Y.mean())/Y.std()#standardise(Y)[1].inverse_transform(gt_EY_do_A)
            
         #plt.scatter((do_A-A.mean())/A.std(),gt_EY_do_A),plt.scatter(standardise(A)[0],standardise(Y)[0])
        n_test=O_test.shape[0]
        n_val=int(O_test.shape[0])

        #print("--- %s seconds ---" % (time.time() - start_time))   

        
        samp_size=[3000]#[500]#500,1000]#,2000]
        c=-1
        results={}
        scale_max=1.5
        scale_min=1.5
        
        l_yw_min=.01
        l_w_min=.001
        optimise_l_yw=False
        optimise_l_w=True
        
        
        for num in range(len(samp_size)):
            ewh_dict_df=pd.DataFrame()
            for key in subkeys3:
                #for scale in np.linspace(1,5,10):
                    
                    params_h=[]
                    c=c+1
                    n_train=samp_size[num] 

                    m1_train=n_train#int(n_train*0.4)## I have to change it back to 50%
                    m2_train=n_train#-m1_train
                    
                     
                    
                    O_train_val_df=pd.DataFrame(O_train_val).sample(n_train, random_state=key)
                    O_train_val_df.columns=lst_O
                    
                    O_test_df=pd.DataFrame(O_test).sample(n_val, random_state=key)
                    O_test_df.columns=lst_O
                    

                    
                    
                    '''
                    sns.pairplot(O_train_val_df)        
                    plt.savefig('Pairplot_uniform_normal_seed'+str(sd)+'_Size'+str(samp_size[num])+'_scale'+str(scale_max)+'.png')
                    plt.show()
                    plt.close()
            
                    '''
                
                    'val,samp1,samp2= sample_split(key, data=O_train_val, n_val=n_val,n_trn=m1_train, n_total=O_train_val.shape[0])'

                    train_sample, val_samp=O_train_val_df, O_test_df
                        
                        
                    samp1=samp2=train_sample
               
                      
                    params_h, lambda_dict =cal_alpha_opt_post(samp1,samp2, m1_train, 
                                                   m2_train,l_w_max=1,l_yw_max=1,lst_var=lst_O_exp,
                                                   lst_a=lst_a,lst_x=lst_x,
                                                   lst_z=lst_z,lst_w=lst_w, lst_y=lst_y, scale_mx=scale_max,
                                                   scale_mn=scale_min,int_lst=int_lst,optimise_l_yw=optimise_l_yw, optimise_l_w=optimise_l_w, l_yw_min=l_yw_min,l_w_min=l_w_min)
                 
                    #print("--- %s seconds ---" % (time.time() - start_time)) 
                    
                    Ew_h=[]
                    sampl_w=val_samp[lst_w]
                    sampl_x=val_samp[lst_x]
                    Ew_Haw_t= cal_h_veclx(params_h,do_A_s,sampl_w, lst_a, sampl_x, int_lst=int_lst) 
                    
                    if preprocess_var:
                        Ew_Haw=Ew_Haw_t* Y.std() + Y.mean()
                    else:
                        Ew_Haw=Ew_Haw_t
                        
                    
                    mse_c=((np.abs(Ew_Haw-gt_EY_do_A)).sum())/do_A.shape[0]
                    
                    
                    fig=plt.figure(2)
                    ax=fig.gca() 
                    plt.scatter(do_A,gt_EY_do_A, c='blue', marker="o", label='y(do(A)')
                    plt.scatter(do_A,Ew_Haw, c='red', marker="o", label='Ave_w(H)')
                    
                    ax.set_xlabel('A', fontweight ='bold') 
                    plt.title("l= {l}, MAE_c={b:.3f}".format(l=lambda_dict, b=mse_c))
                    plt.legend(loc='upper left');
                    
                    plt.savefig('IM_seed'+str(f[-7:])+'_Size'+str(samp_size[num])+'_scale'+str(scale_max)+'.png')#+str(c)+'.png')
                    plt.show()
                    plt.close()
                    
                    print (f[-7:-4],lambda_dict,mse_c)
                    
                    ewh_dict[f[-7:-4],samp_size[num]]=Ew_Haw
       
        
        
pickle.dump(ewh_dict,open('results_IM_seed{sb}_size{size}_{date}.p'.format(sb=f[-7:-4],  size=3000 , date=time.strftime("%Y%m%d_%H%M")),'wb'))    
