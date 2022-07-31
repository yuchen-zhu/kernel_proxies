#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Sep 29 15:48:32 2020


@author: afsaneh

"""

import os,sys
import time
import numpy as np 
import pandas as pd
import functools
from typing import Callable
import jax.scipy.linalg as jsla
import jax.numpy.linalg as jnla
import operator
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Any, Iterator, Tuple
from functools import partial


import random
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
from numpy.linalg import matrix_rank



import statistics 
import itertools as it
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise 

from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.preprocessing import StandardScaler

import numba
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap
from jax import random




    
@jax.jit
def modist(v):   
    return jnp.median(v)

@jax.jit
def sum_jit(A,B):
    return jnp.sum(A,B)

@jax.jit
def linear_kern(x, y):
    return jnp.sum(x * y)

@jax.jit
def l2_dist(x,y):
    return jnp.array((x - y)**2)

#@functools.partial(jax.jit, static_argnums=(0,1))
def identifier(x,y):
    if (x!=y): 
        b=0
    else: 
        b=1
    return b


@functools.partial(jax.jit, static_argnums=(0))
def dist_func(func1: Callable, x,y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func1( x1, y1))(y))(x)


@jax.jit 
def rbf_ker(x,y,scale=1):
    dist_mat=dist_func(l2_dist,x,y)
    gamma=modist(jnp.sqrt(dist_mat))
    #gamma=1
    #coef=1/(2*gamma**2)
    coef=1/(2*scale*(gamma**2))
    return jnp.exp(-coef*dist_mat)

@jax.jit 
def identifier_ker(x,y):
    return dist_func(identifier,x,y)

#% function h
#@jax.jit
def cal_h(params_h,a,w):
        h,s2_A,s1_W, m1,m2=params_h       
        k_AA_r =rbf_ker (s2_A, jnp.asarray(a).reshape(1,))
        k_WW_r =rbf_ker (s1_W, jnp.asarray(w).reshape(1,))       
        b=h.reshape(m1,m2)
        return jnp.dot(jnp.dot(mat_trans(k_WW_r),b), k_AA_r) [0][0]

def cal_h_vec(params_h,a,W):
        h,s2_A,s1_W, m1,m2=params_h       
        k_AA_r =rbf_ker (jnp.array(s2_A), jnp.asarray(a).reshape(1,)).squeeze()
        k_WW_r =rbf_ker (jnp.array(s1_W), jnp.array(W)).squeeze()   
        b=h.reshape(m1,m2)
        return jnp.dot(jnp.dot(mat_trans(k_WW_r),b), k_AA_r)  
    
def cal_h_vecl(params_h,a,W):
        h,s2_A,s1_W, m1,m2=params_h 
        m1_train=m1
        m2_train=m2
        lst_a_ker=[]
        
        for i in s2_A.columns: 
            if np.issubdtype(s2_A[i],np.integer):    

                kern_ma =identifier_k  (jnp.array(s2_A[i]) , jnp.asarray(a).reshape(1,)).reshape(m2_train,1)            
            else: 
                arr=jnp.array(s2_A[i])
                kern_ma =rbf_ker (jnp.array(s2_A[i]) , jnp.asarray(a).reshape(1,)).reshape(m2_train,1)
                
            lst_a_ker.append(kern_ma)
            
        lst_w_ker=[]
        for i in s1_W.columns:           
            if np.issubdtype(s1_W[i],np.integer):
                kern_m = identifier_k  (jnp.array(s1_W[i]) , jnp.array( W[i]))
            else: 
                kern_m =rbf_ker (jnp.array(s1_W[i]) , jnp.array( W[i]))
            lst_w_ker.append(kern_m)

        
        def had_ker(lst_k):
            hk=jnp.ones(lst_k[0].shape)
            for i in range(len(lst_k)):
                hk=Hadamard_prod(hk,lst_k[i])
            return hk
        
        k_AA_r=had_ker(lst_a_ker)
        k_WW_r=had_ker(lst_w_ker)
        b=h.reshape(m1,m2)
        
        return mat_mul(mat_mul(mat_trans(k_WW_r),b), k_AA_r)   
    
    
    
    
def cal_h_veclx(params_h,do_A,sampl_w,lst_a,sampl_x, int_lst=[]):
        
        h,s2_AX,s1_W, m1,m2,k_ww_1=params_h 
        
        m1_train=m1
        m2_train=m2
        lst_a_ker=[]
        lst_x_ker=[]
        lst_w_ker=[]
        
        for i in s2_AX.columns: 
            
            if i in lst_a:
                
                if np.issubdtype(s2_AX[i],np.integer):     
                    arr=jnp.asarray(s2_AX[i]) 
                    kern_ma =identifier_k  (arr, do_A)#.reshape(m2_train,1)            
                
                elif i in int_lst: 
                    arr=jnp.asarray(s2_AX[i]) 
                    kern_ma =identifier_k  (arr, do_A)#.reshape(m2_train,1)            

                else: 
                    arr=jnp.asarray(s2_AX[i]) 
                    kern_ma =rbf_ker  (arr, do_A)#.reshape(m2_train,1)
               
                lst_a_ker.append(kern_ma)
            
            else:
                if np.issubdtype(s2_AX[i],np.integer):  
                    #arr=jnp.asarray(s2_AX[i])         
                    kern_mx =identifier_k (jnp.asarray(s2_AX[i]) , jnp.asarray(sampl_x[i]))          
                
                elif i in int_lst: 
                    kern_mx =identifier_k (jnp.asarray(s2_AX[i]) , jnp.asarray(sampl_x[i]))          

                else: 
                    #arr=jnp.asarray(s2_AX[i]) 
                    kern_mx =rbf_ker (jnp.asarray(s2_AX[i]) , jnp.asarray(sampl_x[i])) 
               
                lst_x_ker.append(kern_mx)

        
        
        for i in s1_W.columns:   
            
            if np.issubdtype(s1_W[i],np.integer):
                    #arr1=jnp.asarray(s1_W[i])                    
                    kern_mw = identifier_k  (jnp.array(s1_W[i]), jnp.array(sampl_w[i]))
            
            elif  i in int_lst:
                    #arr1=jnp.asarray(s1_W[i])                    
                    kern_mw = identifier_k  (jnp.array(s1_W[i]), jnp.array(sampl_w[i]))
            
            else: 
                    #arr1=jnp.asarray(s1_W[i])
                    kern_mw = rbf_ker (jnp.array(s1_W[i]), jnp.array(sampl_w[i]))
            
            lst_w_ker.append(kern_mw)

        
        def had_ker(lst_k):           
            if lst_k==[]: 
                hk=jnp.ones(m2).reshape(m2,1)
                
            else:
                hk=jnp.ones(lst_k[0].shape)
                for i in range(len(lst_k)):
                    hk=Hadamard_prod(hk,lst_k[i])
            return hk
        
        k_XX_r=had_ker(lst_x_ker).mean(axis=1)
        k_AA_r=had_ker(lst_a_ker).squeeze()
        k_AX_r=jnp.array([k_AA_r[:,i]* k_XX_r for i in range(k_AA_r.shape[1])])
        k_WW_r=had_ker(lst_w_ker).mean(axis=1)
        b=h.reshape(m1,m2)
        
        return mat_mul(k_AX_r,mat_mul(b,k_WW_r))
                      



@jax.jit 
def Hadamard_prod(A,B):
    return A*B 


@jax.jit 
def jsla_inv(A):
    return jsla.inv(A)

@jax.jit
def jnla_norm(A):
  return jnla.norm(A)

@jax.jit
def kron_prod(a,b):
    return jnp.kron(a,b)


@jax.jit
def modif_kron(x,y):
    if (y.shape[1]!=x.shape[1]): 
        print("Column_number error")
    else:
        return jnp.array(list(jnp.kron(x[:,i], y[:,i]).T for i in list(range(y.shape[1]))))

   
@jax.jit
def mat_trans(A):
    return jnp.transpose(A)
    

def regularisation_term(A,B,reg_coef,m2,m1,lamd=0.000001): 
    
    term1=reg_coef * m2 * jnp.kron(A,B)
    #dim=m1*m2
    #I=jnp.identity(dim)
    #term2=(jnp.array(lamd).reshape(1,) * jnp.identity(dim))
    return  term1


@jax.jit
def cal_loocv(K, reg, y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    H = I - K.dot(jsla.inv(K + lam * nD * reg))
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.linalg.norm(tildeH_inv.dot(H.dot(y)))


def cal_l_y (K, reg, y, low=0.00001, high=10, n=500):
    lam_values = np.linspace(low, high, num=n)
    grid_search={}
    for lam in lam_values:
        grid_search[lam]=cal_loocv(K, y, lam)
    return min(grid_search.items(), key=operator.itemgetter(1))[0]


@jax.jit
def cal_loocv_emb(K, kernel_y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    Q = jsla.inv(K + lam * nD * I)
    H = I - K.dot(Q)
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.trace(tildeH_inv @ H @ kernel_y @ H @ tildeH_inv)


def cal_l_w (K, kernel_y, low=0.0001, high=1, n=10, abs_low=.001):  
    git=1e-05
    lam_values = np.logspace(np.log10(low), np.log10(high), n)
    tolerance=lam_values [1]-lam_values [0]
    grid_search={}
    for lam in lam_values:
        grid_search[lam]=cal_loocv_emb(K, kernel_y, lam)    
    l,loo=min(grid_search.items(), key=operator.itemgetter(1))
    
    '''while (abs(l-low)<tolerance and low> abs_low) :
            low=low *.1
            high=high *.1 + git
            lam_values = np.linspace(low, high, n)
            tolerance=lam_values [1]-lam_values [0]
            grid_search={}
            for lam in lam_values:
                grid_search[lam]=cal_loocv_emb(K, kernel_y, lam)    
            l,loo=min(grid_search.items(), key=operator.itemgetter(1))
            
    while abs(l-high)<tolerance:
            low= low *10
            high=high *10  +git   
            lam_values = jnp.linspace(low, high, n)
            tolerance=lam_values [1]-lam_values [0]
            grid_search={}
            for lam in lam_values:
                grid_search[lam]=cal_loocv_emb(K, kernel_y, lam)    
            l,loo=min(grid_search.items(), key=operator.itemgetter(1))'''

    return l,loo   
    
    


@jax.jit
def cal_loocv_alpha(K, sigma, gamma, y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    H = I - mat_mul(mat_mul(K,gamma), (jsla.inv(sigma + lam * nD* I)))
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.linalg.norm(tildeH_inv.dot(H.dot(y)))

def cal_l_yw (K, sigma, gamma, y, low=0.01, high=1, n=10, abs_low=.001):
    git=1e-05
    lam_values = np.logspace(np.log10(low), np.log10(high), num=n)
    tolerance=lam_values [1]-lam_values [0]
    grid_search={}
    for lam in lam_values:
        grid_search[lam]=cal_loocv_alpha(K,sigma, gamma, y, lam)
    l,loo=min(grid_search.items(), key=operator.itemgetter(1))
    '''
    while (abs(l-low)<tolerance and low> abs_low):
            low=low *.1
            high=high *.1+git
            lam_values = np.linspace(low, high, num=n)
            tolerance=lam_values [1]-lam_values [0]
            grid_search={}
            for lam in lam_values:
                grid_search[lam]=cal_loocv_alpha(K,sigma, gamma, y, lam)
            l,loo=min(grid_search.items(), key=operator.itemgetter(1))
            
    while abs(l-high)<tolerance:
            low= low *10
            high=high *10 +git
            lam_values = np.linspace(low, high, num=n)
            tolerance=lam_values [1]-lam_values [0]
            grid_search={}
            for lam in lam_values:
                grid_search[lam]=cal_loocv_alpha(K,sigma, gamma, y, lam)
            l,loo=min(grid_search.items(), key=operator.itemgetter(1))
            '''
    return l,loo 

#test=pd.DataFrame(grid_search.items())
#plt.scatter(test[0],test[1])

    

#%Data to store for multiple uses to avoid repeating calculation 
        #k_ZZ_2_act 
        #k_WW_2_act
        #k_W1W2_act
        
def Kernels(samp1,samp2): 
    k_AA_1 =rbf_ker (samp1[:,0], samp1[:,0])
    k_AA_2 =rbf_ker (samp2[:,0], samp2[:,0])
    k_A1A2 =rbf_ker (samp1[:,0], samp2[:,0])
    k_ZZ_1 =rbf_ker (samp1[:,2], samp1[:,2])
    #k_ZZ_2 =rbf_ker (samp2[:,2], samp2[:,2])
    k_Z1Z2 =rbf_ker (samp1[:,2], samp2[:,2])
    k_WW_1 =rbf_ker (samp1[:,3], samp1[:,3])
    #k_WW_2 =rbf_ker (samp2[:,3], samp2[:,3])
    #k_W1W2 =rbf_ker (samp1[:,3], samp2[:,3])
    return k_AA_1, k_AA_2 , k_A1A2,k_ZZ_1 , k_Z1Z2, k_WW_1
        
def Kernels_n(samp1,samp2): 
    k_AA_1 =rbf_ker (samp1[:,0], samp1[:,0])
    k_AA_2 =rbf_ker (samp2[:,0], samp2[:,0])
    k_A1A2 =rbf_ker (samp1[:,0], samp2[:,0])
    k_ZZ_1 =rbf_ker (samp1[:,2], samp1[:,2])
    k_ZZ_2 =rbf_ker (samp2[:,2], samp2[:,2])
    k_Z1Z2 =rbf_ker (samp1[:,2], samp2[:,2])
    k_WW_1 =rbf_ker (samp1[:,3], samp1[:,3])
    k_WW_2 =rbf_ker (samp2[:,3], samp2[:,3])
    k_W1W2 =rbf_ker (samp1[:,3], samp2[:,3])
    return k_AA_1, k_AA_2 , k_A1A2,k_ZZ_1 ,k_ZZ_2 , k_Z1Z2, k_WW_1, k_WW_2, k_W1W2

def is_pos_def(x):
    return (np.linalg.eigvals(x), np.all(np.linalg.eigvals(x) > 0))

def cal_mse(y,ey,n):
    return 1/n*np.square(y-ey)



def sample_split(key, data,n_val,n_trn, n_total):
    val=jnp.split(random.permutation(key,data),
                    (n_val,n_val+n_trn,n_total),axis=0)[0]
    train=jnp.split(random.permutation(key,data),
                    (n_val,n_val+n_trn,n_total),axis=0)[1]
    test=jnp.split(random.permutation(key,data),
                    (n_val,n_val+n_trn,n_total),axis=0)[2]
    
    return val,train,test


def sampling(A, key, n):
    return jnp.split(random.permutation(key,A),(n,A.shape[0]),axis=0)[0]

@jax.jit
def mat_mul(A,B):
    return jnp.matmul(A,B)

@jax.jit
def jsla_solve(A,B):
    return jax.sp.linalg.solve(A, B, assume_a = 'pos')



def ace_point(key,A2,n, mu,vu, 
              mw, vw, my, vy, params_h,a_AY,a_WY): 
    
    causal_effect=pd.DataFrame()
    A_cause=np.arange(A2.min(),A2.max(),(A2.max()-A2.min())/20)
    
    
    for i in range(len(A_cause)):
        counter=i*3
        A=jnp.repeat(A_cause[i],n)
        U=gen_U(mu,n,key[counter],vu)
        W=gen_W( U , mw , n, vw, key[counter+1])
        H=cal_h_vec(params_h,A_cause[i],W).reshape(n,)
        Y=gen_Y(causal_Y,A_cause[i], a_AY, U, W, a_WY,my, n,vy ,key[counter+2])
        y_ind_a=causal_Y(A_cause[i], a_AY)
        Y_c_A=jnp.repeat(y_ind_a,n)
        causal_effect=causal_effect.append(pd.DataFrame([A,W,H,Y,Y_c_A]).T, ignore_index=True)
    
    causal_effect.columns=['A','W','H','Y','Y_c_A']

    return causal_effect


def h_surf(params_h, A2, W2):
    list_aw=list(it.product(jnp.arange(A2.min(),A2.max(),1),
                            jnp.arange(W2.min(),W2.max(),1)))
    #list_aw_df=pd.DataFrame(list(list_aw))
    h_surface=list((i[0],i[1],cal_h(params_h,i[0],i[1])) for i in list_aw)
    #h_surface=pd.concat([pd.DataFrame(list_aw),h_O_aw],axis=1)
    #h_surface.columns=['A','W','H']
    return h_surface

             
def normaliser (K):
    return (K-K.mean())/(jnp.sqrt(K.var()))



def ichol(K, err = 1):
    n = K.shape[0]
    d = np.array(np.diag(K))
    R = np.zeros((n,n))
    I = -1 * np.ones(n, dtype=int)
    a = np.max(d)
    j = 0
    I[j] = np.argmax(d)
    nu = []
    while(a > err and j < n):
        a = np.max(d)
        I[j] = np.argmax(d)
        nu.append(np.sqrt(a))
        for i in range(n):
            R[j,i] = (K[I[j], i] - R[:,i].dot(R[:, I[j]]))/np.sqrt(a)
            d[i] -=  R[j,i]*R[j,i]
        j = j+1
    R = R[:j,:]
    return R,I


@jax.jit
def jsla_inv_svd(A): 
    ur, s, vh =jsla.svd(A, full_matrices=False, overwrite_a=True)
    return jnp.dot(vh.transpose(),jnp.dot(jnp.diag(s**-1),ur.transpose()))


def h_surf_data(params_h, A2,W2):
    list_aw=list(it.product(A2,W2))
    #list_aw_df=pd.DataFrame(list(list_aw))
    h_surface=list((i[0],i[1],cal_h(params_h,i[0],i[1])) for i in list_aw)
    #h_surface=pd.concat([pd.DataFrame(list_aw),h_O_aw],axis=1)
    #h_surface.columns=['A','W','H']
    return h_surface


def ace_point_data(key,A2,n, mu,vu, 
              mw, vw, my, vy, params_h,a_AY,a_WY): 
 
    causal_effect=pd.DataFrame()
    A_cause=np.arange(A2.min(),A2.max(),(A2.max()-A2.min())/20)
      
    for i in range(len(A_cause)):
        counter=i*3
        A=jnp.repeat(A_cause[i],n)
        U=gen_U(mu,n,key[counter],vu)
        W=gen_W( U , mw , n, vw, key[counter+1])
        H=cal_h_vec(params_h,A_cause[i],W).reshape(n,)
        Y=gen_Y(A_cause[i], a_AY, U, W, a_WY,my, n,vy ,key[counter+2])
        #y_ind_a=causal_Y(A_cause[i], a_AY)
        #Y_c_A=jnp.repeat(y_ind_a,n)
        causal_effect=causal_effect.append(pd.DataFrame([A,W,H,Y]).T, ignore_index=True)
    
    causal_effect.columns=['A','W','H','Y']

    return causal_effect


def cal_mse(cal_h_vecAW: callable,params_h,A2, W2, ):
    estimated_h=cal_h_vecAW(params_h,A2, W2)
    estimated_ha=jnp.average(estimated_h, axis=0)
    
    
def identifier(x,y):
    if (x!=y): 
        b=0
    else: 
        b=1
    return b


def identifier_k(A,B):
    l=list(it.product(A,B))
    a=[]
    for i in l:
        a.append(identifier(i[0],i[1]))
    return np.array(a).reshape(A.shape[0],B.shape[0])



def standardise(X):
    scaler = StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler
    
    
    
    
def stage2_weights(Gamma_w, Sigma_inv):
            n_row = Gamma_w.shape[0]
            arr = [mat_mul(jnp.diag(Gamma_w[i, :]), Sigma_inv) for i in range(n_row)]
            return jnp.concatenate(arr, axis=0)    
    



def standardise(X):
    scaler = StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler



def standardise_arr (arr=A):
    return (arr-arr.mean(axis=0))/arr.std(axis=0)
