#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:41:04 2021

@author: afsaneh
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:23:26 2021

@author: afsaneh
"""
import os,sys,datetime
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
import numpy as np 
import pandas as pd
import operator

from functools import partial
import statistics 
import itertools as it
import pickle

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.scipy.linalg as jsla
import jax.numpy.linalg as jnla
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge


from utils import identifier_k,rbf_ker, Hadamard_prod ,cal_l_w,mat_mul,mat_trans, modif_kron

from utils import cal_l_yw ,jsla_inv, cal_h_vecl, cal_h_veclx,cal_loocv_emb,cal_loocv_alpha, stage2_weights



def cal_alpha_opt_post(samp1,samp2,m1_train,m2_train, l_w_max, l_yw_max,lst_var,lst_a,lst_x,lst_z,lst_w,lst_y, scale_mx=10,scale_mn=.5,int_lst=[], optimise_l_yw=False,optimise_l_w=False, l_yw_min=0.01,l_w_min=.01):     
        l_w=l_yw=0        
        
        l_w_max=l_w_max
        l_yw_max=l_yw_max
        
        scale_dict={}
        kernel_var={} 
        
        if optimise_l_w==True:
            
            
                
            for sc in np.logspace(np.log10(scale_mn),np.log10(scale_mx),1):                       
                for i in lst_var:   
                    if  np.issubdtype(samp1[i],np.integer): #'''samp1[i]'''
                            k_11 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp1[i]))
                            
                            if samp1.equals(samp2):
                                k_12 = k_11
                                k_22 = k_11
                                
                            else: 
                                k_12 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp2[i]))
                                k_22 = identifier_k  (jnp.array(samp2[i]), jnp.array(samp2[i]))
                            
                    elif i in int_lst:
                            k_11 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp1[i]))
                            
                            if samp1.equals(samp2):
                                k_12=k_11
                                k_22 = k_11
                            else: 
                                k_12 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp2[i]))
                                k_22 = identifier_k  (jnp.array(samp2[i]), jnp.array(samp2[i]))
                                              
                    
                    else: 
                            k_11 = rbf_ker (jnp.array(samp1[i]), jnp.array(samp1[i]), sc)
                            
                            if samp1.equals(samp2):
                                k_12=k_11
                                k_22 = k_11
                                
                            else: 
                                k_12 = rbf_ker (jnp.array(samp1[i]), jnp.array(samp2[i]), sc)
                                k_22 = rbf_ker (jnp.array(samp2[i]), jnp.array(samp2[i]), sc)
 
                    
                    kernel_var[i]=[k_11 ,k_12,k_22,samp1[i], samp2[i]]
                            
                k_ZZ_1=np.ones(shape=[m1_train, m1_train])
                k_Z1Z2=np.ones(shape=[m1_train, m2_train])
                k_ZZ_2=np.ones(shape=[m2_train, m2_train])
                k_WW_1=np.ones(shape=[m1_train, m1_train])
                k_W1W2=np.ones(shape=[m1_train, m2_train])
                k_WW_2=np.ones(shape=[m2_train, m2_train])   
                k_AA_1=np.ones(shape=[m1_train, m1_train])
                k_A1A2=np.ones(shape=[m1_train, m2_train])
                k_AA_2=np.ones(shape=[m2_train, m2_train])
                k_XX_1=np.ones(shape=[m1_train, m1_train])
                k_X1X2=np.ones(shape=[m1_train, m2_train])
                k_XX_2=np.ones(shape=[m2_train, m2_train])
            
              
                for i in lst_z:       
                        k_ZZ_1= Hadamard_prod(k_ZZ_1,kernel_var[i][0])        
                        k_Z1Z2= Hadamard_prod(k_Z1Z2,kernel_var[i][1])  
                        k_ZZ_2= Hadamard_prod(k_ZZ_2,kernel_var[i][2])  
                    
                for i in lst_w:       
                        k_WW_1= Hadamard_prod(k_WW_1,kernel_var[i][0])        
                        k_W1W2= Hadamard_prod(k_W1W2,kernel_var[i][1])  
                        k_WW_2= Hadamard_prod(k_WW_2,kernel_var[i][2])  
                    
                for i in lst_a:       
                        k_AA_1= Hadamard_prod(k_AA_1,kernel_var[i][0])           
                        k_A1A2= Hadamard_prod(k_A1A2,kernel_var[i][1])      
                        k_AA_2= Hadamard_prod(k_AA_2,kernel_var[i][2])    
                        
                for i in lst_x:       
                        k_XX_1= Hadamard_prod(k_XX_1,kernel_var[i][0])           
                        k_X1X2= Hadamard_prod(k_X1X2,kernel_var[i][1])      
                        k_XX_2= Hadamard_prod(k_XX_2,kernel_var[i][2])    
                        
                    
                    
                hp_K_AZX11 = Hadamard_prod(Hadamard_prod(k_AA_1,k_ZZ_1),k_XX_1)
                hp_K_AZX12 = Hadamard_prod(Hadamard_prod(k_A1A2,k_Z1Z2),k_X1X2)
            

                
                l_w,loo1= cal_l_w (K=hp_K_AZX11, kernel_y=k_WW_1, low=l_w_min ,high=.1, n=5) #low=max(0.000001,l_w_max-0.05)                      
                #print(sc,l_w,loo1/m1_train)
                scale_dict[sc]=[l_w,loo1,hp_K_AZX11,hp_K_AZX12,k_WW_1, k_AA_2,k_XX_2]
                    
                  
            sc_optimal=(sorted(scale_dict.items(), key=lambda e: e[1][1]))[0]
            sc_opt=sc_optimal[0]
            l_w,loo1=sc_optimal[1][0],sc_optimal[1][1]/m1_train
            hp_K_AZX11, hp_K_AZX12 =sc_optimal[1][2],sc_optimal[1][3]
            k_WW_1=sc_optimal[1][4]
            k_AA_2=sc_optimal[1][5]
            k_XX_2=sc_optimal[1][6]
            #print (round(float(sc_opt),2), l_w, loo1)
        
       
        else:
            
            sc=1
            for i in lst_var:   
                    if  np.issubdtype(samp1[i],np.integer): #'''samp1[i]'''
                            k_11 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp1[i]))
                            
                            if samp1.equals(samp2):
                                k_12=k_11
                                k_22 = k_11
                            else: 
                                k_12 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp2[i]))
                                k_22 = identifier_k  (jnp.array(samp2[i]), jnp.array(samp2[i]))
                            
                    elif i in int_lst:
                            k_11 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp1[i]))
                            
                            if samp1.equals(samp2):
                                k_12=k_11
                                k_22 = k_11
                            else: 
                                k_12 = identifier_k  (jnp.array(samp1[i]), jnp.array(samp2[i]))
                                k_22 = identifier_k  (jnp.array(samp2[i]), jnp.array(samp2[i]))
                                              
                    
                    else: 
                            k_11 = rbf_ker (jnp.array(samp1[i]), jnp.array(samp1[i]), sc)
                            
                            if samp1.equals(samp2):
                                k_12=k_11
                                k_22 = k_11
                            else: 
                                k_12 = rbf_ker (jnp.array(samp1[i]), jnp.array(samp2[i]), sc)
                                k_22 = rbf_ker (jnp.array(samp2[i]), jnp.array(samp2[i]), sc)
                    
                    kernel_var[i]=[k_11 ,k_12, k_22 ,samp1[i], samp2[i]]
                        
            k_ZZ_1=np.ones(shape=[m1_train, m1_train])
            k_Z1Z2=np.ones(shape=[m1_train, m2_train])
            k_ZZ_2=np.ones(shape=[m2_train, m2_train])
            k_WW_1=np.ones(shape=[m1_train, m1_train])
            k_W1W2=np.ones(shape=[m1_train, m2_train])
            k_WW_2=np.ones(shape=[m2_train, m2_train])
            k_AA_1=np.ones(shape=[m1_train, m1_train])
            k_A1A2=np.ones(shape=[m1_train, m2_train])
            k_AA_2=np.ones(shape=[m2_train, m2_train])
            k_XX_1=np.ones(shape=[m1_train, m1_train])
            k_X1X2=np.ones(shape=[m1_train, m2_train])
            k_XX_2=np.ones(shape=[m2_train, m2_train])
                
        
          
            for i in lst_z:       
                k_ZZ_1= Hadamard_prod(k_ZZ_1,kernel_var[i][0])        
                k_Z1Z2= Hadamard_prod(k_Z1Z2,kernel_var[i][1])  
                #k_ZZ_2= Hadamard_prod(k_ZZ_2,kernel_var[i][2])  
            
            for i in lst_w:       
                k_WW_1= Hadamard_prod(k_WW_1,kernel_var[i][0])        
                k_W1W2= Hadamard_prod(k_W1W2,kernel_var[i][1])  
                #k_WW_2= Hadamard_prod(k_WW_2,kernel_var[i][2])  
            
            for i in lst_a:       
                k_AA_1= Hadamard_prod(k_AA_1,kernel_var[i][0])           
                k_A1A2= Hadamard_prod(k_A1A2,kernel_var[i][1])      
                k_AA_2= Hadamard_prod(k_AA_2,kernel_var[i][2])    
                
            for i in lst_x:       
                k_XX_1= Hadamard_prod(k_XX_1,kernel_var[i][0])           
                k_X1X2= Hadamard_prod(k_X1X2,kernel_var[i][1])      
                k_XX_2= Hadamard_prod(k_XX_2,kernel_var[i][2])    
                
                
                
            hp_K_AZX11 = Hadamard_prod(Hadamard_prod(k_AA_1,k_ZZ_1),k_XX_1)
            hp_K_AZX12 = Hadamard_prod(Hadamard_prod(k_A1A2,k_Z1Z2),k_X1X2)   
            
            
            l_w=l_w_min
            loo1=cal_loocv_emb(K=hp_K_AZX11, kernel_y= k_WW_1, lam=l_w) /m1_train 
            sc_opt=sc
        
                            
        
        Core_w_az=hp_K_AZX11 + m1_train * l_w * jnp.eye(m1_train)
               
        
        ##### if you comment this in, you can see the graph of Loo1 and MSE of W estimation as a function of A,X,Z
        '''
        Fst_mse={}
        loo1={}
        
        Core_w_az_i   = jsla.solve(Core_w_az,jnp.eye(m1_train),sym_pos=True)
        fix_gamma=mat_mul(k_WW_1,Core_w_az_i)        
        mse1= (1/m1_train) * jnla.norm(k_WW_1 - mat_mul(fix_gamma, hp_K_AZX11)) 
        
        for l_w_t in np.logspace(np.log10(0.0001), np.log10(.1), 20): 
                    Core_w_az_i_t=hp_K_AZX11 + m1_train * l_w_t * jnp.eye(m1_train)
                    Core_w_az_t   = jsla.inv(Core_w_az_i_t)#jsla.solve(Core_w_az_i_t,jnp.eye(m1_train),sym_pos=True)                        
                    fix_gamma_t=mat_mul(k_WW_1,Core_w_az_t)
        
                    Fst_mse[l_w_t]=  jnla.norm(k_WW_1 - mat_mul(fix_gamma_t, 
                                                                   hp_K_AZX11))
                    loo1[l_w_t ]= cal_loocv_emb(K=hp_K_AZX11, kernel_y=k_WW_1, lam=l_w_t )                       

        test1=pd.DataFrame(Fst_mse.items())
        test2=pd.DataFrame(loo1.items())
        
        fig, ax_left = plt.subplots()
        ax_right = ax_left.twinx()        
        ax_left.scatter(test1[0],test1[1], color='black')
        ax_left.set_ylabel('sme',c='black') 
        ax_right.scatter(test2[0],test2[1], color='red')
        ax_right.set_ylabel('loo',c='red') 
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()    
        '''
        #####
        
        
              
        Gamma_w= jsla.solve(Core_w_az,hp_K_AZX12,sym_pos=True)
        
        kw1_gamma=mat_mul(k_WW_1,Gamma_w)
                        
        g_kw1_g=mat_mul(mat_trans(Gamma_w),kw1_gamma)
                
        Sigma=Hadamard_prod(g_kw1_g, Hadamard_prod(k_AA_2, k_XX_2))
        
        
        ## This to improve the efficiency and speed of the solution. One can estimate causal effect for smaller sample size 
        ##n1=n2=500 and use the regularisation penalty of the second stage, as the regularisation coef in larger samples. 
        ## we learned this trick from Liyuan Xu, L. Xu, H. Kanagawa, and A. Gretton. Deep proxy causal learning and its application to confounded bandit policy evaluation. arXiv preprint arXiv:2106.03907, 2021
        ##https://proceedings.neurips.cc/paper/2021/hash/dcf3219715a7c9cd9286f19db46f2384-Abstract.html
        
        
        if optimise_l_yw==True:
            
            D_t= modif_kron(kw1_gamma,Hadamard_prod(k_AA_2, k_XX_2))                   
            mk_gamma_I=mat_trans(modif_kron(Gamma_w,jnp.eye(m2_train)))                       
            l_yw,loo2= cal_l_yw (K=D_t, sigma=Sigma, gamma=mk_gamma_I, y=jnp.array(samp2[lst_y]),low=l_yw_min,high=l_yw_max, n=5) #low=max(0.0005,l_yw_max-0.1), high=l_yw_max, n=10)
            #print(l_yw,loo2/m2_train)
        
        else: 
            l_yw=l_yw_min
       
        '''
        This shows that in larger sample the above modification, which we learnt from Liyuan Xu, improves the speed and memory efficiney.
        start_time = time.time()
        core_v=jsla_inv(Sigma + m2_train*l_yw*jnp.eye(m2_train))     
        print("--- %s seconds ---" % (time.time() - start_time))   
        #coef_v=mat_mul(mk_gamma_I,core_v )
        coef_v=stage2_weights(Gamma_w, core_v)        
        print("--- %s seconds ---" % (time.time() - start_time))   
        alpha1=mat_mul( coef_v, jnp.array(samp2[lst_y]))  
        print("--- %s seconds ---" % (time.time() - start_time))   
        #mse=(1/m2_train)*(sum(jnp.absolute(jnp.array(samp2[lst_y])-mat_mul(D_t, alpha))))
        '''


        alpha= stage2_weights(Gamma_w, jsla.solve(Sigma + m2_train*l_yw*jnp.eye(m2_train),jnp.array(samp2[lst_y])))
  
        
        '''
        snd_mse={}
        loo2={}
        
        for l_yw_t in np.linspace(0.00001, 0.1, 20): 
                    core_v_t=jsla_inv(Sigma + m2_train*l_yw_t*jnp.eye(m2_train))                            
                    coef_v_t=mat_mul(mk_gamma_I,core_v_t )                    
                    alpha_t=mat_mul( coef_v_t, jnp.array(samp2[lst_y]))        

                    snd_mse[l_yw_t]=(sum(jnp.absolute(jnp.array(samp2[lst_y])-mat_mul(D_t, alpha_t))))
        
                    loo2[l_yw_t]= cal_loocv_alpha(K=D_t, sigma=Sigma, gamma=mk_gamma_I, 
                                                  y=jnp.array(samp2[lst_y]), lam=l_yw_t)                                          

        test1=pd.DataFrame(snd_mse.items())
        test2=pd.DataFrame(loo2.items())
        
        fig, ax_left = plt.subplots()
        ax_right = ax_left.twinx()        
        ax_left.scatter(test1[0],test1[1], color='black')
        ax_left.set_ylabel('sme',c='black') 
        ax_right.scatter(test2[0],test2[1], color='red')
        ax_right.set_ylabel('loo',c='red') 
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()    
        '''
        #####

        
        lambda_dict={"scale":round(float(sc_opt),2), "l_w":round(float(l_w),8), "l_yw":round(float(l_yw),8)}
        
        
        #print('lamda_w:{:.6f}, loo1:{:.4f}, lamda_y:{:.6f},loo2:{:.4f}'.format(l_w,loo1,l_yw,loo2))
        params_h=[alpha, samp2[lst_a+lst_x],samp1[lst_w], m1_train, m2_train, k_WW_1]  
        return params_h, lambda_dict




