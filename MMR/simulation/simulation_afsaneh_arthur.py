import os, sys, datetime
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
import numpy as np
import operator
from scipy.stats import norm
from functools import partial
import matplotlib as mpl
import statistics
import itertools as it
import pickle
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.scipy.linalg as jsla
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
# %matplotlib inline
plt.show()

from sklearn import preprocessing


# directory = "/Users/afsaneh/Documents/UCL/Thesis/KPV/"
# for item in os.listdir(directory):
#    if item.endswith(".p"):
#       os.remove(os.path.join(directory, item))

# from GenScm import *
# %Generators parameters

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# param
N = [5, 100, 500, 1000, 3000, 7000, 10000]
n = N[-1]
p = N[0]
train_sz = N[5]

do_A_min, do_A_max, n_A = -1.5, 1.5, 50


# %Generators parameters

# [u,a,y,z,w]
m_e = [4, 0, 0, 0, 0]
m_u = [0, 0, 0]
v_u = [1, 1, 1]
m_z = [0, 0]
v_z = [1, 1]
m_w = [0, 0]
v_w = [1, 1]
# cov_e=[[1],[,1],[,,1],[,,,1]]
C = [1, 2.4, 0.3, 3, 4.2]
seed = []
seed1 = 1850
seed2 = 3569
seed3 = 10
seed4 = 157257
seed5 = 42
seed6 = 368641
seed = [seed1, seed2, seed3, seed4]
num_var = 10


# % generate random state
# extra 2keys for choosing train_test data subsets

keyu = random.PRNGKey(seed1)
keyu, *subkeysu = random.split(keyu, 4)

keyz = random.PRNGKey(seed2)
keyz, *subkeysz = random.split(keyz, 4)


keyw = random.PRNGKey(seed3)
keyw, *subkeysw = random.split(keyw, 4)


keyx = random.PRNGKey(seed4)
keyx, *subkeysx = random.split(keyx, 100)

keya = random.PRNGKey(seed5)
keya, *subkeysa = random.split(keya, 100)


# %Generative models


def gen_sigma(p=10):
    # s=jnp.eye(p)
    s = []
    for i in range(p):
        for j in range(p):
            if j == i:
                s.append(1)
            elif abs(i - j) == 1:
                s.append(0.5)
            else:
                s.append(0)
    return jnp.array(s).reshape(p, p)


def gen_beta(p):
    return jnp.array([b ** (-2) for b in range(1, p + 1)])


def gen_U(n=100, key=subkeysu):
    e1 = random.uniform(key[0], (n,), minval=0, maxval=1)
    U2 = 3 * random.uniform(key[1], (n,), minval=0, maxval=1) - 1
    e3 = np.where((U2 > 1), 0, -1)
    e4 = np.where((U2 < 0), 0, -1)
    e5 = (e3 + e4)
    U1 = e1 + e5
    # U1=np.where((U2<0),0,0)
    return U1, U2


def gen_Z(U1, U2, m_z=m_z, v_z=v_z, n=100, key=subkeysz):
    ##Z1= U1*0.25+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    ##Z2= U2**2 + random.uniform(key[1],(n,),minval=0,maxval=1)
    ##Z1= U1+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    ##Z2= U2+ (random.normal(key[1],(n,))*v_z[1])+m_z[1]
    # Z1= U1+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    Z2 = U2 + random.uniform(key[1], (n,), minval=0, maxval=1)
    # return Z1,Z2
    return Z2


def gen_W(U1, U2, m_w=m_w, v_w=v_w, n=100, key=subkeysw):
    ##W1= U1+ random.normal(key[0],(n,))*v_w[0]+m_w[0]
    ##W2= U2+ random.normal(key[1],(n,))*v_w[1]+m_w[1]
    W1 = U1 + random.uniform(key[0], (n,), minval=0, maxval=1)
    # W2= U2+ random.normal(key[1],(n,))*v_w[1]+m_w[1]
    # return W1,W2
    return W1


def gen_X(p=20, n=100, key=keyx):
    if p != 0:
        m_x = np.zeros(p)
        sigma = gen_sigma(p)
        X = np.random.multivariate_normal(m_x, sigma, n)
    else:
        X = jnp.asarray([])
    return X.T


def gen_A(U1, U2, beta=0.05, n=100, key=keya):
    A = U2 + random.normal(key, (n,)) * beta
    return A


def gen_Y(A, U1, U2, n):
    y = U2 * (np.cos(2 * (A + .3 * U1 + .2)))
    return y


# % generative dist

sigma = gen_sigma(p)
beta = gen_beta(p)
'''
U = standardise((jnp.asarray(gen_U(m_u,n,subkeysu,v_u))).T) [0]                                  
Z = standardise(gen_Z(U[:,0], m_v,n, v_v, key=subkeysv[0])) [0] 
W = standardise(gen_W(U[:,1], m_v,n, key=subkeysv[1])) [0]
X=  standardise(gen_X(p,n, keyx).T) [0]
A = standardise(gen_A(U[:,1],Z,X,beta, n, key=keya)) [0]
Y = standardise(gen_Y(A, X, U[:,0], W, beta, n)) [0]
'''


def standardise(X):
    scaler = preprocessing.StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler


U = standardise((jnp.asarray(gen_U(n, key=subkeysu))).T)[0]
Z = standardise((jnp.asarray(gen_Z(U[:, 0], U[:, 1], m_z, v_z, n, key=subkeysz))).T)[0]
W = standardise((jnp.asarray(gen_W(U[:, 0], U[:, 1], m_w, v_w, n, key=subkeysw))).T)[0]
X = standardise(gen_X(p, n, keyx).T)[0]
A = standardise(gen_A(U[:, 0], U[:, 1], 0.05, n, key=keya))[0]
Y, Y_scaler = standardise(gen_Y(A, U[:, 0], U[:, 1], n))


'''non standar'
U = (jnp.asarray(gen_U(n,key=subkeysu))).T                              
#Z = standardise(gen_Z(U[:,0], m_v,n, v_v, key=subkeysv[0])) [0] 
#W = standardise(gen_W(U[:,1], m_v,n, key=subkeysv[1])) [0]
#X=  standardise(gen_X(p,n, keyx).T) [0]
A = (gen_A(U[:,0],U[:,1],0.05, n, key=keya)) 
Y = (gen_Y(A, U[:,0], U[:,1], n)) 
'''

train_u, test_u = U[:train_sz], U[train_sz:]
train_z, test_z = Z[:train_sz], Z[train_sz:]
train_w, test_w = W[:train_sz], W[train_sz:]
train_x, test_x = X[:train_sz], X[train_sz:]
train_a, test_a = A[:train_sz], A[train_sz:]
train_y, test_y = Y[:train_sz], Y[train_sz:]


n_eval = 1000
seed_eval = 49765
key_ev = random.PRNGKey(seed_eval)
key_ev, *subkeys_ev = random.split(key_ev, 4)


# causal ground truth standardised
do_A_scaled = jnp.linspace(do_A_min, do_A_max, n_A)
do_A = standardise(A)[1].inverse_transform(do_A_scaled)  # transform
print('do_A: ', do_A)
print('do_A_scaled: ', do_A_scaled)
Uns = (jnp.asarray(gen_U(n=n_eval, key=subkeys_ev))).T
Uss = standardise(Uns)[0]
EY_do_A = standardise(jnp.array([(gen_Y(A=a, U1=Uss[:, 0], U2=Uss[:, 1], n=n_eval)).mean() for a in do_A]))[0]  # should use Y scaler
EY_do_A = Y_scaler.transform(jnp.array([(gen_Y(A=a, U1=Uss[:, 0], U2=Uss[:, 1], n=n_eval)).mean() for a in do_A]).reshape(-1,1))  # should use Y scaler

plt.scatter(do_A, EY_do_A)
plt.title('GT standardised'), plt.show()


# causal ground truth UNstandardised
EY_do_A_ns = jnp.array([(gen_Y(A=a, U1=Uns[:, 0], U2=Uns[:, 1], n=n_eval)).mean() for a in do_A_scaled])
plt.scatter(do_A_scaled, EY_do_A_ns)
plt.title('GT UNstandardised'), plt.show()

np.savez(os.path.join(PATH, '../main_seed{}_single_dim_nonoise.npz'.format(seed)),
         splits=['train', 'test'],
         train_y=train_y,
         train_a=train_a,
         train_z=train_z,
         train_w=train_w,
         train_u=train_u,
         train_x=train_x,
         test_y=test_y,
         test_a=test_a,
         test_z=test_z,
         test_w=test_w,
         test_u=test_u,
         test_x=test_x)

np.savez(os.path.join(PATH, '../do_A_seed{}_single_dim_nonoise.npz'.format(seed)),
         do_A=do_A,
         gt_EY_do_A=EY_do_A)


############
# plotting #
############

D = pd.DataFrame([U[:500, 0], U[:500, 1], train_a[:500].flatten(), train_y[:500].flatten(), train_z[:500].flatten(),
                  train_w[:500].flatten()]).T
D.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']
D_doA = pd.DataFrame([do_A.flatten(), EY_do_A.flatten()]).T
D_doA.columns = ['A', 'EY_do_A']

ecorr_v = D.corr()
ecorr_v.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']

sem = 'arthur_afsaneh'
if not os.path.exists(PATH + sem + '_seed' + str(seed)):
    os.mkdir(PATH + sem + '_seed' + str(seed))
for v in ['U1', 'U2', 'A', 'Y', 'Z', 'W']:
    sns.displot(D, x=v, label=v, kde=True), plt.savefig(
        PATH + sem + '_seed' + str(seed) + '/' + v + '_dist.png'), plt.close()

sns.set_theme(font="tahoma", font_scale=1)
sns.pairplot(D), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'full_pairwise.png'), plt.close()
sns.pairplot(D_doA), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'ate_pairwise.png'), plt.close()

sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.savefig(
    PATH + sem + '_seed' + str(seed) + '/' + 'corr_all.png'), plt.close()
