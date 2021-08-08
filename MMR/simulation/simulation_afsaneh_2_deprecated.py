import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import random
import scipy.sparse as sp
import matplotlib as mpl
import statistics
# from keras.models import Sequential
# import tensorflow
from itertools import product
import itertools
import math
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import time
# import tensorflow as tf
import scipy.linalg as la
from decimal import *
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression as LR
# from sklearn.gaussian_process import kernels
# from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import sklearn.metrics.pairwise

##%matplotlib inline
plt.show()
sns.set_theme(font="tahoma", font_scale=0.6)


# %%Generators
def fun_y(a, b):
    Y = []
    for i in range(len(a)):
        if a[i] == 0:
            y = a_AY
        else:
            y = np.sin(a[i] * a_AY) / a[i]

        Y.append(y)
    return Y


def fun_w(a):
    W = []
    m_u = np.log2(min(U[U != 0]))

    for i in range(len(a)):
        if a[i] <= 0:
            w = m_u - 1
        else:
            w = np.log2(a[i])
        W.append(w)
    return W


# W=fun_w(a=U)
# plt.scatter(U,W)
# %%
# param
N = [5, 200, 10000, 100000]
# [a,y,z,w] heat as a catalyst z temprature w
a_AY = 2
a_AZ = 0.5

# [u,a,y,z,w]
m_e = [1, 1, 0, -1, 0]
# cov_e=[[1],[,1],[,,1],[,,,1]]
C = [1, 1, 1, 1, 4]

# U is a chi2 distribution
random.seed(100)
U = np.random.chisquare(m_e[0], N[1]).round(2)
U_inst = np.ones(N[1]).round(2)

# Z is noisy reading of U
random.seed(110)
eZ = np.random.normal(m_e[3], C[3], N[1])
Z = (eZ - U).round(2)
Z_conU = (eZ - U_inst).round(2)

random.seed(120)
eW = np.random.normal(m_e[4], C[4], N[1])
W0 = fun_w(U)
W = (eW + W0).round(2)
W_conU = (eW + fun_w(U_inst)).round(2)

random.seed(130)
eA = np.random.normal(m_e[1], C[1], size=N[1])
# A  = (eA + a_AZ * Z  +2* U**0.5).round(3)
# A_conU=(eA + a_AZ * Z_conU +2* U_inst**0.5).round(3)
A = (eA + U).round(2)
A_conU = (eA + U_inst).round(2)

random.seed(19500)
eY = np.random.normal(m_e[2], C[2], N[1])
# Y  = (np.exp(a_AY *A)+ (eY   -np.log10(U+10))).round(3)
# Y_conU=(np.exp(a_AY * A_conU)+ eY -np.log10(U_inst+10)).round(3)
Y0 = fun_y(a=A, b=a_AY)
Y = (Y0 + eY + U).round(2)
Y_conU = (Y0 + eY + (U_inst)).round(2)

D = pd.DataFrame([U, A, Y, Z, W]).T
D.columns = ['U', 'A', 'Y', 'Z', 'W']
O = pd.DataFrame([A, Y, Z, W]).T
O.columns = ['A', 'Y', 'Z', 'W']
D_conU = pd.DataFrame([U_inst, A_conU, Y_conU, Z_conU, W_conU]).T
D_conU.columns = ['U', 'A', 'Y', 'Z', 'W']
# plt.scatter(data=D, x='A',y='Y')

#%%Kernel parameters
l_1=0.1
l_2=.1
n_test=int(N[1]*.1)
n_train=N[1]-n_test
m1_test=int(n_test*.5)
m2_test=n_test-m1_test
m1_train=int(n_train*0.5)## I have to change it back to 50%
m2_train=n_train-m1_train
low_b=.001

print(N[1], n_test, m1_test, m2_test, n_train, m1_train, m2_train)

# %% corr structure
ecov_v = pd.DataFrame.cov(D)

ecorr_v = D.corr()
ecorr_v.columns = ['U', 'A', 'Y', 'Z', 'W']
ecorr_O = O.corr()
ecorr_O.columns = ['A', 'Y', 'Z', 'W']
ecov_v_conU = pd.DataFrame.cov(D_conU)
ecov_v_conU.columns = ['U', 'A', 'Y', 'Z', 'W']
ecorr_v_conU = D_conU.corr()
ecorr_v_conU.columns = ['U', 'A', 'Y', 'Z', 'W']

O_train, O_test = train_test_split(O, test_size=n_test, train_size=n_train)

samp1, samp2 = train_test_split(O_train, test_size=m2_train, train_size=m1_train)
samp1_test, samp2_test = train_test_split(O_test, test_size=m2_test, train_size=m1_test)

# %%
sns.displot(D, x='U', label="U", kde=True), plt.show()
sns.displot(D, x='A', label="A", kde=True), plt.show()
sns.displot(D, x='Y', label="y", kde=True), plt.show()
sns.displot(D, x='Z', label="Z", kde=True), plt.show()
sns.displot(D, x='W', label="W", kde=True), plt.show()

sns.set_theme(font="tahoma", font_scale=1) # this gives us the pariswise plots


sns.displot(O, x='A', y='Y', kind="kde"), plt.show()
sns.displot(O, x='Z', y='Y', kind="kde"), plt.show()
sns.displot(O, x='W', y='Y', kind="kde"), plt.show()
sns.displot(D, x='U', y='Y', kind="kde"), plt.show()

# sns.heatmap(ecorr_v,annot=True, fmt='g'),plt.show()
sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.show()
sns.heatmap(ecorr_O, annot=True, fmt=".2"), plt.show()

# sns.heatmap(ecov_v_conU,annot=True, fmt='g'),plt.show()

# sns.heatmap(ecorr_v_conU,annot=True, fmt='g'),plt.show()
sns.heatmap(ecorr_v_conU, annot=True, fmt=".2"), plt.show()