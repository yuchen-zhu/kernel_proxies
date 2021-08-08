#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:58:08 2020

@author: afsaneh
"""
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import random
import scipy.sparse as sp
import matplotlib as mpl
import statistics 
from keras.models import Sequential
import tensorflow

import math
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import os


import itertools as it

from sklearn.linear_model import LinearRegression as LR
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessClassifier as GPC

##%matplotlib inline
# plt.show()
# sns.set_theme(font="tahoma", font_scale=0.6)


#%%
#param
N = [5,5000,10000,100000, 500000]
# [a,y,z,w] heat as a catalyst z temprature w 
a_AY = 0.5
a_AZ = 0.5

# [u,a,y,z,w]
m_e = [5, 0, 0, -1, 2]
# cov_e = [[1],[,1],[,,1],[,,,1]]
C = [1,1,2,1,4]

def main():
    # U is a chi2 distribution
    np.random.seed(100)
    U = np.random.chisquare(m_e[0], N[-1]).round(3)  # generates 5000 U's to 3.d.p.
    train_u, test_u, dev_u, rest_u = U[:3000], U[3000:4000], U[4000:5000], U[5000:]
    U_inst = np.ones(N[-1]).round(3)

    X = np.random.chisquare(m_e[0], N[-1]).round(3)  # generates 5000 X's to 3.d.p.
    train_x, test_x, dev_x, rest_x = X[:3000], X[3000:4000], X[4000:5000], X[5000:]
    X_inst = np.ones(N[-1]).round(3)

    # Z is noisy reading of U
    # random.seed(110)
    eZ = np.random.normal(m_e[3], C[3], N[-1])  # noise for Z
    Z = (eZ - U).round(3)
    train_z, test_z, dev_z, rest_z = Z[:3000], Z[3000:4000], Z[4000:5000], Z[5000:]
    Z_conU = (eZ - U_inst).round(3)  # TODO: what is this? constant U, or confounded by U?


    # random.seed(120)  # noise for W
    eW = np.random.normal(m_e[4], C[4], N[-1])
    W = (eW + 2 * U).round(3)
    train_w, test_w, dev_w, rest_w = W[:3000], W[3000:4000], W[4000:5000], W[5000:]
    W_conU = (eW + 2 * U_inst).round(3)


    # random.seed(130)
    eA = np.random.normal(m_e[1], C[1], N[-1])
    # A = (eA + a_AZ * Z + 2 * U ** 0.5).round(3)
    A = (eA + 2 * U ** 0.5).round(3)  # this is the structural equation without dependence on Z
    train_a, test_a, dev_a, rest_a = A[:3000], A[3000:4000], A[4000:5000], A[5000:]
    A_conU = (eA + a_AZ * Z_conU + 2 * U_inst ** 0.5).round(3)

    # random.seed(19500)
    eY = np.random.normal(m_e[2], C[2], N[-1])
    Y = (np.exp(a_AY * A) + eY + -np.log10(U)).round(3)
    train_y, test_y, dev_y, rest_y = Y[:3000], Y[3000:4000], Y[4000:5000], Y[5000:]
    Y_conU = (np.exp(a_AY * A_conU) + eY - np.log10(U_inst)).round(3)

    # causal ground truth
    do_A = np.linspace(1, 20, 20)
    EY_do_A = []
    for a in do_A:
        A_ = np.repeat(a, [N[1]])
        Y_do_A = (np.exp(a_AY * A_) + eY[:N[1]] + -np.log10(U[:N[1]])).round(3)
        eY_do_A = np.mean(Y_do_A)
        EY_do_A.append(eY_do_A)

    EY_do_A = np.array(EY_do_A)

    # D = pd.DataFrame([U,A,Y,Z,W]).T
    # D.columns = ['U', 'A', 'Y', 'Z', 'W']
    # O = pd.DataFrame([A, Y, Z, W]).T
    # O.columns = ['A', 'Y', 'Z', 'W']
    # D_conU = pd.DataFrame([U_inst, A_conU, Y_conU, Z_conU, W_conU]).T
    # D_conU.columns = ['U', 'A', 'Y', 'Z', 'W']

    # convert to dict and save to npz format.

    # data_dict = {"splits": ['train', 'test', 'dev'],
    #              "train_y": Y, "train_a": A, "train_z": Z, "train_w": W, "train_u": U}

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/zoo/sim_1d_no_x/main_orig.npz'),
             splits=['train', 'test', 'dev'],
             train_y=train_y,
             train_a=train_a,
             train_z=train_z,
             train_w=train_w,
             train_u=train_u,
             test_y = test_y,
             test_a = test_a,
             test_z = test_z,
             test_w = test_w,
             test_u = test_u,
             dev_y = dev_y,
             dev_a = dev_a,
             dev_z = dev_z,
             dev_w = dev_w,
             dev_u = dev_u)

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/zoo/sim_1d_no_x/do_A_orig.npz'),
             do_A = do_A,
             gt_EY_do_A = EY_do_A)

    plot_dict = {'U': U[:100],
                 'Z': Z[:100],
                 'W': W[:100],
                 'A': A[:100],
                 'Y': Y[:100]}

    # plt.scatter(plot_dict['U'], plot_dict['Y'])
    # plt.show()
    # plt.close()

    #  make scatter plots for visualisation
    fig, axs = plt.subplots(5, 5)

    print('U: {} - {}, Z: {} - {}, W: {} - {}, A: {} - {}, Y: {} - {}'.format(np.max(U[:100]),
                                                                              np.min(U[:100]),
                                                                              np.max(Z[:100]),
                                                                              np.min(Z[:100]),
                                                                              np.max(W[:100]),
                                                                              np.min(W[:100]),
                                                                              np.max(A[:100]),
                                                                              np.min(A[:100]),
                                                                              np.max(Y[:100]),
                                                                              np.min(Y[:100])))



    for idx1, var1 in enumerate(plot_dict.keys()):
        for idx2, var2 in enumerate(plot_dict.keys()):
            if idx1 == idx2:
                print('plotting {} against {} at the ({}, {}) position in the grid'.format(var1, var1, idx1, idx2))
                axs[idx1, idx2].hist(plot_dict[var1], 20)
            else:
                print('plotting {} against {} at the ({}, {}) position in the grid'.format(var1, var2, idx1, idx2))
                axs[idx1, idx2].scatter(plot_dict[var1], plot_dict[var2], marker='.')

    for idx, label in enumerate(plot_dict.keys()):
        axs[-1, idx].set_xlabel(label)
        axs[idx, 0].set_ylabel(label)


    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scatter_plot_{}.png'.format(time.time())))
    plt.close()

    fig_do_A = plt.figure()
    plt.plot(do_A, EY_do_A)
    plt.xlabel('A')
    plt.ylabel('E[Y|do(A)]')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ground_truth_EYdoA_{}.png'.format(time.time())))
    plt.close()

    print('expectation evaluation starts.')
    test_sample_sz = 1000
    w_sample_thresh = 20
    axz = np.vstack([test_a, test_x, test_z])  # shape: 3 x 1000
    axzwy = np.vstack([rest_a, rest_x, rest_z, rest_w, rest_y])
    gen_eval_samples(test_sample_size=test_sample_sz,
                     w_sample_thresh=w_sample_thresh,
                     axz=axz,
                     axzwy=axzwy)


# U is a chi2 distribution
def gen_u(u_size):
    np.random.seed(100)
    U = np.random.chisquare(m_e[0], u_size).round(3)  # generates 5000 U's to 3.d.p.
    train_u, test_u, dev_u = U[:N[1] - 2000], U[-2000:-1000], U[-1000:]
    U_inst = np.ones(N[1]).round(3)
    return U

# Z is noisy reading of U
def gen_z(z_size):
    np.random.seed(100)
    U = np.random.chisquare(m_e[0], z_size).round(3)  # generates 5000 U's to 3.d.p.
    eZ = np.random.normal(m_e[3], C[3], z_size)  # noise for Z
    Z = (eZ - U).round(3)
    train_z, test_z, dev_z = Z[:N[1] - 2000], Z[-2000:-1000], Z[-1000:]
    Z_conU = (eZ - U_inst).round(3)  # TODO: what is this? constant U, or confounded by U?
    return Z

def gen_w(w_size):
    np.random.seed(100)
    U = np.random.chisquare(m_e[0], w_size).round(3)  # generates 5000 U's to 3.d.p.
    # noise for W
    eW = np.random.normal(m_e[4], C[4], w_size)
    W = (eW + 2 * U).round(3)
    train_w, test_w, dev_w = W[:N[1] - 2000], W[-2000:-1000], W[-1000:]
    W_conU = (eW + 2 * U_inst).round(3)
    return W

def gen_eval_samples(test_sample_size, w_sample_thresh, axz, axzwy):
    inp = input('Check the input order is a, x, z. ans: y/n')
    if inp == 'y':
        pass
    else:
        raise ValueError('incorrectly ordered input.')

    assert axz.shape[0] == 3
    axz = axz[:, :test_sample_size]
    assert axz.shape[1] == test_sample_size

    assert axzwy.shape[0] == 5
    # assert axzwy.shape[1] == test_sample_size * 1000

    axz_out, y_av_out, w_samples_out, y_samples_out = [], [], [], []

    for i in range(test_sample_size):
        axz_all = axzwy[:3,:]
        axz_diff = axz_all - axz[:, i:i+1]
        print('axz_all: ', axz_all, '\n', 'axz vec: ', axz[:,i:i+1], '\n', 'difference: ', axz_diff)
        axz_valid_idx = (axz_diff > -0.12) * (axz_diff < 0.12)
        axz_valid_col_idx = np.prod(axz_valid_idx, axis=0)
        print('valid row idx: ', axz_valid_col_idx)
        num_valid = np.sum(axz_valid_col_idx)
        print('num valid: ', num_valid)
        if num_valid < w_sample_thresh:
            continue
        else:
            axz_valid_col_idx = np.nonzero(axz_valid_col_idx)
            print('valid indices: ', axz_valid_col_idx)
            subTuple = np.squeeze(axzwy[:, axz_valid_col_idx])
            subTuple = subTuple[:, :w_sample_thresh]
            y_axz_av = np.mean(subTuple[-1, :])
            y_samples_out.append(subTuple[-1, :])
            axz_out.append(axz[:, i])
            y_av_out.append(y_axz_av)
            w_samples_out.append(subTuple[-2,:])
            print('subTuples: ', subTuple)
            print('axzwy: ', axzwy)
            print('w_samples: ', subTuple[-2, :])
    axz_np = np.array(axz_out)
    y_np = np.array(y_av_out)
    axzy_np = np.concatenate([axz_np, y_np.reshape(-1,1)], axis=1)
    w_samples_out_np = np.array(w_samples_out)
    y_samples_out_np = np.array(y_samples_out)
    print('num eval tuples: ', axzy_np.shape[0], 'axzy: ', axzy_np, 'w_samples: ', w_samples_out_np)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/zoo/sim_1d_no_x/cond_exp_metric_orig.npz'),
             axzy=axzy_np,
             w_samples=w_samples_out_np,
             y_samples=y_samples_out_np)


if __name__ == "__main__":
    main()







