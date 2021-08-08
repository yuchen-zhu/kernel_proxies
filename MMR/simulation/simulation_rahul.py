import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn import preprocessing
from scipy.stats import norm

# TODO: in the final simulation, U should not be from a uniform distribution, bc unif is not exponential family - use normal or chi-squared. UPDATE: actually this might not matter.
# seeds = [100, 200, 300, 400, 500]
seeds = [100, 200]

do_A_min, do_A_max, n_A = -2., 2., 30

N = [500, 5000, 10000, 100000, 10000]
train_sz = 1000

m_e = [5, 0, 0, -1, 2]
C = [3, 3, 3, 3, 3, 2]

eps = 1e-2

# centres_no_w = np.array([[0., 4.], [0., 12.], [0., 20.], [0., 28.], [0., 36.]])
# centres_w = np.array([[2., -5., 8.], [4., -2., 12.], [6., 0., 15.], [8., 0., 20.], [10., 5., 25.]])
# coeffs_ = np.array([1., 1., 1., 1., 1.])
# sig_ = 2.

centres_u = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]])


parser = argparse.ArgumentParser(description='settings for simulation')
parser.add_argument('--sem', type=str, help='sets SEM')

args = parser.parse_args()

def data_transform(X):
    scaler = preprocessing.StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler

# def asina(a, b):
#     return np.sin(b * a) * a
#
#
# def asina_mild(a, b):
#     return np.sin(b * a) * a * b
#
#
# def expa(a, b):
#     return np.exp(b * a)
#
#
# def linear(a, b):
#     return b * a
#
#
# def get_a_inpsininp(inp):
#     return 3 * np.sin(inp) + 1.5*inp
#
#
# def get_a_quadratic(inp):
#     return -1 * (inp - 5) ** 2 +10
#
#
# def get_y_rbf(u, w, a, centres, sig, coeffs):
#     """
#     input: u, w, a - shape = N x dim
#     """
#
#     u = u.reshape(-1, 1) if len(u.shape) == 1 else None
#     w = w.reshape(-1, 1) if len(w.shape) == 1 else None
#     a = a.reshape(-1, 1) if len(a.shape) == 1 else None
#
#     uwa = np.concatenate([u,w,a], axis=-1)
#     # centres = np.repeat(centres, uwa.shape[-1], axis=-1).reshape(-1, uwa.shape[-1])
#
#     CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1,1), uwa.shape[0], axis=-1)
#     UWAUWA = np.repeat(np.sum(uwa ** 2, axis=-1).reshape(1,-1), centres.shape[0], axis=0)
#     CUWA = centres @ uwa.T
#
#     H = np.exp(-1 * (CC + UWAUWA - 2 * CUWA)/2/sig/sig)
#     Y = coeffs.dot(H)
#
#     return Y
#
#
# def get_y_rbf_no_w(u, a, centres, sig, coeffs):
#     """
#     input: u, a - shape = N x dim
#     """
#     # print(len(u.shape))
#     u = u.reshape(-1,1) if len(u.shape) == 1 else None
#     a = a.reshape(-1, 1) if len(a.shape) == 1 else None
#
#     ua = np.concatenate([u,a], axis=-1)
#     # centres = np.repeat(centres, ua.shape[-1], axis=-1).reshape(-1, ua.shape[-1])
#
#     CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1,1), ua.shape[0], axis=-1)
#     UAUA = np.repeat(np.sum(ua ** 2, axis=-1).reshape(1,-1), centres.shape[0], axis=0)
#     # print(ua.shape)
#     CUWA = centres @ ua.T
#
#     H = np.exp(-1 * (CC + UAUA - 2 * CUWA)/2/sig/sig)
#     Y = coeffs.dot(H)
#
#     return Y


# func_dict = {'asina': asina, 'expa': expa, 'linear': linear, 'asina_mild': asina_mild, 'rbf_y': get_y_rbf, 'rbf_y_no_w': get_y_rbf_no_w}


# def gen_eval_samples(test_sample_size, w_sample_thresh, axz, axzwy):
#     inp = input('Check the input order is a, x, z. ans: y/n')
#     if inp == 'y':
#         pass
#     else:
#         raise ValueError('incorrectly ordered input.')
#
#     assert axz.shape[0] == 4
#     axz = axz[:, :test_sample_size]
#     assert axz.shape[1] == test_sample_size
#
#     assert axzwy.shape[0] == 7
#     # assert axzwy.shape[1] == test_sample_size * 1000
#
#     axz_out, y_av_out, w_samples_out, y_samples_out = [], [], [], []
#
#     for i in range(test_sample_size):
#         axz_all = axzwy[:4,:]
#         axz_diff = axz_all - axz[:, i:i+1]
#         print('axz_all: ', axz_all, '\n', 'axz vec: ', axz[:,i:i+1], '\n', 'difference: ', axz_diff)
#         axz_valid_idx = (axz_diff > -0.12) * (axz_diff < 0.12)
#         axz_valid_col_idx = np.prod(axz_valid_idx, axis=0)
#         print('valid row idx: ', axz_valid_col_idx)
#         num_valid = np.sum(axz_valid_col_idx)
#         print('num valid: ', num_valid)
#         if num_valid < w_sample_thresh:
#             continue
#         else:
#             axz_valid_col_idx = np.nonzero(axz_valid_col_idx)
#             print('valid indices: ', axz_valid_col_idx)
#             subTuple = np.squeeze(axzwy[:, axz_valid_col_idx])
#             subTuple = subTuple[:, :w_sample_thresh]
#             y_axz_av = np.mean(subTuple[-1, :])
#             y_samples_out.append(subTuple[-1, :])
#             axz_out.append(axz[:, i])
#             y_av_out.append(y_axz_av)
#             w_samples_out.append(subTuple[-2,:])
#             print('subTuples: ', subTuple)
#             print('axzwy: ', axzwy)
#             print('w_samples: ', subTuple[-2, :])
#     axz_np = np.array(axz_out)
#     y_np = np.array(y_av_out)
#     axzy_np = np.concatenate([axz_np, y_np.reshape(-1,1)], axis=1)
#     w_samples_out_np = np.array(w_samples_out)
#     y_samples_out_np = np.array(y_samples_out)
#     print('num eval tuples: ', axzy_np.shape[0], 'axzy: ', axzy_np, 'w_samples: ', w_samples_out_np)
#
#     return axzy_np, w_samples_out_np, y_samples_out_np


def main(args, seed):
    print('start')
    np.random.seed(seed)

    e1, e2, e3 = np.random.normal(0, 1, N[-1]), np.random.normal(0, 1, N[-1]), np.random.normal(0, 1, N[-1])
    v1, v2 = np.random.uniform(-1, 1, N[-1]), np.random.uniform(-1, 1, N[-1])

    U1 = (e1 + e3)
    U2 = (e2 + e3)

    U = np.concatenate([U1.reshape(-1,1), U2.reshape(-1,1)], axis=-1)
    train_u, test_u, dev_u, rest_u = U[:train_sz], \
                                     U[train_sz:train_sz + 1000], \
                                     U[train_sz + 1000:train_sz + 2000], \
                                     U[train_sz + 2000:]

    Z = v1 + 0.25 * U1
    Z_scaled, Z_scaler = data_transform(Z)
    train_z, test_z, dev_z, rest_z = Z_scaled[:train_sz], \
                                     Z_scaled[train_sz:train_sz + 1000], \
                                     Z_scaled[train_sz + 1000:train_sz + 2000], \
                                     Z_scaled[train_sz + 2000:]

    W = v2 + 0.25 * U2
    W_scaled, W_scaler = data_transform(W)
    train_w, test_w, dev_w, rest_w = W_scaled[:train_sz], \
                                     W_scaled[train_sz:train_sz + 1000], \
                                     W_scaled[train_sz + 1000:train_sz + 2000], \
                                     W_scaled[train_sz + 2000:]

    print('generated U,Z,W')

    A = norm.cdf(3 * Z) + 0.25 * U2
    A_scaled, A_scaler = data_transform(A)
    # print('shape of A: ', A_scaled.shape)
    train_a, test_a, dev_a, rest_a = A_scaled[:train_sz], \
                                     A_scaled[train_sz:train_sz + 1000], \
                                     A_scaled[train_sz + 1000:train_sz + 2000], \
                                     A_scaled[train_sz + 2000:]

    print('generated A')

    X = np.zeros((N[-1],))
    train_x, test_x, dev_x, rest_x = X[:train_sz], \
                                     X[train_sz:train_sz + 1000], \
                                     X[train_sz + 1000:train_sz + 2000], \
                                     X[train_sz + 2000:]


    Y = 1.2 * (A + W) + A**2 + 0.25 * U1
    Y_scaled, Y_scaler = data_transform(Y)
    train_y, test_y, dev_y, rest_y = Y_scaled[:train_sz], \
                                     Y_scaled[train_sz:train_sz + 1000], \
                                     Y_scaled[train_sz + 1000:train_sz + 2000], \
                                     Y_scaled[train_sz + 2000:]

    print('generated Y')

    # causal ground truth
    do_A_scaled = np.linspace(do_A_min, do_A_max, n_A)
    do_A = A_scaler.inverse_transform(do_A_scaled).squeeze()
    do_A_save = do_A_scaled
    EY_do_A = []
    for a in do_A:
        A__ = np.repeat(a, [N[1]]).reshape(N[1],1)
        Y_do_A = 1.2 * (A__ + W) + A__**2 + 0.25 * U1
        Y_do_A_scaled = Y_scaler.transform(Y_do_A)
        eY_do_A = np.mean(Y_do_A_scaled)
        EY_do_A.append(eY_do_A)


    EY_do_A = np.array(EY_do_A)

    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/main_{}_seed{}.npz'.format(args.sem, seed)),
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


    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seed)),
             do_A = do_A_save,
             gt_EY_do_A = EY_do_A)

    # plotting
    D = pd.DataFrame([U[:500, 0], U[:500, 1], train_a[:500], train_y[:500], train_z[:500],
                      train_w[:500]]).T
    D.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']
    # O = pd.DataFrame([train_a[:300], train_y[:300], train_z[:300], train_w[:300]]).T
    # O.columns = ['A', 'Y', 'Z', 'W']
    D_doA = pd.DataFrame([do_A_save, EY_do_A]).T
    D_doA.columns = ['A', 'EY_do_A']

    ecorr_v = D.corr()
    ecorr_v.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']
    # ecorr_O = O.corr()
    # ecorr_O.columns = ['A', 'Y', 'Z', 'W']
    # ecorr_v_conU = D_conU.corr()
    # ecorr_v_conU.columns = ['U', 'A', 'Y', 'Z', 'W']


    sem = args.sem
    if not os.path.exists(PATH+sem + '_seed' + str(seed)):
        os.mkdir(PATH + sem + '_seed' + str(seed))
    for v in ['U1', 'U2', 'A', 'Y', 'Z', 'W']:
        sns.displot(D, x=v, label=v, kde=True), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + v + '_dist.png'), plt.close()

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'full_pairwise.png'), plt.close()
    # sns.pairplot(O), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'observed_pairwise.png'), plt.close()
    sns.pairplot(D_doA), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'ate_pairwise.png'), plt.close()

    sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'corr_all.png'), plt.close()
    # sns.heatmap(ecorr_O, annot=True, fmt=".2"), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'corr_observed.png'), plt.close()
    # sns.heatmap(ecorr_v_conU, annot=True, fmt=".2"), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'corr_fixed_U.png'), plt.close()





if __name__ == '__main__':
    for seed in seeds:
        main(args, seed)