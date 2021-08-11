# Arthur simulation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn import preprocessing
import pandas as pd
import seaborn as sns


PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
seeds = [100, 200, 300, 400, 500]

do_A_min, do_A_max, n_A = -2., 2., 100

# number of samples
N = [500, 5000, 10000, 100000, 80000]
train_sz = 50000

aStd = 0.05
wStd = 3

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


def main(args, seed):
    print('start')
    np.random.seed(seed)

    U2 = np.random.rand(N[-1], 1) * 3 - 1
    U1 = np.random.rand(N[-1], 1) - ((U2 > 0) & (U2 < 1)).astype(int)

    plt.plot(U1, U2, 'b.')
    plt.xlabel('u1')
    plt.ylabel('u2')

    os.makedirs(PATH + args.sem + '_seed' + str(seed) + '/', exist_ok=True)
    plt.savefig(PATH + args.sem + '_seed' + str(seed) + '/' + 'u1u2.png'), plt.close()

    U = np.concatenate([U1, U2], axis=-1)
    train_u, test_u, dev_u, rest_u = U[:train_sz], U[train_sz:train_sz + 1000], U[train_sz + 1000:train_sz + 2000], U[train_sz+2000:]

    Z = U2 + np.random.rand(N[-1], 1)
    Z_scaled, Z_scaler = data_transform(Z)
    train_z, test_z, dev_z, rest_z = Z_scaled[:train_sz].reshape(-1,1), Z_scaled[train_sz:train_sz+1000].reshape(-1,1), Z_scaled[train_sz+1000:train_sz+2000].reshape(-1,1), Z_scaled[train_sz+2000:].reshape(-1,1)


    W = U1 + np.random.rand(N[-1], 1) * wStd
    W_scaled, W_scaler = data_transform(W)
    train_w, test_w, dev_w, rest_w = W_scaled[:train_sz].reshape(-1,1), W_scaled[train_sz:train_sz+1000].reshape(-1,1), W_scaled[train_sz+1000:train_sz+2000].reshape(-1,1), W_scaled[train_sz+2000:].reshape(-1,1)

    assert train_w.ndim == 2
    assert train_z.ndim == 2

    print('generated U,Z,W')


    A = U2 + np.random.randn(*U2.shape) * aStd
    A_scaled, A_scaler = data_transform(A)
    train_a, test_a, dev_a, rest_a = A_scaled[:train_sz].reshape(-1,1), A_scaled[train_sz:train_sz+1000].reshape(-1,1), A_scaled[train_sz+1000:train_sz+2000].reshape(-1,1), A_scaled[train_sz+2000:].reshape(-1,1)

    assert train_a.ndim == 2
    print('generated A')

    Y = U2 * np.cos(2 * (A + 0.3 * U1 + 0.2))
    Y_scaled, Y_scaler = data_transform(Y)
    train_y, test_y, dev_y, rest_y = Y_scaled[:train_sz].reshape(-1,1), Y_scaled[train_sz:train_sz+1000].reshape(-1,1), Y_scaled[train_sz+1000:train_sz+2000].reshape(-1,1), Y_scaled[train_sz+2000:].reshape(-1,1)

    assert train_y.ndim == 2
    print('generated Y')


    ##########################
    # backdoor: ground truth #
    ##########################

    # causal ground truth
    do_A_scaled = np.linspace(do_A_min, do_A_max, n_A)
    do_A = A_scaler.inverse_transform(do_A_scaled).squeeze()
    do_A_save = do_A_scaled
    nVal = 10000
    EY_do_A = np.zeros((n_A, 1))
    for indA in range(n_A):
        u2_BkD = np.random.rand(nVal, 1) * 3 - 1
        u1_BkD = np.random.rand(nVal, 1) - ((u2_BkD > 0) & (u2_BkD < 1))
        EY_do_A[indA] = np.mean(u2_BkD * np.cos(2 * (do_A[indA]) + 0.3 * u1_BkD + 0.2))
    EY_do_A_scaled = Y_scaler.transform(EY_do_A).reshape(-1,1)


    #################################
    # Marginalisation: ground truth #
    #################################

    yMar = np.zeros((n_A, 1))

    for indA in range(n_A):
        u2_mar = do_A[indA] + np.random.randn(nVal, 1) * aStd
        u1_mar = np.random.rand(nVal, 1) - ((u2_mar > 0) & (u2_mar < 1))

        # debug: check that conditioning is correct
        if 0:
            aAx[indA]
            plt.clf()
            plt.plot(u1_mar, u2_mar, '.')
            plt.plot(u1, u2, 'r.')
            plt.show()

        yMar[indA] = np.mean(Y_scaler.transform(u2_mar * np.cos(2 * (do_A[indA])) + 0.3 * u1_mar + 0.2))

    plt.plot(do_A_scaled, EY_do_A_scaled, label='causal-gt')
    plt.plot(do_A_scaled, yMar, label='marginalisation')
    plt.legend()
    plt.savefig(PATH + args.sem + '_seed' + str(seed) + '/' + 'bkd-marg.png'), plt.close()

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
             gt_EY_do_A = EY_do_A_scaled)


    # plotting
    D = pd.DataFrame([U[:500, 0], U[:500, 1], train_a[:500].flatten(), train_y[:500].flatten(), train_z[:500].flatten(),
                      train_w[:500].flatten()]).T
    D.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']
    D_doA = pd.DataFrame([do_A_save.flatten(), EY_do_A_scaled.flatten()]).T
    D_doA.columns = ['A', 'EY_do_A']

    ecorr_v = D.corr()
    ecorr_v.columns = ['U1', 'U2', 'A', 'Y', 'Z', 'W']


    sem = args.sem
    if not os.path.exists(PATH+sem + '_seed' + str(seed)):
        os.mkdir(PATH + sem + '_seed' + str(seed))
    for v in ['U1', 'U2', 'A', 'Y', 'Z', 'W']:
        sns.displot(D, x=v, label=v, kde=True), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + v + '_dist.png'), plt.close()

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'full_pairwise.png'), plt.close()
    sns.pairplot(D_doA), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'ate_pairwise.png'), plt.close()

    sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'corr_all.png'), plt.close()


if __name__ == '__main__':
    for seed in seeds:
        main(args, seed)
