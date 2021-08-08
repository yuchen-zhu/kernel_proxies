import os, sys
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv, bundle_az_aw, visualise_ATEs
from joblib import Parallel, delayed
import time
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import date
import argparse
import random

Nfeval = 1
# seed = 527
# np.random.seed(seed)
JITTER = 1e-7
nystr_M = 2000
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None

parser = argparse.ArgumentParser(description='run rhks model with general gaussian-based kernel')

parser.add_argument('--av-kernel', type=bool, default=False, help='use single bandwidth or average')
parser.add_argument('--sem', type=str, help='set which SEM to use data from')
args = parser.parse_args()
print('parsed av-kernel: ', args.av_kernel)


def experiment(sname, seed, datasize, nystr=False, args=None):
    np.random.seed(seed)
    random.seed(seed)
    def LMO_err(params, M=10):
        # np.random.seed(2)
        # random.seed(2)
        al, bl = np.exp(params)
        print('lmo_err params', params)
        print('lmo_err al, bl', al, bl)
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y * N2
        c_y = c - Y
        lmo_err = 0
        N = 0
        for ii in range(1):
            permutation = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices, indices)] * N2
                C_i = C[np.ix_(indices, indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                lmo_err += b_y.T @ K_i @ b_y
                N += 1
        return lmo_err[0, 0] / N / M ** 2

    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        # np.random.seed(3)
        # random.seed(3)
        if Nfeval % 1 == 0:
            al, bl = np.exp(params)
            print('callback al, bl', al, bl)
            L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
            if nystr:
                alpha = EYEN - eig_vec_K @ np.linalg.inv(
                    eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                alpha = alpha @ W_nystr @ Y * N2
            else:
                LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                alpha = LWL_inv @ L @ W @ Y
                # L_W_inv = chol_inv(W*N2+L_inv)
            test_L = bl * bl * np.exp(-test_L0 / al / al / 2)
            pred_mean = test_L @ alpha
            if timer:
                return
            test_err = ((pred_mean - test_Y) ** 2).mean()  # ((pred_mean-test_Y)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha

        Nfeval += 1
        if prev_norm is not None:
            if norm[0, 0] / prev_norm >= 3:
                if opt_params is None:
                    opt_test_err = test_err
                    opt_params = params
                print(True, opt_params, opt_test_err, prev_norm)
                raise Exception

        if prev_norm is None or norm[0, 0] <= prev_norm:
            prev_norm = norm[0, 0]
        opt_test_err = test_err
        opt_params = params
        print('params,test_err, norm: ', opt_params, opt_test_err, norm[0, 0])

    def get_causal_effect(params, do_A, w):
        "to be called within experiment function."
        # np.random.seed(4)
        # random.seed(4)
        # al, bl = np.exp(params)
        al, bl = params
        print('al, bl = params')
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        if nystr:
            alpha = EYEN - eig_vec_K @ np.linalg.inv(
                eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
            alpha = alpha @ W_nystr @ Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            alpha = LWL_inv @ L @ W @ Y
            # L_W_inv = chol_inv(W*N2+L_inv)

        EYhat_do_A = []
        for a in do_A:
            a = np.repeat(a, [w.shape[0]]).reshape(-1, 1)
            w = w.reshape(-1, 1)
            aw = np.concatenate([a, w], axis=-1)
            ate_L0 = _sqdist(aw, X)
            ate_L = bl * bl * np.exp(-ate_L0 / al / al / 2)
            h_out = ate_L @ alpha

            mean_h = np.mean(h_out).reshape(-1, 1)
            EYhat_do_A.append(mean_h)
            print('a = {}, beta_a = {}'.format(np.mean(a), mean_h))

        return np.concatenate(EYhat_do_A)

    def plot_h(params, inp_sample):
        "helper function to plot h."
        # np.random.seed(4)
        # random.seed(4)
        # al, bl = np.exp(params)
        al, bl = params
        print('al, bl = params')
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        if nystr:
            alpha = EYEN - eig_vec_K @ np.linalg.inv(
                eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
            alpha = alpha @ W_nystr @ Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            alpha = LWL_inv @ L @ W @ Y
            # L_W_inv = chol_inv(W*N2+L_inv)

        a_sample, w_sample = inp_sample[:, 0], inp_sample[:, 1]
        a_max, a_min, w_max, w_min = a_sample.max(), a_sample.min(), w_sample.max(), w_sample.min()
        a_linspace, w_linspace = np.linspace(a_min, a_max, 100), np.linspace(w_min, w_max, 100)
        aa, ww = np.meshgrid(a_linspace, w_linspace)
        a_s, w_s = aa.flatten(), ww.flatten()

        aw = np.stack([a_s,w_s], axis=-1)
        ate_L0 = _sqdist(aw, X)
        ate_L = bl * bl * np.exp(-ate_L0 / al / al / 2)
        h_out = ate_L @ alpha

        a_s, w_s, h_out = a_s.reshape(100, -1), w_s.reshape(100, -1), h_out.reshape(100, -1)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(a_s, w_s, h_out, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('A'), ax.set_ylabel('W'), ax.set_zlabel('h')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(PATH, str(date.today()), args.sem, 'h_trainsz{}_seed{}'.format(AW_train.shape[0], seed) + '.png'))
        plt.close()



    # train,dev,test = load_data(ROOT_PATH+'/data/zoo/{}_{}.npz'.format(sname,datasize))

    # X = np.vstack((train.x,dev.x))
    # Y = np.vstack((train.y,dev.y))
    # Z = np.vstack((train.z,dev.z))
    # test_X = test.x
    # test_Y = test.g
    t1 = time.time()
    train, dev, test = load_data(ROOT_PATH + "/data/zoo/" + sname + '/main_{}.npz'.format(args.sem))
    # train, dev, test = train[:300], dev[:100], test[:100]
    t2 = time.time()
    print('t2 - t1 = ', t2 - t1)
    Y = np.concatenate((train.y, dev.y), axis=0).reshape(-1, 1)
    # test_Y = test.y
    AZ_train, AW_train = bundle_az_aw(train.a, train.z, train.w)
    AZ_test, AW_test = bundle_az_aw(test.a, test.z, test.w)
    AZ_dev, AW_dev = bundle_az_aw(dev.a, dev.z, test.w)

    X, Z = np.concatenate((AW_train, AW_dev), axis=0), np.concatenate((AZ_train, AZ_dev), axis=0)
    test_X, test_Y = AW_test, test.y.reshape(-1, 1)  # TODO: is test.g just test.y?

    t3 = time.time()
    print('t3 - t2', t3-t2)
    EYEN = np.eye(X.shape[0])
    # ak0, ak1 = get_median_inter_mnist(Z[:, 0:1]), get_median_inter_mnist(Z[:, 1:2])
    ak = get_median_inter_mnist(Z)
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z, None)
    print('av kernel indicator: ', args.av_kernel)
    # W = np.exp(-W0 / ak0 / ak0 / 2) / N2 if not args.av_kernel \
    #     else (np.exp(-W0 / ak0 / ak0 / 2) + np.exp(-W0 / ak0 / ak0 / 200) + np.exp(-W0 / ak0 / ak0 * 50)) / 3 / N2
    W = np.exp(-W0 / ak / ak / 2) / N2 if not args.av_kernel \
        else (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2

    del W0
    L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)
    t4 = time.time()
    print('t4 - t3', t4-t3)
    # measure time
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return

    params0 = np.random.randn(2)  # /10
    # params0 = np.array([1., 10.])
    print('starting param: ', params0)
    bounds = None  # [[0.01,10],[0.01,5]]
    if nystr:
        # for _ in range(seed + 1):
        for _ in range(1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
        W_nystr_Y = W_nystr @ Y

    t5 = time.time()
    print('t5 - t4', t5-t4)
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    # try:
    #     res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                callback=callback0)
    # # res stands for results (not residuals!).
    # except Exception as e:
    #     print(e)

    PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
    if not os.path.exists(os.path.join(PATH, str(date.today()), args.sem)):
        os.makedirs(os.path.join(PATH, str(date.today()), args.sem), exist_ok=True)

    # assert opt_params is not None
    # params = opt_params
    al = get_median_inter_mnist(X)
    print('bandwidth = ', al)
    # al = 8.4787
    # bl_dict = [0.2, 2., 3., 10., 0.4, 0.6, 0.8, 1.]
    bl_dict = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    params = [al, bl_dict[seed]]

    plot_h(params=params, inp_sample=X)

    do_A = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}.npz'.format(args.sem))['do_A']
    EY_do_A_gt = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}.npz'.format(args.sem))['gt_EY_do_A']
    w_sample = train.w
    EYhat_do_A = get_causal_effect(params=params, do_A=do_A, w=w_sample)
    plt.figure()
    plt.plot([i + 1 for i in range(20)], EYhat_do_A, label='est')
    plt.plot([i + 1 for i in range(20)], EY_do_A_gt, label='gt')
    plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    plt.savefig(
        os.path.join(PATH, str(date.today()), args.sem, 'causal_effect_estimates_nystr_trainsz{}_seed{}'.format(AW_train.shape[0], seed) + '.png'))
    plt.close()
    print('ground truth ate: ', EY_do_A_gt)
    visualise_ATEs(EY_do_A_gt, EYhat_do_A,
                   x_name='E[Y|do(A)] - gt',
                   y_name='beta_A',
                   save_loc=os.path.join(PATH, str(date.today()), args.sem),
                   save_name='ate_trainsz{}_seed{}_nystr'.format(AW_train.shape[0], seed))
    causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt - EYhat_do_A))
    causal_effect_mae_file = open(os.path.join(PATH, str(date.today()), args.sem, "ate_mae_trainsz{}_nystrom.txt".format(AW_train.shape[0])),
                                  "a")
    causal_effect_mae_file.write("seed: {}, mae_: {}\n".format(seed, causal_effect_mean_abs_err))
    causal_effect_mae_file.close()

    os.makedirs(PATH, exist_ok=True)
    np.save(os.path.join(PATH, str(date.today()), args.sem, 'LMO_errs_seed{}_nystr_trainsz{}.npy'.format(seed, AW_train.shape[0])), [opt_params, prev_norm, opt_test_err])

    return causal_effect_mean_abs_err

    # TODO: where is alpha? and how is it making a prediction? alpha is defined in the callback function. how is it reached?


def summarize_res(sname, datasize):
    print(sname)
    res = []
    times = []
    for i in range(100):
        PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
        filename = os.path.join(PATH, str(date.today()), args.sem, 'LMO_errs_seed{}_nystr_trainsz{}.npy'.format(i, datasize))
        if os.path.exists(filename):
            tmp_res = np.load(filename, allow_pickle=True)
            if tmp_res[-1] is not None:
                res += [tmp_res[-1]]
        time_path = os.path.join(PATH, str(date.today()), args.sem, '/LMO_errs_seed{}_nystr_trainsz{}_time.npy'.format(i, datasize))
        if os.path.exists(time_path):
            t = np.load(time_path)
            times += [t]
    res = np.array(res)
    times = np.array(times)
    res = remove_outliers(res)
    times = np.sort(times)[:80]
    print(times)
    print('mean, std: ', np.mean(res), np.std(res))
    print('time: ', np.mean(times), np.std(times))



if __name__ == '__main__':
    # np.random.seed(6)
    # random.seed(6)
    # snames = ['step','sin','abs','linear']
    snames = ["sim_1d_no_x"]
    causal_effect_mean_abs_err_best, best_seed = None, None
    for datasize in [5000]:
        for sname in snames:
            for seed in range(100):
                np.random.seed(seed)
                random.seed(seed)
                causal_effect_mean_abs_err = experiment(sname, seed, datasize, False if datasize < 1000 else True, args=args)
                if causal_effect_mean_abs_err_best is None or causal_effect_mean_abs_err < causal_effect_mean_abs_err_best:
                    causal_effect_mean_abs_err_best, best_seed = causal_effect_mean_abs_err, seed

            print('best mae for {} found at seed: {}'.format(sname, best_seed))
            PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
            with open(os.path.join(PATH, str(date.today()), args.sem, 'best_seed.txt'), 'w') as f:
                f.write('%d' % best_seed)
            # summarize_res(sname, datasize)

    
    
    
    
    
    
    
    
    

