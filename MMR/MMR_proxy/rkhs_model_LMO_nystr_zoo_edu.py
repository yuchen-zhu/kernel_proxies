import os, sys
# import autograd.numpy as np
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from MMR_proxy.util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv, bundle_az_aw, visualise_ATEs, data_transform, data_inv_transform, indicator_kern
from joblib import Parallel, delayed
import time
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import date
import argparse
import random

Nfeval = 1
seed = 527

# np.random.seed(seed)
# np.random.seed(seed)
JITTER = 1e-10
nystr_M = 1000 # was 2000
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None
# a_diffmax, a_diffmin = (0.5, -0.5)
# b_max, b_min = (10, 1)
a_diffmax, a_diffmin, a_mesh_size = (0., -0., 1)
bn_max, bn_min, b_mesh_size = (450, 2, 30)
# train_size, test_size, dev_size = 6000, 1000, 1000
test_size, dev_size = 1000, 1000

parser = argparse.ArgumentParser(description='run rhks model with general gaussian-based kernel')

parser.add_argument('--av-kernel', type=bool, default=False, help='use single bandwidth or average')
parser.add_argument('--sem', type=str, help='set which SEM to use data from')
parser.add_argument('--hparam', type=str, help='set which hparam method to use. options are cube or lmo')
parser.add_argument('--al', type=float, default=None, help='set the parameter for rkhs, al.')
parser.add_argument('--bl', type=float, default=None, help='set the parameter for rkhs, bl.')
# parser.add_argument('--train-size', type=int, default=train_size, help='set training size.')
args = parser.parse_args()
print('parsed av-kernel: ', args.av_kernel)


def scale_all(train_A, train_Y, train_Z, train_W, test_A, test_Y, test_Z, test_W):
    A_scaled, A_scaler = data_transform(train_A)
    Y_scaled, Y_scaler = data_transform(train_Y)
    Z_scaled, Z_scaler = data_transform(train_Z)
    W_scaled, W_scaler = data_transform(train_W)

    test_A_scaled = A_scaler.transform(test_A)
    test_Y_scaled = Y_scaler.transform(test_Y)
    test_Z_scaled = Z_scaler.transform(test_Z)
    test_W_scaled = W_scaler.transform(test_W)

    return A_scaled, Y_scaled, Z_scaled, W_scaled, test_A_scaled, test_Y_scaled, test_Z_scaled, test_W_scaled, A_scaler, Y_scaler, Z_scaler, W_scaler



def rank_nums_in_array(arr):
    '''
    helper to rank the numbers in an array. e.g. input = np.arr([2,1,3]), output = np.arr([1,0,2]).
    '''
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))

    return ranks


def hparam_selection_from_metric_votes(mmr_v, mmr_v_supp, lmo, hparam_arr):
    ranks_mmr_v = rank_nums_in_array(mmr_v.squeeze())
    # ranks_mmr_v_supp = rank_nums_in_array(mmr_v_supp.squeeze())
    # ranks_lmo = rank_nums_in_array(lmo.squeeze())
    # sum_ranks = ranks_mmr_v + ranks_mmr_v_supp + ranks_lmo
    sum_ranks = ranks_mmr_v
    idx_argmin = sum_ranks.argmin()
    return hparam_arr[idx_argmin]


def get_best_hparam(results_dict):
    hparam_arr = np.array(results_dict['causal_mae'])[:, :2]
    mmr_vs = np.array(results_dict['mmr_v'])[:, -1]
    mmr_v_supps = None
    lmos = np.array(results_dict['lmo'])[:, -1]
    return hparam_selection_from_metric_votes(mmr_v=mmr_vs, mmr_v_supp=mmr_v_supps, lmo=lmos, hparam_arr=hparam_arr)


def cube_search(a_diffmin, a_diffmax, a_mesh_size, b_min, b_max, b_mesh_size, args, PATH, train_size):
    al_arr, bl_arr = np.linspace(a_diffmin, a_diffmax, a_mesh_size), np.linspace(b_min, b_max, b_mesh_size)
    results_dict = {'causal_mae': [],
                    'mse_standard': [],
                    'mmr_v': [],
                    'mmr_u': [],
                    'lmo': [],
                    'ate_est': {}}
    for al in al_arr:
        for bl in bl_arr:
            params = [al, bl]
            print('fitting al = {}, bl = {}'.format(al, bl))
            causal_effect_mean_abs_err, al_median_dist, causal_effect_est, mse_standard, mmr_v, mmr_u, lmo = \
                experiment(sname=sname, seed=seed, param_arg=params,
                           train_size=train_size,
                           nystr=(False if train_size <= 500 else True),
                           args=args, PATH=PATH, hparam=args.hparam)
            # if err_in_expectation_best is None or err_in_expectation < err_in_expectation_best:
            #     err_in_expectation_best, best_params, causal_effect_mean_abs_err_best = err_in_expectation, params, causal_effect_mean_abs_err
            # if causal_effect_mean_abs_err_best is None or causal_effect_mean_abs_err < causal_effect_mean_abs_err_best:
            #     causal_effect_mean_abs_err_best, best_params = causal_effect_mean_abs_err, params
            if 'al_median_dist' not in results_dict:
                results_dict['al_median_dist'] = al_median_dist
            results_dict['causal_mae'].append([al_median_dist + al, bl, causal_effect_mean_abs_err])
            results_dict['mse_standard'].append([al_median_dist + al, bl, mse_standard])
            results_dict['mmr_v'].append([al_median_dist + al, bl, mmr_v])
            results_dict['mmr_u'].append([al_median_dist + al, bl, mmr_u])
            results_dict['lmo'].append([al_median_dist + al, bl, lmo])
            results_dict['ate_est'][(al_median_dist + al, bl)] = causal_effect_est
    do_hparam_analysis_plots(PATH=PATH, args=args, train_size=train_size, **results_dict)
    best_hparams = get_best_hparam(results_dict)
    best_ate_est = results_dict['ate_est'][(best_hparams[0], best_hparams[1])]

    print('best mae for {} found at params: {} using hparam search method: {}'.format(sname, best_hparams,
                                                                                      args.hparam))
    with open(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'best_params_cube.txt'), 'w') as f:
        f.write('best al: {:.3f}, best bl: {:.3f}'.format(best_hparams[0] + al_median_dist, best_hparams[1]))

    return best_hparams, best_ate_est


def hyparameter_selection(a_diffmin, a_diffmax, a_mesh_size, b_min, b_max, b_mesh_size, args, PATH, train_size):
    if args.hparam == 'cube':
        best_hparams, best_ate_est = cube_search(a_diffmin=a_diffmin, a_diffmax=a_diffmax, a_mesh_size=a_mesh_size,
                                                   b_min=b_min, b_max=b_max, b_mesh_size=b_mesh_size,
                                                   args=args, PATH=PATH, train_size=train_size)
        return best_hparams, best_ate_est

    elif args.hparam == 'lmo':
        # al_arr = np.linspace(a_diffmin, a_diffmax, 10)
        # for al in al_arr:
        #     params = [al]
        #     print('fitting al = {}'.format(al))
        causal_effect_mean_abs_err, err_in_expectation, al_median_dist, causal_effect_est, mse_standard, mmr_v, mmr_u, lmo = experiment(sname=sname, seed=seed, param_arg=[],
                                                                         train_size=train_size,
                                                                         nystr=(False if train_size <= 500 else True),
                                                                         args=args, PATH=PATH, hparam=args.hparam)
        # if err_in_expectation_best is None or err_in_expectation < err_in_expectation_best:
        #     err_in_expectation_best, best_params, causal_effect_mean_abs_err_best = err_in_expectation, params, causal_effect_mean_abs_err
        # if causal_effect_mean_abs_err_best is None or causal_effect_mean_abs_err < causal_effect_mean_abs_err_best:
        #     causal_effect_mean_abs_err_best, best_params = causal_effect_mean_abs_err, params
        return None, causal_effect_est
    else:
        raise NotImplementedError


def evaluate_ate_est(ate_est, ate_gt):
    causal_mae = np.mean(np.abs(ate_gt.squeeze() - ate_est.squeeze()))
    causal_std = np.std(np.abs(ate_est.squeeze() - ate_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((ate_est.squeeze() - ate_gt.squeeze())/ate_gt.squeeze()))

    return causal_mae, causal_std, causal_rel_err


def get_results(EYhat_do_A, EY_do_A_gt, train_sz, mse_standard, mmr_v, mmr_u, lmo, params, args, PATH):
    plt.figure()
    plt.plot([i + 1 for i in range(len(EY_do_A_gt))], EYhat_do_A, label='est')
    plt.plot([i + 1 for i in range(len(EY_do_A_gt))], EY_do_A_gt, label='gt')
    plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    plt.savefig(
        os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                     'causal_effect_estimates_nystr_trainsz{}_al{}_bl{}.png'.format(train_sz, params[0], params[1])))
    plt.close()
    print('ground truth ate: ', EY_do_A_gt)
    visualise_ATEs(EY_do_A_gt, EYhat_do_A,
                   x_name='E[Y|do(A)] - gt',
                   y_name='beta_A',
                   save_loc=os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed)),
                   save_name='ate_trainsz{}_al{}_bl{}_nystr'.format(train_sz, params[0], params[1]))

    causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt.squeeze() - EYhat_do_A.squeeze()))

    causal_std = np.std(np.abs(EYhat_do_A.squeeze() - EY_do_A_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((EYhat_do_A.squeeze() - EY_do_A_gt.squeeze())/EY_do_A_gt.squeeze()))

    summary_file = open(
        os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), "summary_trainsz{}_nystrom_hparam{}.txt".format(train_sz, args.hparam)),
        "a")
    summary_file.write("al: {}, bl: {}, causal_mae_: {}, causal_std: {}, causal_rel_err: {}\n"
                       "mmr_v: {},  mmr_u: {}\n"
                       "mse: {}, lmo: {}\n"
                       "causal_est: {}\n".format(params[0], params[1], causal_effect_mean_abs_err, causal_std,
                                                 causal_rel_err, mmr_v, mmr_u, mse_standard, lmo, EYhat_do_A))
    summary_file.close()

    os.makedirs(PATH, exist_ok=True)
    np.save(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                         'LMO_errs_params{}_nystr_trainsz{}.npy'.format(params, train_sz)),
            [opt_params, prev_norm, opt_test_err])

    return causal_effect_mean_abs_err


def experiment(sname, seed, param_arg, train_size, nystr=False, args=None, PATH=None, hparam=None):
    np.random.seed(seed)
    random.seed(seed)
    # def LMO_err(params, M=10):
    def LMO_err(params, M=10):
        # np.random.seed(2)
        # random.seed(2)
        log_al, log_bl = params
        al, bl = np.exp(log_al).squeeze(), np.exp(log_bl).squeeze()
        # print('lmo_err params', params)
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
            # permutation = np.random.permutation(X.shape[0])
            idxs = np.arange(X.shape[0])
            for i in range(0, X.shape[0], M):
                indices = idxs[i:i + M]
                K_i = W[np.ix_(indices, indices)] * N2
                C_i = C[np.ix_(indices, indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                lmo_inc = b_y.T @ K_i @ b_y
                lmo_err = lmo_err + lmo_inc
                # print('lmo_inc: ', lmo_inc)
                N += 1
        return lmo_err[0, 0] / N / M ** 2

    # def callback0(params, timer=None):
    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        # np.random.seed(3)
        # random.seed(3)
        if Nfeval % 1 == 0:
            log_al, log_bl = params
            al, bl = np.exp(log_al).squeeze(), np.exp(log_bl).squeeze()
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
            if norm[0, 0] / prev_norm >= 3: # early stopping
                if opt_params is None:
                    opt_test_err = test_err
                    opt_params = (al, bl)
                print(True, opt_params, opt_test_err, prev_norm)
                raise Exception

        if prev_norm is None or norm[0, 0] <= prev_norm:
            prev_norm = norm[0, 0]
        opt_test_err = test_err
        opt_params = (al, bl)
        print('params,test_err, norm: ', opt_params, opt_test_err, norm[0, 0])

    def get_causal_effect(params, do_A, w, Y_scaler):
        "to be called within experiment function."
        # np.random.seed(4)
        # random.seed(4)
        # al, bl = np.exp(params)
        al, bl = params
        print('al, bl = ', params)
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        if nystr:
            alpha = EYEN - eig_vec_K @ np.linalg.inv(
                eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
            alpha = alpha @ W_nystr @ Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            alpha = LWL_inv @ L @ W @ Y
            # L_W_inv = chol_inv(W*N2+L_inv)

        assert w.ndim == 2
        assert do_A.ndim == 2
        print('W_shape: ', w.shape[0])
        do_A_rep = np.repeat(do_A, [w.shape[0]], axis=-1).reshape(-1, 1)
        w_rep = np.tile(w, [do_A.shape[0], 1])
        aw_rep = np.concatenate([do_A_rep, w_rep], axis=-1)
        ate_L0 = _sqdist(aw_rep, X)
        ate_L = bl * bl * np.exp(-ate_L0/ al/ al/ 2)
        h_out = ate_L @ alpha

        h_out_a_as_rows = h_out.reshape(-1, w.shape[0])
        ate_est = np.mean(h_out_a_as_rows, axis=-1).reshape(-1,1)
        ate_est_orig_scale = Y_scaler.inverse_transform(ate_est)

        print('ate ESTIMATES: ', ate_est_orig_scale.squeeze())
        #
        # EYhat_do_A = []
        # for a in do_A:
        #     a = np.repeat(a, [w.shape[0]]).reshape(-1, 1)
        #     w = w.reshape(-1, 1)
        #     aw = np.concatenate([a, w], axis=-1)
        #     ate_L0 = _sqdist(aw, X)
        #     ate_L = bl * bl * np.exp(-ate_L0 / al / al / 2)
        #     h_out = ate_L @ alpha
        #
        #     mean_h = np.mean(h_out).reshape(-1, 1)
        #     EYhat_do_A.append(mean_h)
        #     print('a = {}, beta_a = {}'.format(np.mean(a), mean_h))

        return ate_est_orig_scale.squeeze()


    def compute_losses(params, AW_test, AZ_test, Y_test):
        "to calculated the expected error E_{A,X,Z ~ unif}[E[Y - h(A,X,W)|A,X,Z]]."
        al, bl = params
        print('evaluating conditional expected loss. al, bl = params')

        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        if nystr:
            alpha = EYEN - eig_vec_K @ np.linalg.inv(
                eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
            alpha = alpha @ W_nystr @ Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            alpha = LWL_inv @ L @ W @ Y
            # L_W_inv = chol_inv(W*N2+L_inv)

        # standard mse
        mse_L0 = _sqdist(AW_test, X)
        mse_L = bl * bl * np.exp(-mse_L0 / al / al / 2)
        mse_h = mse_L @ alpha
        mse_standard = np.mean((test_Y.flatten() - mse_h.flatten()) ** 2)

        mmr_v, mmr_u = mmr_loss(al=al, bl=bl, alpha=alpha, y_test=Y_test, aw_test=AW_test, az_test=AZ_test, ak=ak)

        # lmo
        lmo_err = LMO_err(params=np.log(params), M=1)
        return mse_standard, mmr_v, mmr_u, lmo_err
        # mean abs error is E_{A,X,Z~uniform}[E[|Y-h||A,X,Z]], mse is E_{A,X,Z~uniform}[E[(y-h)^2|A,X,Z]], y_axz_recon = E[h|A,X,Z] for the uniformly sampled (a,x,z)'s.


    def compute_loss_on_supported_test_set(X, al, bl, alpha, supp_y, supp_aw):
        mse_L0 = _sqdist(supp_aw, X)
        mse_L = bl * bl * np.exp(-mse_L0 / al / al / 2)
        mse_h = mse_L @ alpha
        mse_supp = np.mean((supp_y.flatten() - mse_h.flatten()) ** 2)

        return mse_supp


    def mmr_loss(ak, al, bl, alpha, y_test, aw_test, az_test):
        mse_L0 = _sqdist(aw_test, X)
        mse_L = bl * bl * np.exp(-mse_L0 / al / al / 2)
        mse_h = mse_L @ alpha

        EYEN = np.eye(y_test.shape[0])
        # ak0, ak1 = get_median_inter_mnist(Z[:, 0:1]), get_median_inter_mnist(Z[:, 1:2])
        print('supp_az shape: ', az_test.shape)
        # ak = get_median_inter_mnist(Z)
        N2 = y_test.shape[0] ** 2
        N = y_test.shape[0]
        W0 = _sqdist(az_test, None)
        print('av kernel indicator: ', args.av_kernel)
        # W = np.exp(-W0 / ak0 / ak0 / 2) / N2 if not args.av_kernel \
        #     else (np.exp(-W0 / ak0 / ak0 / 2) + np.exp(-W0 / ak0 / ak0 / 200) + np.exp(-W0 / ak0 / ak0 * 50)) / 3 / N2
        K = np.exp(-W0 / ak / ak / 2)  if not args.av_kernel \
            else (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3

        # W_U = (K - np.diag(K))/N/(N-1)
        # W_V = (K - np.diag(K))/N2

        W_U = (K - np.diag(K))
        W_V = K

        assert y_test.ndim > 1
        assert mse_h.ndim == y_test.ndim
        for dim in range(mse_h.ndim):
            assert mse_h.shape[dim] == y_test.shape[dim]

        d = mse_h - y_test
        # if indices is None:
        #     W = K
        # else:
        #     W = K[indices[:, None], indices]
            # print((kernel(Z[indices],None,a,1)+kernel(Z[indices],None,a/10,1)+kernel(Z[indices],None,a*10,1))/3-W)
        # loss_V = d.T @ W_V @ d / (d.shape[0]) ** 2
        # loss_U = d.T @ W_U @ d / (d.shape[0]) ** 2

        loss_V = d.T @ W_V @ d / N / N
        loss_U = d.T @ W_U @ d / N / (N - 1)
        return loss_V[0, 0], loss_U[0, 0]

    def plot_h(params, inp_sample, PATH):
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
            print('chol_inv isnt working')
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
        plt.savefig(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'h_trainsz{}_seed{}'.format(AW_train.shape[0], seed) + '.png'))
        plt.close()

    # train,dev,test = load_data(ROOT_PATH+'/data/zoo/{}_{}.npz'.format(sname,train_size))

    # X = np.vstack((train.x,dev.x))
    # Y = np.vstack((train.y,dev.y))
    # Z = np.vstack((train.z,dev.z))
    # test_X = test.x
    # test_Y = test.g
    t1 = time.time()
    train, dev, test = load_data(ROOT_PATH + "/data/zoo/" + sname + '/main_{}_seed{}.npz'.format(args.sem, data_seed))
    # train, dev, test = train[:300], dev[:100], test[:100]
    t2 = time.time()
    print('t2 - t1 = ', t2 - t1)

    A_scaled, Y_scaled, Z_scaled, W_scaled, test_A_scaled, test_Y_scaled, test_Z_scaled, test_W_scaled, A_scaler, Y_scaler, Z_scaler, W_scaler = \
    scale_all(train_A=train.a[:train_size], train_Y=train.y[:train_size], train_Z=train.z[:train_size], train_W=train.w[:train_size],
              test_A=test.a[:test_size], test_Y=test.y[:test_size], test_Z=test.z[:test_size], test_W=test.w[:test_size])


    # Y = np.concatenate((train.y[:train_size], dev.y[:dev_size]), axis=0).reshape(-1, 1)
    Y = Y_scaled.reshape(-1,1)
    # test_Y = test.y
    AZ_train, AW_train = bundle_az_aw(A_scaled, Z_scaled, W_scaled)
    AZ_test, AW_test = bundle_az_aw(test_A_scaled, test_Z_scaled, test_W_scaled)
    # AZ_dev, AW_dev = bundle_az_aw(dev.a[:dev_size], dev.z[:dev_size], dev.w[:dev_size])

    X, Z = AW_train, AZ_train
    test_X, test_Y = AW_test, test_Y_scaled  # TODO: is test.g just test.y?
    train_sz, test_sz = X.shape[0], test_X.shape[0]

    # # load expectation eval data
    # axzy = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['axzy']
    # w_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['w_samples']
    # y_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['y_samples']
    # y_axz = axzy[:, -1]
    # ax = axzy[:, :2]

    # # load supported test eval data
    # test_supp = np.load(ROOT_PATH + "/data/zoo/" + sname + '/supported_test_metric_{}.npz'.format(args.sem))
    # aw_test_supp = test_supp['test_aw']
    # az_test_supp = test_supp['test_az']
    # y_test_supp = test_supp['test_y']

    t3 = time.time()
    print('t3 - t2', t3-t2)
    EYEN = np.eye(X.shape[0])
    # ak0, ak1 = get_median_inter_mnist(Z[:, 0:1]), get_median_inter_mnist(Z[:, 1:2])
    print('Z shape: ', Z.shape)
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
    if args.hparam == 'lmo':
        log_al0, log_bl0 = np.log(get_median_inter_mnist(X)), np.random.randn(1) + 5
        # params0 = np.random.randn(2)  # /10
        params0 = np.array([log_al0, log_bl0])
        print('starting param log_al0: {}, log_bl0: {}'.format(log_al0, log_bl0))
        bounds = None  # [[0.01,10],[0.01,5]]

        obj_grad = value_and_grad(lambda params: LMO_err(params))
        # try:
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
                   callback=callback0)
        # res stands for results (not residuals!).
        # except Exception as e:
        #     print(e)

    # PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
    # if not os.path.exists(os.path.join(PATH, str(date.today()), args.sem)):
    #     os.makedirs(os.path.join(PATH, str(date.today()), args.sem), exist_ok=True)

        assert opt_params is not None
        params = opt_params
    elif args.hparam == 'cube':
        al, bl = get_median_inter_mnist(X) + param_arg[0], param_arg[1]
        print('bandwidth = ', al)
        # al = 8.4787
        # bl_dict = [0.2, 2., 3., 10., 0.4, 0.6, 0.8, 1.]
        # bl_dict = [50, 55, 60, 65, 70, 75, 80, 85, 90]
        params = [al, bl]

    elif args.hparam == 'fixed':
        al, bl = param_arg[0], param_arg[1]
        params=[al, bl]

    else:
        raise NotImplementedError

    # plot_h(params=params, inp_sample=X, PATH=PATH)

    do_A = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['do_A']
    # print('do A: ', do_A)
    # do_A = np.array([0,1,2])
    do_A = A_scaler.transform(do_A)
    EY_do_A_gt = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['gt_EY_do_A']
    # EY_do_A_gt = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['gt_EY_do_A_cat']

    # print('do_A: ', do_A)
    # print('EY_do_A_gt: ', EY_do_A_gt)

    w_sample = W_scaled[:50]
    EYhat_do_A = get_causal_effect(params=params, do_A=do_A, w=w_sample, Y_scaler=Y_scaler)

    mse_standard, mmr_v, mmr_u, lmo = compute_losses(params=params, AW_test=AW_test, AZ_test=AZ_test, Y_test=test_Y)

    causal_effect_mean_abs_err = get_results(EYhat_do_A=EYhat_do_A, EY_do_A_gt=EY_do_A_gt, train_sz=train_sz,
                                             mse_standard=mse_standard, mmr_v = mmr_v, mmr_u=mmr_u, lmo=lmo,
                                             params=params, args=args, PATH=PATH)

    # plt.figure()
    # plt.plot([i + 1 for i in range(20)], EYhat_do_A, label='est')
    # plt.plot([i + 1 for i in range(20)], EY_do_A_gt, label='gt')
    # plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    # plt.savefig(
    #     os.path.join(PATH, str(date.today()), args.sem, 'causal_effect_estimates_nystr_trainsz{}_seed{}'.format(AW_train.shape[0], seed) + '.png'))
    # plt.close()
    # print('ground truth ate: ', EY_do_A_gt)
    # visualise_ATEs(EY_do_A_gt, EYhat_do_A,
    #                x_name='E[Y|do(A)] - gt',
    #                y_name='beta_A',
    #                save_loc=os.path.join(PATH, str(date.today()), args.sem),
    #                save_name='ate_trainsz{}_seed{}_nystr'.format(AW_train.shape[0], seed))
    # causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt - EYhat_do_A))
    # causal_effect_mae_file = open(os.path.join(PATH, str(date.today()), args.sem, "ate_mae_trainsz{}_nystrom.txt".format(AW_train.shape[0])),
    #                               "a")
    # causal_effect_mae_file.write("seed: {}, mae_: {}\n".format(seed, causal_effect_mean_abs_err))
    # causal_effect_mae_file.close()
    #
    # os.makedirs(PATH, exist_ok=True)
    # np.save(os.path.join(PATH, str(date.today()), args.sem, 'LMO_errs_seed{}_nystr_trainsz{}.npy'.format(seed, AW_train.shape[0])), [opt_params, prev_norm, opt_test_err])
    if args.hparam == 'lmo':
        return causal_effect_mean_abs_err, None, EYhat_do_A, mse_standard, mmr_v, mmr_u, lmo
    else:
        return causal_effect_mean_abs_err, al - param_arg[0], EYhat_do_A, mse_standard, mmr_v, mmr_u, lmo

    # TODO: where is alpha? and how is it making a prediction? alpha is defined in the callback function. how is it reached?


def do_hparam_analysis_plots(PATH, args, train_size, **h_param_results_dict):
    print('plotting')
    os.makedirs(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed)), exist_ok=True)
    causal_mae = np.array(h_param_results_dict['causal_mae'])[:, -1]
    mean_causal_mae = np.mean(causal_mae)
    causal_mae_rescaled = 1/mean_causal_mae * causal_mae
    print('keys: ', h_param_results_dict.keys())
    for var_str in h_param_results_dict.keys():
        print(var_str)
        print((var_str == 'al_median_dist') or (var_str=='causal_mae') or (var_str=='ate_est'))
        boolean = (var_str == 'al_median_dist') or (var_str=='causal_mae') or (var_str=='ate_est')
        if boolean:
            continue
        var = np.array(h_param_results_dict[var_str])[:, -1]
        mean_var = np.mean(var)
        var_rescaled = 1/mean_var * var
        length = var.shape[0]
        plt.figure()
        plt.plot(np.arange(length), var_rescaled, label=var_str)
        plt.plot(np.arange(length), causal_mae_rescaled, label='causal_mae')
        plt.xlabel('Hyperparameter labels'), plt.ylabel('causal_MAE/{}-rescaled'.format(var_str)), plt.legend()
        print('save path: ', os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}.png'.format(var_str, train_size)))
        plt.savefig(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}.png'.format(var_str, train_size)))
        plt.close()


def summarize_res(sname, train_size):
    print(sname)
    res = []
    times = []
    for i in range(100):
        PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
        filename = os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'LMO_errs_seed{}_nystr_trainsz{}.npy'.format(i, train_size))
        if os.path.exists(filename):
            tmp_res = np.load(filename, allow_pickle=True)
            if tmp_res[-1] is not None:
                res += [tmp_res[-1]]
        time_path = os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), '/LMO_errs_seed{}_nystr_trainsz{}_time.npy'.format(i, train_size))
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


def run_rkhs(args, train_size, sname):
    # np.random.seed(6)
    # random.seed(6)
    # snames = ['step','sin','abs','linear']
    # snames = ["sim_1d_no_x"]
    # for train_size in [5000]:
    #     for sname in snames:

    PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
    if not os.path.exists(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed))):
        os.makedirs(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed)), exist_ok=True)

    summary_file = open(
        os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                     "summary_trainsz{}_nystrom_hparam{}.txt".format(int(train_size), args.hparam)), "w")
    summary_file.close()
    # err_in_expectation_best, best_params, causal_effect_mean_abs_err_best, al_median_dist = None, None, None, None
    causal_mae, _, causal_effect_estimates, mse_standard, mmr_v, mmr_u, lmo = experiment(sname=sname, seed=seed,
                                                  param_arg=[args.al, args.bl],
                                                  train_size=train_size,
                                                  nystr=(False if train_size <= 500 else True),
                                                  args=args, PATH=PATH, hparam=args.hparam)

    return causal_mae, causal_effect_estimates


if __name__ == '__main__':
    snames = ["sim_1d_no_x"]
    for train_size in [1500]:
        b_max, b_min = bn_max / train_size, bn_min / train_size
        for data_seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            for sname in snames:
                PATH = ROOT_PATH + "/MMR_proxy/results/zoo/" + sname + "/"
                if not os.path.exists(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed))):
                    os.makedirs(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed)), exist_ok=True)

                summary_file = open(
                    os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                                 "summary_trainsz{}_nystrom_hparam{}.txt".format(int(train_size), args.hparam)), "w")
                summary_file.close()

                do_A = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['do_A']
                EY_do_A_gt = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['gt_EY_do_A']

                best_hparams, best_ate_est = hyparameter_selection(a_diffmax=a_diffmax, a_diffmin=a_diffmin, a_mesh_size=a_mesh_size, b_min=b_min, b_max=b_max, b_mesh_size=b_mesh_size, args=args, PATH=PATH, train_size=train_size)
                best_causal_mae, best_causal_std, best_causal_rel_err = evaluate_ate_est(ate_est=best_ate_est, ate_gt=EY_do_A_gt)
                np.savez(os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'mmr_res_trainsz{}.npz'.format(train_size)), do_A=do_A, ate_est=best_ate_est, lambda_dict={'al': best_hparams[0], 'bl': best_hparams[1]}, train_sz=train_size,
                         causal_mae=best_causal_mae, causal_std=best_causal_std, causal_rel_err=best_causal_rel_err)

                # summarize_res(sname, train_size)





    
    
    
    
    
    
    
    
    
    
    

