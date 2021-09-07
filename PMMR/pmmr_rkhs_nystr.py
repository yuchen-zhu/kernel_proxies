import os
import numpy as np
import plotly.graph_objects as go
from PMMR.util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp_from_orig, nystrom_decomp_from_sub, chol_inv, bundle_az_aw, visualise_ATEs, data_transform, data_inv_transform, indicator_kern
from datetime import date
import argparse
import random
import time
import autograd.numpy as anp
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Global parameters
Nfeval = 1
JITTER_W, JITTER_L, JITTER_LW = 1e-12, 1e-5, 1e-7
sname = "sim_1d_no_x"
test_sz, dev_sz = 1000, 1000
w_marginal_size = 50
nystr_M = 3000
log_al_bounds, log_bl_bounds = np.array([-0.5, 0.5]), np.array([-5, 1])
nystr_thresh = 5000
seed = 527

al_diff_search_range = [-0., -0., 1]
bl_search_range = [3, 25., 300]
train_sizes = [200, 500, 1000]
# data_seeds = np.arange(100, 501, 100)
data_seeds = np.array([1009, 1102, 1656, 1816, 2029,
                       2297, 2533, 2807, 4259, 4379,
                       4388, 4987, 5518, 5654, 5949,
                       7422, 7987, 8455, 9783, 9886])

# data_seeds = np.array([5654, 5949, 7422, 7987, 8455, 9783, 9886])
# data_seeds = np.array([1009, 1102, 1656, 1816, 2029])

selection_metric = 'mmr_v'  # mmr_v or mmr_v_supp or lmo

parser = argparse.ArgumentParser(description='run rkhs model with universal (gaussian) kernel')

parser.add_argument('--sem', type=str, help='set which SEM to use data from.')
parser.add_argument('--hparam', type=str, help='set which hparam method to use. Options are cube or lmo or fixed.')
parser.add_argument('--selection-metric', type=str, default=selection_metric, help='set which metric to use to select hparams.')
parser.add_argument('--cond-metric', type=bool, default=False, help='on/off conditional expectation metric.')
parser.add_argument('--supp-test', type=bool, default=False, help='on/off supported test set.')
parser.add_argument('--scenario-name', type=str, default=sname, help='set scenario name')
parser.add_argument('--JITTER-L', type=float, default=JITTER_L, help='set jitter value to stabilise kernel inversion for L.')
parser.add_argument('--JITTER-W', type=float, default=JITTER_W, help='set jitter value to stabilise kernel inversion for W.')
parser.add_argument('--nystr-M', type=str, default=nystr_M, help='Set subsample size M for nystrom approximation.')
parser.add_argument('--log-al-bounds', type=tuple, default=log_al_bounds, help='Set log al bounds in lmo')
parser.add_argument('--log-bl-bounds', type=tuple, default=log_bl_bounds, help='Set log bl bounds in lmo')
parser.add_argument('--w-marginal-size', type=int, default=w_marginal_size, help='Set W marginal sample size.')
parser.add_argument('--offset', type=bool, default=False, help='Turn on/off offset calculation.')
parser.add_argument('--lmo', type=str, default=None, help='set which parameter to do lmo search over.')
args = parser.parse_args()


def compute_bandwidth_median_dist(X):
    al_default = []
    for dim in range(X.shape[-1]):
        al_default.append(get_median_inter_mnist(X[:, dim:dim + 1]))
    al_default = np.array(al_default)
    print('al default: ', al_default)
    return al_default


def make_gaussian_prodkern(arr1, arr2, sigma):
    dims = arr1.shape[-1]
    assert arr1.shape[-1] == arr2.shape[-1]

    K = 1
    for dim in range(dims):
        K_0 = _sqdist(arr1[:, dim].reshape(-1, 1), arr2[:, dim].reshape(-1, 1))
        sig = sigma[dim]
        if (type(sig) is not np.float64) and (type(sig) is not np.float32):
            from math import e
            print('K dim {}: '.format(dim), np.linalg.cond(e ** (-K_0 / sig._value / sig._value / 2)))
        else:
            print('K dim {}: '.format(dim), np.linalg.cond(np.exp(-K_0 / sig / sig / 2)))
        K = K * anp.exp(-K_0 / sig / sig / 2)
        del K_0
    return K


def process_data(train_size, dev_size, test_size, args, data_seed, LOAD_PATH):
    t1 = time.time()

    # loads all data
    train, dev, test = load_data(os.path.join(LOAD_PATH, 'main_{}_seed{}.npz'.format(args.sem, data_seed)))

    train_Y = train.y[:train_size].reshape(-1, 1) # Y always has only 1 feature
    AZ_train, AW_train = bundle_az_aw(train.a[:train_size], train.z[:train_size], train.w[:train_size])
    AZ_test, AW_test = bundle_az_aw(test.a[:test_size], test.z[:test_size], test.w[:test_size])
    # AZ_dev, AW_dev = bundle_az_aw(dev.a[:dev_size], dev.z[:dev_size], dev.w[:dev_size])

    train_X, train_Z = AW_train, AZ_train
    test_X, test_Y, test_Z = AW_test, test.y.reshape(-1,1)[:test_size], AZ_test

    W_marginal = train.w[:args.w_marginal_size].reshape(args.w_marginal_size, -1)
    w_dim = W_marginal.shape[-1]
    do_A = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))['do_A'].reshape(-1,1)
    EY_do_A_gt = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))['gt_EY_do_A'].reshape(-1,1)


    t2 = time.time()
    print('data loading used {}s'.format(t2 - t1))
    return train_X, train_Y, train_Z, test_X, test_Y, test_Z, W_marginal, do_A, EY_do_A_gt, w_dim


def load_err_in_expectation_metric(args, data_seed, LOAD_PATH):
    t1 = time.time()
    do_A = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))['do_A']
    EY_do_A_gt = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))['gt_EY_do_A']

    axzy = np.load(os.path.join(LOAD_PATH, 'cond_exp_metric_{}_seed{}.npz'.format(args.sem, data_seed)))['axzy']
    w_samples = np.load(os.path.join(LOAD_PATH, 'cond_exp_metric_{}_seed{}.npz'.format(args.sem, data_seed)))['w_samples']
    y_samples = np.load(os.path.join(LOAD_PATH, 'cond_exp_metric_{}_seed{}.npz'.format(args.sem, data_seed)))['y_samples']
    y_axz = axzy[:, -1]
    ax = axzy[:, :2]
    t2 = time.time()

    print('err_in_expectation metric loading used {}s'.format(t2 - t1))

    return w_samples, y_samples, y_axz, ax, axzy


def load_test_supp_metric(args, data_seed, LOAD_PATH):
    """
    load supported test eval data.
    """
    t2 = time.time()

    test_supp = np.load(os.path.join(LOAD_PATH, 'supported_test_metric_{}_seed{}.npz'.format(args.sem, data_seed)))
    aw_test_supp = test_supp['test_aw']
    az_test_supp = test_supp['test_az']
    y_test_supp = test_supp['test_y']
    t3 = time.time()
    print('supp_metric loading used {}s'.format(t3-t2))
    return aw_test_supp, az_test_supp, y_test_supp


def compute_alpha(train_size, eig_vec_K, W_nystr, X, Y, W, eig_val_K, nystr, params_l):
    N2 = train_size ** 2
    EYEN = np.eye(train_size)

    al, bl = params_l
    print('al, bl = ', params_l)

    print('making K_L')
    K_L = make_gaussian_prodkern(X, X, al)
    print('end of making K_L')

    L = bl * bl * K_L + JITTER_L * EYEN  # L = (1/lambda * 1/(n^2)) * L_true
    # L = bl * bl * K_L
    print('bl * bl * K_L: ', bl * bl * K_L[:10, :10])
    print('L[:10, :10]: ', L[:10, :10])

    if nystr:
        tmp = eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)
        print('condition number of 1st term: ', np.linalg.cond(eig_vec_K.T @ L @ eig_vec_K / N2, p=2))
        print('condition number of tmp: ', np.linalg.cond(tmp, p=2))
        # print('condition number of tmp not divided by N2: ', np.linalg.cond(eig_vec_K.T @ L @ eig_vec_K + np.diag(1 / eig_val_K)))
        print('tmp: ', tmp)
        print('condition number of V~: ', np.linalg.cond(np.diag(1 / eig_val_K / N2), p=2))
        print('V~: ', np.diag(1 / eig_val_K / N2))
        # print('condition number of V~, not divided by N2: ', np.linalg.cond(np.diag(1 / eig_val_K), p=2))
        # print('V~ not divided by N2: ', np.diag(1 / eig_val_K))
        # raise ValueError
        fig = go.Figure(data=[go.Surface(z=tmp, x=np.arange(nystr_M),
                                         y=np.arange(nystr_M))])
        fig.update_layout(title='tmp plot', autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
        del tmp
        alpha = EYEN - eig_vec_K @ np.linalg.inv(
            eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
        alpha = alpha @ W_nystr @ Y * N2
    else:
        print('L condition number: ', np.linalg.cond(L))
        print('W condition number: ', np.linalg.cond(W))
        print('L @ W @ L + L / N2 condition number: ', np.linalg.cond(L @ W @ L + L / N2))
        print('L @ W @ L + L / N2 + JITTER_LW * EYEN condition number: ', np.linalg.cond(L @ W @ L + L / N2 + JITTER_LW * EYEN))
        # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
        LWL_inv = chol_inv(L @ W @ L + L / N2)
        alpha = LWL_inv @ L @ W @ Y

    return alpha


def compute_offset(X, W, Y, alpha, params_l):

    al, bl = params_l
    assert Y.ndim == 2

    K_L = make_gaussian_prodkern(X, X, al)
    L = bl * bl * K_L
    h_train = L @ alpha

    tmp = np.repeat(Y - h_train, [Y.shape[0]], axis=-1)
    M = tmp + tmp.T
    assert M[1,4] == M[4,1]

    offset = 0.5 * np.sum(W * M)

    return offset


def get_causal_effect(do_A, w_marginal, X, alpha, params_l, offset=0):
    "to be called within experiment function."
    assert w_marginal.ndim == 2
    assert do_A.ndim == 2
    al, bl = params_l
    print('W_shape: ', w_marginal.shape[0])
    do_A_rep = np.repeat(do_A, [w_marginal.shape[0]], axis=-1).reshape(-1, 1)
    w_rep = np.tile(w_marginal, [do_A.shape[0], 1])
    aw_rep = np.concatenate([do_A_rep, w_rep], axis=-1)

    print('making K_L ate.')
    K_L_ate = make_gaussian_prodkern(aw_rep, X, al)
    ate_L = bl * bl * K_L_ate  # lambda = 1/(b_l^2n^2), b_l^2 = 1/(lambda n^2)
    print('end of making K_L ate.')
    h_out = ate_L @ alpha

    h_out_a_as_rows = h_out.reshape(-1, w_marginal.shape[0])
    ate_est = np.mean(h_out_a_as_rows, axis=-1).reshape(-1,1) + offset
    print('Estimated ATE: ', ate_est)

    return ate_est.squeeze()


def compute_loss_on_supported_test_set(X, al, bl, alpha, supp_y, supp_aw, supp_az, offset):
    print('making K_L_mse.')
    K_L_mse = make_gaussian_prodkern(supp_aw, X, al)
    print('end of making K_L_mse.')
    mse_L = bl * bl * K_L_mse
    mse_h = mse_L @ alpha + offset
    mse_supp = np.mean((supp_y.flatten() - mse_h.flatten()) ** 2)

    return mse_supp


def mmr_loss(ak, al, bl, alpha, y_test, aw_test, az_test, X, offset):
    print('making K_L_mse not supported.')
    K_L_mse = make_gaussian_prodkern(aw_test, X, al)
    print('end of making K_L_mse not supported.')
    mse_L = bl * bl * K_L_mse
    mse_h = mse_L @ alpha + offset

    print('supp_az shape: ', az_test.shape)
    N = y_test.shape[0]
    print('making K_W test.')
    K = make_gaussian_prodkern(az_test, az_test, ak)
    print('end of making K_W test.')

    W_U = (K - np.diag(np.diag(K)))
    W_V = K

    assert y_test.ndim > 1
    assert mse_h.ndim == y_test.ndim
    for dim in range(mse_h.ndim):
        assert mse_h.shape[dim] == y_test.shape[dim]

    d = mse_h - y_test

    loss_V = d.T @ W_V @ d / N / N
    loss_U = d.T @ W_U @ d / N / (N - 1)
    return loss_V[0, 0], loss_U[0, 0]


def LMO_err_global(log_params_l, train_size, W, W_nystr_Y, eig_vec_K, inv_eig_val_K, X, Y, nystr, offset, M=10):
    EYEN = np.eye(train_size)
    N2 = train_size ** 2

    log_al, log_bl = log_params_l[:-1], log_params_l[-1]
    al, bl = anp.exp(log_al).squeeze(), anp.exp(log_bl).squeeze()
    # print('lmo_err params_l', params_l)
    print('lmo_err al, bl', al, bl)
    K_L = make_gaussian_prodkern(X, X, al)
    L = bl * bl * K_L + JITTER_L * EYEN
    # L = bl * bl * K_L
    print('condition number of L: ', np.linalg.cond(L, p=2))
    if nystr:
        tmp_mat = L @ eig_vec_K
        C = L - tmp_mat @ anp.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
        c = C @ W_nystr_Y * N2
    else:
        # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
        LWL_inv = chol_inv(L @ W @ L + L / N2)
        C = L @ LWL_inv @ L / N2
        c = C @ W @ Y * N2
    c_y = c - Y
    lmo_err = 0
    N = 0
    for ii in range(1):
        idxs = np.arange(X.shape[0])
        for i in range(0, X.shape[0], M):
            indices = idxs[i:i + M]
            K_i = W[np.ix_(indices, indices)] * N2
            C_i = C[np.ix_(indices, indices)]
            c_y_i = c_y[indices]
            b_y = anp.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
            lmo_inc = (b_y - offset).T @ K_i @ (b_y - offset)
            lmo_err = lmo_err + lmo_inc
            # print('lmo_inc: ', lmo_inc)
            N += 1
    print('LMO-err: ', lmo_err[0, 0] / N / M ** 2)
    return lmo_err[0, 0] / N / M ** 2


def compute_losses(params_l, ax, w_samples, y_samples, y_axz, x_on,
                   AW_test, AZ_test, Y_test, supp_y, supp_aw, supp_az,
                   X, Y, W, W_nystr_Y, eig_vec_K, inv_eig_val_K, nystr, test_Y, ak, alpha, offset, w_dim, args):
    "to calculated the expected error E_{A,X,Z ~ unif}[E[Y - h(A,X,W)|A,X,Z]]."

    al, bl = params_l
    args.al = al
    args.bl = bl

    if args.cond_metric:
        if not x_on:
            ax = ax[:, 0:1]

        num_reps = w_samples.shape[1] // w_dim
        assert len(ax.shape) == 2
        assert ax.shape[1] < 3
        assert ax.shape[0] == w_samples.shape[0]
        # print('number of points: ', w_samples.shape[0])

        ax_rep = np.repeat(ax, [num_reps], axis=0)
        assert ax_rep.shape[0] == (w_samples.shape[1] * ax.shape[0])

        w_samples_flat = w_samples.flatten().reshape(-1, w_dim)
        axw = np.concatenate([ax_rep, w_samples_flat], axis=-1)

        K_L_axw = make_gaussian_prodkern(axw, X, al)
        expected_err_L = bl * bl * K_L_axw
        h_out = expected_err_L @ alpha + offset

        h_out = h_out.reshape([-1, w_samples.shape[1]//w_dim])
        y_axz_recon = np.mean(h_out, axis=1)
        assert y_axz_recon.shape[0] == y_axz.shape[0]
        mean_sq_error = np.mean(np.square(y_axz - y_axz_recon))

        # for debugging compute the mse between y samples and h
        y_samples_flat = y_samples.flatten()
        mse_alternative = np.mean((y_samples_flat - h_out.flatten()) ** 2)
    else:
        mean_sq_error, mse_alternative, y_axz_recon = None, None, None

    # standard mse
    K_L_mse = make_gaussian_prodkern(AW_test, X, al)
    mse_L = bl * bl * K_L_mse
    mse_h = mse_L @ alpha + offset
    mse_standard = np.mean((test_Y.flatten() - mse_h.flatten()) ** 2)

    # standard mse on support
    if args.supp_test:
        mse_supp = compute_loss_on_supported_test_set(X=X, al=al, bl=bl, alpha=alpha,
                                                      supp_y=supp_y, supp_aw=supp_aw, supp_az=supp_az, offset=offset)
        mmr_v_supp, mmr_u_supp = mmr_loss(ak=ak, al=al, bl=bl, alpha=alpha, y_test=supp_y, aw_test=supp_aw,
                                          az_test=supp_az, X=X, offset=offset)
    else:
        mse_supp, mmr_v_supp, mmr_u_supp = None, None, None

    # mmr losses
    mmr_v, mmr_u = mmr_loss(ak=ak, al=al, bl=bl, alpha=alpha, y_test=Y_test, aw_test=AW_test, az_test=AZ_test, X=X, offset=offset)

    # lmo
    log_params = np.append(np.log(params_l[0]), np.log(params_l[1]))
    lmo_err = LMO_err_global(log_params_l=log_params, train_size=X.shape[0], W=W, W_nystr_Y=W_nystr_Y,
                             eig_vec_K=eig_vec_K, inv_eig_val_K=inv_eig_val_K, X=X, Y=Y, nystr=nystr, M=1, offset=offset)

    return {'err_in_expectation': mean_sq_error,
            'mse_alternative': mse_alternative,
            'y_axz_recon': y_axz_recon,
            'mse_standard': mse_standard,
            'mse_supp': mse_supp,
            'mmr_v_supp': mmr_v_supp,
            'mmr_v': mmr_v,
            'mmr_u_supp': mmr_u_supp,
            'mmr_u': mmr_u,
            'lmo': lmo_err}


def get_results(do_A, EYhat_do_A, EY_do_A_gt, train_sz, err_in_expectation, mse_alternative, mse_standard, mse_supp,
                mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo, params_l, args, SAVE_PATH, LOAD_PATH, data_seed):
    plt.figure()
    plt.plot(do_A.squeeze(), EYhat_do_A, label='est')
    plt.plot(do_A.squeeze(), EY_do_A_gt, label='gt')
    plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    plt.savefig(
        os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                     'causal_effect_estimates_nystr_trainsz{}_al{}_bl{}_offset{}.png'.format(train_sz, params_l[0], params_l[1], args.offset)))
    plt.close()
    print('ground truth ate: ', EY_do_A_gt)
    print('Causal MAE: ', np.mean(np.abs(EY_do_A_gt.squeeze() - EYhat_do_A.squeeze())))
    visualise_ATEs(EY_do_A_gt, EYhat_do_A,
                   x_name='E[Y|do(A)] - gt',
                   y_name='beta_A',
                   save_loc=os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed)),
                   save_name='ate_trainsz{}_al{}_bl{}_nystr_offset{}'.format(train_sz, params_l[0], params_l[1], args.offset))

    causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt.squeeze() - EYhat_do_A.squeeze()))

    causal_std = np.std(np.abs(EYhat_do_A.squeeze() - EY_do_A_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((EYhat_do_A.squeeze() - EY_do_A_gt.squeeze())/EY_do_A_gt.squeeze()))

    summary_file = open(
        os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), "summary_trainsz{}_nystrom_hparam{}_offset{}.txt".format(train_sz, args.hparam, args.offset)),
        "a")
    summary_file.write("al: {}, bl: {}, causal_mae_: {}, causal_std: {}, causal_rel_err: {}\n"
                       "expected_error : {}, mmr_v_supp: {}, mmr_v: {}, mmr_u_supp: {}, mmr_u: {}\n"
                       "mse: {}, mse_supp: {}, lmo: {}\n"
                       "causal_est: {}\n"
                       "causal_gt: {}\n".format(params_l[0], params_l[1], causal_effect_mean_abs_err, causal_std,
                                                 causal_rel_err, err_in_expectation, mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, mse_standard, mse_supp, lmo, EYhat_do_A, EY_do_A_gt.squeeze()))
    summary_file.close()

    if args.hparam == 'lmo':
        os.makedirs(SAVE_PATH, exist_ok=True)
        np.save(os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed),
                             'LMO_errs_params_l{}_nystr_trainsz{}_offset{}.npy'.format(params_l, train_sz, args.offset)),
                [opt_params_l, prev_norm, opt_test_err])

    return causal_effect_mean_abs_err


def experiment(seed, data_seed, param_l_arg, train_size, dev_size, test_size, nystr, args, SAVE_PATH, LOAD_PATH):
    np.random.seed(seed)
    random.seed(seed)

    X, Y, Z, test_X, test_Y, test_Z, W_marginal, do_A, EY_do_A_gt, w_dim = process_data(train_size=train_size, dev_size=dev_size,
                                                                                        test_size=test_size,
                                                                                        args=args, data_seed=data_seed,
                                                                                        LOAD_PATH=LOAD_PATH)

    w_samples, y_samples, y_axz, ax, axzy = load_err_in_expectation_metric(args, data_seed=data_seed) if args.cond_metric else (None, None, None, None, None)
    aw_test_supp, az_test_supp, y_test_supp = load_test_supp_metric(args, data_seed=data_seed) if args.supp_test else (None, None, None)

    al_default = compute_bandwidth_median_dist(X=X)

    EYEN = np.eye(X.shape[0])

    print('Z shape: ', Z.shape)
    # ak = compute_bandwidth_median_dist(Z) / [1.5, 1.5, 1.5]
    ak = compute_bandwidth_median_dist(Z) / 0.6
    N2 = X.shape[0] ** 2
    print('making W_.')
    W_ = make_gaussian_prodkern(Z, Z, sigma=ak) + JITTER_W * EYEN
    print('end of making W_.')
    print('W_ condition number: ', np.linalg.cond(W_, p=2))
    W = W_ / N2
    print('ak = ', ak)
    print('W[:10, :10]: ', W[:10, :10])

    # L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)

    if nystr:
        random_indices = np.sort(np.random.choice(range(W.shape[0]), args.nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp_from_orig(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr_ = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T
        W_nystr = W_nystr_ / N2
        print('(W_nystr sub - W)/W: ', (W_nystr_[:10, :10] - W_[:10, :10])/W_[:10, :10])
        print('sum (W_nystr sub - W)/W: ', np.sum(np.abs(W_nystr_ - W_)))
        W_nystr_Y = W_nystr @ Y
    else:
        eig_vec_K, eig_val_K, inv_eig_val_K = None, None, None
        W_nystr, W_nystr_Y = None, None



    al, bl = None, None
    if args.hparam == 'lmo':
        global Nfeval, prev_norm, opt_params_l, opt_test_err
        log_al0, log_bl0 = np.log(al_default), np.random.randn(1)
        params_l0 = np.append(log_al0, log_bl0)
        opt_params_l = (np.exp(log_al0), np.exp(log_bl0))
        prev_norm, opt_test_err = None, None
        print('starting param log_al0: {}, log_bl0: {}'.format(log_al0, log_bl0))
        bounds = None

        def LMO_err(params_l, M=10):
            log_al, log_bl = params_l[:-1], params_l[-1]
            al, bl = anp.exp(log_al).squeeze(), anp.exp(log_bl).squeeze()
            # print('lmo_err params_l', params_l)
            print('lmo_err al, bl', al, bl)
            K_L = make_gaussian_prodkern(X, X, al)
            # L = bl * bl * K_L + JITTER * EYEN
            L = bl * bl * K_L + JITTER_L * EYEN

            if nystr:
                tmp_mat = L @ eig_vec_K
                C = L - tmp_mat @ anp.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
                c = C @ W_nystr_Y * N2
            else:
                # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                LWL_inv = chol_inv(L @ W @ L + L / N2)
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
                    b_y = anp.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                    lmo_inc = b_y.T @ K_i @ b_y
                    lmo_err = lmo_err + lmo_inc
                    # print('lmo_inc: ', lmo_inc)
                    N += 1
            print('LMO-err: ', lmo_err[0, 0] / N / M ** 2)
            # alpha = compute_alpha(train_size, eig_vec_K, W_nystr, X, Y, W, eig_val_K, nystr, anp.exp(params_l))
            # get_causal_effect(do_A, w_marginal, X, alpha, params_l, offset=0)
            return lmo_err[0, 0] / N / M ** 2

        def LMO_log_bl(log_bl):
            params_l = np.array([log_al0, log_bl0])
            return LMO_err(params_l=params_l, M=10)

        def callback0(params_l, timer=None):
            global Nfeval, prev_norm, opt_params_l, opt_test_err
            # np.random.seed(3)
            # random.seed(3)
            if Nfeval % 1 == 0:
                log_al, log_bl = params_l[:-1], params_l[-1]
                al, bl = np.exp(log_al).squeeze(), np.exp(log_bl).squeeze()
                print('callback al, bl', al, bl)
                K_L = make_gaussian_prodkern(arr1=X, arr2=X, sigma=al)
                L = bl * bl * K_L + JITTER_L * EYEN
                # L = bl * bl * K_L
                if nystr:
                    alpha = EYEN - eig_vec_K @ np.linalg.inv(
                        eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                    alpha = alpha @ W_nystr @ Y * N2
                else:
                    # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                    LWL_inv = chol_inv(L @ W @ L + L / N2)
                    alpha = LWL_inv @ L @ W @ Y
                    # L_W_inv = chol_inv(W*N2+L_inv)
                K_L_test = make_gaussian_prodkern(arr1=test_X, arr2=X, sigma=al)
                test_L = bl * bl * K_L_test
                pred_mean = test_L @ alpha
                if timer:
                    return
                test_err = ((pred_mean - test_Y) ** 2).mean()  # ((pred_mean-test_Y)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
                norm = alpha.T @ L @ alpha

            Nfeval += 1
            prev_norm = norm[0, 0]
            opt_test_err = test_err
            opt_params_l = (al, bl)
            print('params_l,test_err, norm: ', opt_params_l, opt_test_err, norm[0, 0])

        def callback_log_bl(log_bl):
            global Nfeval, prev_norm, opt_params_l, opt_test_err
            # np.random.seed(3)
            # random.seed(3)
            if Nfeval % 1 == 0:
                al, bl = np.exp(log_al0).squeeze(), np.exp(log_bl).squeeze()
                print('callback al, bl', al, bl)
                K_L = make_gaussian_prodkern(arr1=X, arr2=X, sigma=al)
                L = bl * bl * K_L + JITTER_L * EYEN
                # L = bl * bl * K_L
                if nystr:
                    alpha = EYEN - eig_vec_K @ np.linalg.inv(
                        eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                    alpha = alpha @ W_nystr @ Y * N2
                else:
                    # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                    LWL_inv = chol_inv(L @ W @ L + L / N2)
                    alpha = LWL_inv @ L @ W @ Y
                    # L_W_inv = chol_inv(W*N2+L_inv)
                K_L_test = make_gaussian_prodkern(arr1=test_X, arr2=X, sigma=al)
                test_L = bl * bl * K_L_test
                pred_mean = test_L @ alpha
                test_err = ((pred_mean - test_Y) ** 2).mean()  # ((pred_mean-test_Y)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
                norm = alpha.T @ L @ alpha

            Nfeval += 1
            prev_norm = norm[0, 0]
            opt_test_err = test_err
            opt_params_l = bl
            print('params_l,test_err, norm: ', opt_params_l, opt_test_err, norm[0, 0])

        if args.lmo == 'albl':
            obj_grad = value_and_grad(lambda params_l: LMO_err(params_l))
            x0 = params_l0
            dim_bandwidth = x0.shape[0] - 1
            log_al_bounds = np.tile(args.log_al_bounds, [dim_bandwidth, 1])
            log_bl_bounds = args.log_bl_bounds
            bounds = np.concatenate((log_al_bounds, log_bl_bounds.reshape(1,-1)), axis=0)
            cb = callback0
        elif args.lmo == 'bl':
            obj_grad = value_and_grad(lambda log_bl: LMO_log_bl(log_bl))
            x0 = np.array([log_bl0])
            bounds = [args.bl_bounds]
            cb = callback_log_bl
        else:
            raise NotImplementedError

        # res = minimize(obj_grad, x0=x0, bounds=bounds, method='L-BFGS-B',
        #            jac=True, options={'maxiter': 5000}, callback=cb, tol=1e-3)

        try:
            res = minimize(obj_grad, x0=x0, bounds=bounds, method='L-BFGS-B',
                       jac=True, options={'maxiter': 5000}, callback=cb, tol=1e-3)
        except Exception as e:
            print(e)
        # print(res)

        if opt_params_l is None:
            params_l_final = np.exp(params_l0)
        else:
            params_l_final = opt_params_l

        args.al_lmo = params_l_final[:-1]
        args.bl_lmo = params_l_final[1]

    elif args.hparam == 'cube':
        al, bl = al_default + param_l_arg[0], param_l_arg[1]
        print('bandwidth = ', al)
        params_l_final = [al, bl]

    elif args.hparam == 'fixed':
        params_l_final = param_l_arg
    else:
        raise NotImplementedError

    alpha = compute_alpha(train_size=train_size, eig_vec_K=eig_vec_K, W_nystr=W_nystr, X=X, Y=Y, W=W,
                          eig_val_K=eig_val_K, nystr=nystr, params_l=params_l_final)

    offset = compute_offset(X=X, W=W, Y=Y, alpha=alpha, params_l=params_l_final) if args.offset else 0
    print('******************* al, bl = {}, {}, offset = {}'.format(al, bl, offset))
    EYhat_do_A = get_causal_effect(do_A=do_A, w_marginal=W_marginal, X=X, alpha=alpha, params_l=params_l_final, offset=offset)

    losses = compute_losses(params_l=params_l_final, ax=ax, w_samples=w_samples, y_samples=y_samples, y_axz=y_axz,
                                            x_on=False, AW_test=test_X, AZ_test=test_Z, Y_test=test_Y,
                                            supp_aw=aw_test_supp, supp_y=y_test_supp, supp_az=az_test_supp, X=X, Y=Y,
                                            W=W, W_nystr_Y=W_nystr_Y, eig_vec_K=eig_vec_K, inv_eig_val_K=inv_eig_val_K,
                                            nystr=nystr, test_Y=test_Y, ak=ak, alpha=alpha, offset=offset, w_dim=w_dim, args=args)

    causal_effect_mean_abs_err = get_results(do_A=do_A, EYhat_do_A=EYhat_do_A, EY_do_A_gt=EY_do_A_gt, train_sz=train_size,
                                             err_in_expectation=losses['err_in_expectation'], mse_alternative=losses['mse_alternative'],
                                             mse_standard=losses['mse_standard'], mse_supp=losses['mse_supp'],
                                             mmr_v_supp=losses['mmr_v_supp'], mmr_v=losses['mmr_v'],
                                             mmr_u_supp=losses['mmr_u_supp'], mmr_u=losses['mmr_u'], lmo=losses['lmo'],
                                             params_l=params_l_final, args=args,
                                             SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, data_seed=data_seed)

    return causal_effect_mean_abs_err, losses['err_in_expectation'], al_default, EYhat_do_A, losses['mse_standard'], \
           losses['mse_supp'], losses['mmr_v_supp'], losses['mmr_v'], losses['mmr_u_supp'], losses['mmr_u'], losses['lmo']


def do_bl_hparam_analysis_plots(SAVE_PATH, LOAD_PATH, args, train_size, bl_min, bl_max, bl_mesh_size, data_seed, **h_param_results_dict):
    deltas = np.linspace(bl_min, bl_max, bl_mesh_size)**2
    ldas = 1/deltas/train_size/train_size
    print('ldas: ', ldas)

    print('plotting')
    os.makedirs(os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed)), exist_ok=True)
    causal_mae = np.array(h_param_results_dict['causal_mae'])[:, -1]
    mean_causal_mae = np.mean(causal_mae)
    causal_mae_rescaled = 1/mean_causal_mae * causal_mae
    print('keys: ', h_param_results_dict.keys())
    for var_str in h_param_results_dict.keys():
        print(var_str)
        print((var_str == 'al_default') or (var_str=='causal_mae') or (var_str=='ate_est'))
        boolean = (var_str == 'al_default') or (var_str=='causal_mae') or (var_str=='ate_est')
        if boolean:
            continue
        var = np.array(h_param_results_dict[var_str])[:, -1]
        if var[0] is None:
            continue
        mean_var = np.mean(var)
        var_rescaled = 1/np.abs(mean_var) * var
        length = var.shape[0]
        plt.figure()
        plt.plot(ldas, var_rescaled, label=var_str)
        plt.plot(ldas, causal_mae_rescaled, label='causal_mae')
        plt.xlim(max(ldas), min(ldas))
        plt.xlabel('Hyperparameter labels'), plt.ylabel('causal_MAE/{}-rescaled'.format(var_str)), plt.legend()
        print('save path: ', os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}.png'.format(var_str, train_size, args.offset)))
        plt.savefig(os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}.png'.format(var_str, train_size, args.offset)))
        plt.close()

        plt.figure()
        plt.plot(np.arange(length), var_rescaled, label=var_str)
        plt.plot(np.arange(length), causal_mae_rescaled, label='causal_mae')
        plt.xlabel('Hyperparameter labels'), plt.ylabel('causal_MAE/{}-rescaled'.format(var_str)), plt.legend()
        print('save path: ', os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}_inversc.png'.format(var_str, train_size, args.offset)))
        plt.savefig(os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'hparam_anal_{}_trainsz{}_offset{}_inversc.png'.format(var_str, train_size, args.offset)))
        plt.close()


def rank_nums_in_array(arr):
    '''
    helper to rank the numbers in an array. e.g. input = np.arr([2,1,3]), output = np.arr([1,0,2]).
    '''
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))

    return ranks


def hparam_selection_from_metric_votes(mmr_v, mmr_v_supp, lmo, hparam_arr, args):
    ranks = None
    if args.selection_metric == 'mmr_v':
        ranks_mmr_v = rank_nums_in_array(mmr_v.squeeze())
        ranks = ranks_mmr_v
    elif args.selection_metric == 'mmr_v_supp':
        ranks_mmr_v_supp = rank_nums_in_array(mmr_v_supp.squeeze())
        ranks = ranks_mmr_v_supp
    elif args.lmo == 'lmo':
        ranks_lmo = rank_nums_in_array(lmo.squeeze())
        ranks = ranks_lmo
    # sum_ranks = ranks_mmr_v + ranks_mmr_v_supp + ranks_lmo
    sum_ranks = ranks
    idx_argmin = sum_ranks.argmin()
    return hparam_arr[idx_argmin]


def get_best_hparam(results_dict, args):
    hparam_arr = np.array(results_dict['causal_mae'])[:, :2]
    mmr_vs = np.array(results_dict['mmr_v'])[:, -1]
    mmr_v_supps = np.array(results_dict['mmr_v_supp'])[:, -1]
    lmos = np.array(results_dict['lmo'])[:, -1]
    return hparam_selection_from_metric_votes(mmr_v=mmr_vs, mmr_v_supp=mmr_v_supps, lmo=lmos, hparam_arr=hparam_arr, args=args)


def cube_search(al_diff_min, al_diff_max, al_mesh_size, bl_min, bl_max, bl_mesh_size, args, SAVE_PATH, LOAD_PATH, train_size, seed, data_seed):
    al_arr, bl_arr = np.linspace(al_diff_min, al_diff_max, al_mesh_size), np.linspace(bl_min, bl_max, bl_mesh_size)
    results_dict = {'causal_mae': [],
                    'err_in_expectation': [],
                    'mse_supp': [],
                    'mse_standard': [],
                    'mmr_v_supp': [],
                    'mmr_v': [],
                    'mmr_u_supp': [],
                    'mmr_u': [],
                    'lmo': [],
                    'ate_est': {}}

    for al in al_arr:
        for bl in bl_arr:
            params_l = [al, bl]
            print('fitting al = {}, bl = {}'.format(al, bl))
            causal_effect_mean_abs_err, err_in_expectation, al_default, causal_effect_est, mse_standard, mse_supp, \
            mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo = experiment(seed=seed, data_seed=data_seed, param_l_arg=params_l,
                                                                   train_size=train_size, test_size=test_sz, dev_size=dev_sz,
                                                                   nystr=(False if train_size <= nystr_thresh else True),
                                                                   args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH)

            if 'al_default' not in results_dict:
                results_dict['al_default'] = al_default
            results_dict['causal_mae'].append([al_default + al, bl, causal_effect_mean_abs_err])
            results_dict['err_in_expectation'].append([al_default + al, bl, err_in_expectation])
            results_dict['mse_supp'].append([al_default + al, bl, mse_supp])
            results_dict['mse_standard'].append([al_default + al, bl, mse_standard])
            results_dict['mmr_v_supp'].append([al_default + al, bl, mmr_v_supp])
            results_dict['mmr_v'].append([al_default + al, bl, mmr_v])
            results_dict['mmr_u_supp'].append([al_default + al, bl, mmr_u_supp])
            results_dict['mmr_u'].append([al_default + al, bl, mmr_u])
            results_dict['lmo'].append([al_default + al, bl, lmo])
            results_dict['ate_est'][tuple(np.append(np.array(al_default + al), bl))] = causal_effect_est
    # do_bl_hparam_analysis_plots(SAVE_PATH=SAVE_PATH, args=args, train_size=train_size,
    #                          bl_min=bl_min, bl_max=bl_max, bl_mesh_size=bl_mesh_size,
    #                          data_seed=data_seed, **results_dict)
    best_hparams_l = get_best_hparam(results_dict, args=args)
    best_ate_est = results_dict['ate_est'][tuple(np.append(best_hparams_l[0], best_hparams_l[1]))]

    print('best mae for {} found at params_l: {} using hparam search method: {}'.format(sname, best_hparams_l,
                                                                                      args.hparam))
    with open(os.path.join(SAVE_PATH, str(date.today()), args.sem+'_seed'+str(data_seed), 'best_params_l_cube_trainsz{}_offset{}.txt'.format(train_size, args.offset)), 'w') as f:
        f.write('best al: {}, best bl: {:.3f}'.format(best_hparams_l[0] + al_default, best_hparams_l[1]))

    return best_hparams_l, best_ate_est


def hyparameter_selection(al_diff_min, al_diff_max, al_mesh_size, bl_min, bl_max, bl_mesh_size, args, SAVE_PATH, LOAD_PATH, train_size, seed, data_seed):
    if args.hparam == 'cube':
        best_hparams_l, best_ate_est = cube_search(al_diff_min=al_diff_min, al_diff_max=al_diff_max, al_mesh_size=al_mesh_size,
                                                   bl_min=bl_min, bl_max=bl_max, bl_mesh_size=bl_mesh_size,
                                                   args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH, train_size=train_size, data_seed=data_seed, seed=seed)
        return best_hparams_l, best_ate_est

    elif args.hparam == 'lmo':
        causal_effect_mean_abs_err, err_in_expectation, al_default, causal_effect_est, mse_standard, \
        mse_supp, mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, lmo = experiment(seed=seed, data_seed=data_seed, param_l_arg=[],
                                                                         train_size=train_size, dev_size=dev_sz, test_size=test_sz,
                                                                         nystr=(False if train_size <= nystr_thresh else True),
                                                                         args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH)

        return [args.al_lmo, args.bl_lmo], causal_effect_est
    else:
        raise NotImplementedError


def evaluate_ate_est(ate_est, ate_gt):
    causal_mae = np.mean(np.abs(ate_gt.squeeze() - ate_est.squeeze()))
    causal_std = np.std(np.abs(ate_est.squeeze() - ate_gt.squeeze()))
    causal_rel_err = np.mean(np.abs((ate_est.squeeze() - ate_gt.squeeze())/ate_gt.squeeze()))

    return causal_mae, causal_std, causal_rel_err


def run_pmmr_rkhs(seed, al_diff_search_range, bl_search_range, train_sizes, data_seeds, args):
    al_diff_min, al_diff_max, al_mesh_size = al_diff_search_range
    bl_min, bl_max, bl_mesh_size = bl_search_range
    for train_size in train_sizes:
        causal_mae_over_seeds = []
        for data_seed in data_seeds:
            SAVE_PATH = os.path.join(ROOT_PATH, "results", sname)
            LOAD_PATH = os.path.join(ROOT_PATH, "data", sname)
            os.makedirs(os.path.join(SAVE_PATH, str(date.today()), args.sem + '_seed' + str(data_seed)), exist_ok=True)

            summary_file = open(
                os.path.join(SAVE_PATH, str(date.today()), args.sem + '_seed' + str(data_seed),
                             "summary_trainsz{}_nystrom_hparam{}_offset{}.txt".format(int(train_size), args.hparam, args.offset)), "w")
            summary_file.close()

            do_A = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))['do_A']
            EY_do_A_gt = np.load(os.path.join(LOAD_PATH, 'do_A_{}_seed{}.npz'.format(args.sem, data_seed)))[
                'gt_EY_do_A']

            best_hparams_l, best_ate_est = hyparameter_selection(seed=seed, al_diff_max=al_diff_max, al_diff_min=al_diff_min,
                                                                 al_mesh_size=al_mesh_size, bl_min=bl_min, bl_max=bl_max,
                                                                 bl_mesh_size=bl_mesh_size, args=args, SAVE_PATH=SAVE_PATH, LOAD_PATH=LOAD_PATH,
                                                                 train_size=train_size, data_seed=data_seed)
            best_causal_mae, best_causal_std, best_causal_rel_err = evaluate_ate_est(ate_est=best_ate_est,
                                                                                     ate_gt=EY_do_A_gt)
            np.savez(os.path.join(SAVE_PATH, str(date.today()), args.sem + '_seed' + str(data_seed),
                                  'mmr_res_trainsz{}_offset{}.npz'.format(train_size, args.offset)), do_A=do_A, ate_est=best_ate_est,
                     bl=best_hparams_l[1], train_sz=train_size,
                     causal_mae=best_causal_mae, causal_std=best_causal_std, causal_rel_err=best_causal_rel_err)
            causal_mae_over_seeds.append(best_causal_mae)
        print('av c-MAE: ', np.mean(causal_mae_over_seeds))


if __name__ == '__main__':
    run_pmmr_rkhs(seed=seed, al_diff_search_range=al_diff_search_range, bl_search_range=bl_search_range,
                  train_sizes=train_sizes, data_seeds=data_seeds, args=args)

