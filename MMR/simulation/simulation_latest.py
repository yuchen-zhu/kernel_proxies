import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# TODO: in the final simulation, U should not be from a uniform distribution, bc unif is not exponential family - use normal or chi-squared.
do_A_min, do_A_max = 5, 14

N = [500, 5000, 10000, 100000, 5000000]
a_AY = 0.5
a_AZ = 0.5
a_WY = 0.5

m_e = [5, 0, 0, -1, 2]
C = [3, 3, 3, 3, 3, 2]

centres_no_w = np.array([[0., 4.], [0., 12.], [0., 20.], [0., 28.], [0., 36.]])
centres_w = np.array([[2., 4., 4.], [4., 12., 12.], [6., 20., 20.], [8., 28., 28.], [10., 36., 36.]])
coeffs_ = np.array([1., 1., 1., 1., 1.])
sig_ = 7.


parser = argparse.ArgumentParser(description='settings for simulation')
parser.add_argument('--fullgraph', type=bool, default=False, help='generate the full proxy graph')
parser.add_argument('--sem', type=str, help='sets SEM')
parser.add_argument('--fun-ay', type=str, default=None, help='sets function from a to y')
parser.add_argument('--udist', type=str, help='sets the distribution for U')
parser.add_argument('--get-a', type=str, help='sets function to get a')
parser.add_argument('--get-y', type=str, help='sets function to get y')
args = parser.parse_args()



def asina(a, b):
    return np.sin(b * a) * a


def asina_mild(a, b):
    return np.sin(b * a) * a * b


def expa(a, b):
    return np.exp(b * a)


def linear(a, b):
    return b * a


def get_a_inpsininp(inp):
    return 3 * np.sin(inp) + 1.5*inp


def get_a_quadratic(inp):
    return -1 * (inp - 5) ** 2 +10


def get_y_rbf(u, w, a, centres, sig, coeffs):
    """
    input: u, w, a - shape = N x dim
    """

    u = u.reshape(-1,1) if len(u.shape) == 1 else None
    w = w.reshape(-1, 1) if len(w.shape) == 1 else None
    a = a.reshape(-1, 1) if len(a.shape) == 1 else None

    uwa = np.concatenate([u,w,a], axis=-1)
    # centres = np.repeat(centres, uwa.shape[-1], axis=-1).reshape(-1, uwa.shape[-1])

    CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1,1), uwa.shape[0], axis=-1)
    UWAUWA = np.repeat(np.sum(uwa ** 2, axis=-1).reshape(1,-1), centres.shape[0], axis=0)
    CUWA = centres @ uwa.T

    H = np.exp(-1 * (CC + UWAUWA - 2 * CUWA)/2/sig/sig)
    Y = coeffs.dot(H)

    return Y


def get_y_rbf_no_w(u, a, centres, sig, coeffs):
    """
    input: u, a - shape = N x dim
    """
    # print(len(u.shape))
    u = u.reshape(-1,1) if len(u.shape) == 1 else None
    a = a.reshape(-1, 1) if len(a.shape) == 1 else None

    ua = np.concatenate([u,a], axis=-1)
    # centres = np.repeat(centres, ua.shape[-1], axis=-1).reshape(-1, ua.shape[-1])

    CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1,1), ua.shape[0], axis=-1)
    UAUA = np.repeat(np.sum(ua ** 2, axis=-1).reshape(1,-1), centres.shape[0], axis=0)
    # print(ua.shape)
    CUWA = centres @ ua.T

    H = np.exp(-1 * (CC + UAUA - 2 * CUWA)/2/sig/sig)
    Y = coeffs.dot(H)

    return Y

# def get_z(u, centres, sig, coeff):
#     CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1, 1), u.shape[0], axis=-1)
#     UU = np.repeat(np.sum(u ** 2, axis=-1).reshape(1, -1), centres.shape[0], axis=0)
#     CU = centres @ u.T
#
#     H = np.exp(-1 * (CC + UU - 2 * CU) / 2 / sig / sig)
#     W = coeff.dot(H)
#
# def get_w(u, centres, sig, coeff):
#     CC = np.repeat(np.sum(centres ** 2, axis=-1).reshape(-1, 1), u.shape[0], axis=-1)
#     UU = np.repeat(np.sum(u ** 2, axis=-1).reshape(1, -1), centres.shape[0], axis=0)
#     CU = centres @ u.T
#
#     H = np.exp(-1 * (CC + UU - 2 * CU) / 2 / sig / sig)
#     W = coeff.dot(H)
#
#     return W


func_dict = {'asina': asina, 'expa': expa, 'linear': linear, 'asina_mild': asina_mild, 'rbf_y': get_y_rbf, 'rbf_y_no_w': get_y_rbf_no_w}


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

    return axzy_np, w_samples_out_np, y_samples_out_np


def main(args):
    np.random.seed(100)
    if args.udist == 'chisq':
        U = np.random.chisquare(m_e[0], N[-1]).round(3)
    else:
        U = np.random.uniform(low=1, high=10, size=N[-1]).round(3)
    train_u, test_u, dev_u, rest_u = U[:3000], U[3000:4000], U[4000:5000], U[5000:]
    U_inst = np.ones(N[-1]).round(3)

    # X = np.random.chisquare(m_e[0], N[-1]).round(3)  # generates 5000 X's to 3.d.p.
    # train_x, test_x, dev_x, rest_x = X[:3000], X[3000:4000], X[4000:5000], X[5000:]
    # X_inst = np.ones(N[-1]).round(3)

    X = np.zeros((N[-1],)).round(3)  # generates 5000 X's to 3.d.p.
    train_x, test_x, dev_x, rest_x = X[:3000], X[3000:4000], X[4000:5000], X[5000:]

    # m_e = [5, 0, 0, -1, 2]
    # C = [1, 1, 1, 1, 1, 2]

    # Z is noisy reading of U
    eZ = np.random.normal(m_e[3], C[3], N[-1])  # noise for Z
    Z = (eZ - U).round(3)
    train_z, test_z, dev_z, rest_z = Z[:3000], Z[3000:4000], Z[4000:5000], Z[5000:]
    Z_conU = (eZ - U_inst).round(3)  # constant U

    # noise for W
    eW = np.random.normal(m_e[4], C[4], N[-1])
    W = (1.4*eW - 0.6 * U).round(3)
    train_w, test_w, dev_w, rest_w = W[:3000], W[3000:4000], W[4000:5000], W[5000:]
    W_conU = (eW + 2 * U_inst).round(3)

    eA = np.random.normal(m_e[1], C[1], N[-1])
    if args.fullgraph:
        if args.get_a == 'linear':
            A = (eA + 2 * U + a_AZ * Z).round(3)
            A_conU = (eA + 2 * U_inst + a_AZ * Z_conU).round(3)
        elif args.get_a == 'inpsininp':
            A = (get_a_inpsininp(2 * U + a_AZ * 0.5 * Z) + 0.8*eA).round(3)
            A_conU = (get_a_inpsininp(2 * U_inst + a_AZ * 0.5 * Z_conU) + 0.8*eA).round(3)
        elif args.get_a == 'quadratic':
            A = get_a_quadratic((2 * U + a_AZ * Z) + eA).round(3)
            A_conU = (get_a_quadratic(2 * U_inst + a_AZ * Z) + eA).round(3)
    else:
        if args.get_a == 'linear':
            A = (eA + 2 * U).round(3)
            A_conU = (eA + 2 * U_inst).round(3)
        elif args.get_a == 'inpsininp':
            A = (get_a_inpsininp(2 * U) + 1.2*eA).round(3)
            A_conU = (get_a_inpsininp(2 * U_inst) + 1.2*eA).round(3)
        elif args.get_a == 'quadratic':
            A = get_a_quadratic((2 * U) + eA).round(3)
            A_conU = (get_a_quadratic(2 * U_inst) + eA).round(3)
    train_a, test_a, dev_a, rest_a = A[:3000], A[3000:4000], A[4000:5000], A[5000:]


    eY = np.random.normal(m_e[2], C[2], N[-1])

    if args.fun_ay is not None:
        fun_ay, get_y = func_dict[args.fun_ay], None
    else:
        fun_ay, get_y = None, func_dict[args.get_y]

    if args.fullgraph:
        Y = (fun_ay(a=A, b=a_AY) + eY + 2 * U + a_WY * W).round(3) if fun_ay is not None else (get_y(U, W, A, centres=centres_w, sig=sig_, coeffs=coeffs_) + 0.1*eY).round(3)
        Y_conU = (fun_ay(a=A_conU, b=a_AY) + eY + 2 * U_inst + a_WY * W_conU).round(3) if fun_ay is not None else (get_y(U_inst, W_conU, A_conU, centres=centres_w, sig=sig_, coeffs=coeffs_) + 0.1*eY).round(3)
    else:
        Y = (fun_ay(a=A, b=a_AY) + eY + 2 * U).round(3) if fun_ay is not None else get_y(U, A, centres=centres_no_w, sig=sig_, coeffs=coeffs_)
        Y_conU = (fun_ay(a=A_conU, b=a_AY) + eY + 2 * U_inst).round(3) if fun_ay is not None else get_y(U_inst, A_conU, centres=centres_no_w, sig=sig_, coeffs=coeffs_)
    train_y, test_y, dev_y, rest_y = Y[:3000], Y[3000:4000], Y[4000:5000], Y[5000:]


    # causal ground truth
    do_A = np.linspace(do_A_min, do_A_max, 10)
    EY_do_A = []
    for a in do_A:
        A_ = np.repeat(a, [N[1]])
        if args.fullgraph:
            Y_do_A = (fun_ay(A_, b=a_AY) + eY[:N[1]] + 2 * U[:N[1]] + a_WY * W[:N[1]]).round(3) if fun_ay is not None else get_y(U[:N[1]], W[:N[1]], A_, centres=centres_w, sig=sig_, coeffs=coeffs_)
        else:
            Y_do_A = (fun_ay(A_, b=a_AY) + eY[:N[1]] + 2 * U[:N[1]]).round(3) if fun_ay is not None else get_y(U[:N[1]], A_, centres=centres_no_w, sig=sig_, coeffs=coeffs_)
        eY_do_A = np.mean(Y_do_A)
        EY_do_A.append(eY_do_A)

    EY_do_A = np.array(EY_do_A)

    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/main_{}.npz'.format(args.sem)),
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



    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/do_A_{}.npz'.format(args.sem)),
             do_A = do_A,
             gt_EY_do_A = EY_do_A)

    # plotting
    D = pd.DataFrame([U[:300], A[:300], Y[:300], Z[:300], W[:300]]).T
    D.columns = ['U', 'A', 'Y','Z', 'W']
    O = pd.DataFrame([A[:300], Y[:300], Z[:300], W[:300]]).T
    O.columns = ['A', 'Y', 'Z', 'W']
    D_conU = pd.DataFrame([U_inst[:300], A_conU[:300], Y_conU[:300], Z_conU[:300], W_conU[:300]]).T
    D_conU.columns = ['U', 'A', 'Y', 'Z', 'W']
    D_doA = pd.DataFrame([do_A, EY_do_A]).T
    D_doA.columns = ['A', 'EY_do_A']

    ecorr_v = D.corr()
    ecorr_v.columns = ['U', 'A', 'Y', 'Z', 'W']
    ecorr_O = O.corr()
    ecorr_O.columns = ['A', 'Y', 'Z', 'W']
    ecorr_v_conU = D_conU.corr()
    ecorr_v_conU.columns = ['U', 'A', 'Y', 'Z', 'W']


    sem = args.sem
    if not os.path.exists(PATH+sem):
        os.mkdir(PATH + sem)
    for v in ['U', 'A', 'Y', 'Z', 'W']:
        sns.displot(D, x=v, label=v, kde=True), plt.savefig(PATH + sem + '/' + v + '_dist.png'), plt.close()

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(PATH + sem + '/' + 'full_pairwise.png'), plt.close()
    sns.pairplot(O), plt.savefig(PATH + sem + '/' + 'observed_pairwise.png'), plt.close()
    sns.pairplot(D_conU), plt.savefig(PATH + sem + '/' + 'fixed_U_pairwise.png'), plt.close()
    sns.pairplot(D_doA), plt.savefig(PATH + sem + '/' + 'ate_pairwise.png'), plt.close()

    sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_all.png'), plt.close()
    sns.heatmap(ecorr_O, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_observed.png'), plt.close()
    sns.heatmap(ecorr_v_conU, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_fixed_U.png'), plt.close()

    # generating conditional expectation
    print('expectation evaluation starts.')
    test_sample_sz = 1000
    w_sample_thresh = 20
    axz = np.vstack([np.linspace(do_A_min,do_A_max,1000), test_x, test_z])  # shape: 3 x 1000
    axzwy = np.vstack([rest_a, rest_x, rest_z, rest_w, rest_y])
    axzy_np, w_samples_out_np, y_samples_out_np = gen_eval_samples(test_sample_size=test_sample_sz,
                                                                   w_sample_thresh=w_sample_thresh,
                                                                   axz=axz,
                                                                   axzwy=axzwy)
    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/cond_exp_metric_{}.npz'.format(args.sem)),
             axzy=axzy_np,
             w_samples=w_samples_out_np,
             y_samples=y_samples_out_np)


if __name__ == '__main__':
    main(args)