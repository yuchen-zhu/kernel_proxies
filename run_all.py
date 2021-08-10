# import torch, add_path
import numpy as np
# from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
#     GMM, DeepIV, AGMM
# import os
# import tensorflow
from tabulate import tabulate
# from MMR_proxy.util import ROOT_PATH, load_data
# import random
# random.seed(527)
import argparse
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
from MMR_proxy.util import visualise_ATEs, ROOT_PATH

import os, sys
# import autograd.numpy as np
# from autograd import value_and_grad
# from scipy.optimize import minimize
# from joblib import Parallel, delayed
# import time
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
# from matplotlib import cm
from datetime import date
# import random

from MMR_proxy.rkhs_model_LMO_nystr_zoo_hparam_lmo_makemain import run_rkhs
from other_baselines_scripts.run_zoo_experiments_more_baselines import run_baselines

# al, bl = 1.7554502814125845, 100.0
# al, bl = 10.0599250311491, 100.0
al, bl = 1.6554502814125847, 100.0
kpv_result_path = os.path.join(os.path.dirname(__file__), '../results/kpv_res_trainsz7000_dataseed100.npz')
sname, train_sizes, data_seeds = 'sim_1d_no_x', [500, 1000, 3000, 7000], [100, 200, 300, 400, 500]
from_res = True

parser = argparse.ArgumentParser(description='run models')

parser.add_argument('--av-kernel', type=bool, default=False, help='use single bandwidth or average')
parser.add_argument('--sem', type=str, help='set which SEM to use data from')
parser.add_argument('--hparam', type=str, default='fixed', help='set which hparam method to use. options are cube, lmo or fixed')
parser.add_argument('--al', type=float, default=al, help='set the parameter for rkhs, al.')
parser.add_argument('--bl', type=float, default=bl, help='set the parameter for rkhs, bl.')
parser.add_argument('--_kpv_ab-result-path', type=str, default=kpv_result_path, help='set path to _kpv_ab results.')
parser.add_argument('--from-res', type=bool, default=from_res, help='set load results from saved results or train from scratch.')

args = parser.parse_args()


def split_true_baselines(ate_true_and_baselines):
    assert len(ate_true_and_baselines) > 0
    ate_true, ate_baselines = None, []

    for method_name, ate in ate_true_and_baselines:
        if method_name == 'gt':
            ate_true = ate
        else:
            ate_baselines.append((method_name, ate))

    assert ate_true is not None
    assert len(ate_baselines) > 0
    return ate_true, ate_baselines




# -0.2923312, -0.23064951, -0.16211458, -0.09079472, -0.02073781, 0.04420233, 0.10053497, 0.14528799, 0.17614129, 0.19153024, 0.19071272, 0.1737948, 0.14171244, 0.09616852, 0.03952717
#
# -0.34579602, -0.29746631, -0.24493731, -0.17859577, -0.09341105,  0.00403354, 0.09781003,  0.1714569,   0.21370793,  0.2174826,   0.17913798,  0.10424218, 0.01392498, -0.06021529, -0.09425195
#
# -0.03764629,-0.03273315,-0.02782002,-0.02290689,-0.01799375,-0.01308062,-0.00816748,-0.00325435, 0.00165878, 0.00657192, 0.01148505, 0.01639819, 0.02131132, 0.02622445, 0.03113759
#


def add_rkhs_performance(causal_mae_rkhs, err_in_expectation_rkhs, ate_rkhs, ate_true, our_method_name, sname, PATH, args):
    causal_std = np.std(np.abs(ate_rkhs.squeeze() - ate_true.squeeze()))
    causal_rel_err = np.mean(np.abs((ate_rkhs.squeeze() - ate_true.squeeze())/ate_true.squeeze()))
    performance_log_file = open(os.path.join(PATH, str(date.today()), args.sem,
                                             "method_performance_log_datasize{}_{}.txt".format(datasize,
                                                                                               sname)), "a")
    performance_log_file.write(
        "Method name: {}, expected err: {}, causal mae: {}, causal std: {}, causal rel err: {}\n".format(our_method_name,
                                                                                                     err_in_expectation_rkhs,
                                                                                                     causal_mae_rkhs,
                                                                                                     causal_std,
                                                                                                     causal_rel_err))
    performance_log_file.close()


def make_ate_comparison_plots(do_A, ate_true, ate_baselines, ate_mmr, ate_kpv, PATH, datasize, args):

    ate_comparison_pairwise(do_A=do_A, ate_true=ate_true, ate_baselines=ate_baselines, ate_mmr=ate_mmr, ate_kpv=ate_kpv, PATH=PATH, datasize=datasize, args=args)
    ate_all(do_A=do_A, ate_true=ate_true, ate_baselines=ate_baselines, ate_mmr=ate_mmr, ate_kpv=ate_kpv, PATH=PATH, datasize=datasize, args=args)
    causal_abs_err_all(do_A=do_A, ate_true=ate_true, ate_baselines=ate_baselines, ate_mmr=ate_mmr, ate_kpv=ate_kpv, PATH=PATH, datasize=datasize, args=args)


def causal_abs_err_all(do_A, ate_true, ate_baselines, ate_mmr, ate_kpv, PATH, datasize, args):
    plt.figure()
    for method_name, EY_do_A in ate_baselines:
        plt.plot(do_A, np.abs(EY_do_A.squeeze() - ate_true.squeeze()), label=method_name)
    # plt.plot(do_A, np.squeeze(ate_true), label='gt')
    plt.plot(do_A, np.abs(ate_mmr.squeeze() - ate_true.squeeze()), label='mmr-rkhs')
    plt.plot(do_A, np.abs(ate_kpv.squeeze() - ate_true.squeeze()), label='_kpv_ab')
    plt.xlabel('A'), plt.ylabel('|E[Y|do(A)] - est.E[Y|do(A)]|'), plt.legend()
    plt.savefig(
        os.path.join(PATH, str(date.today()), args.sem, 'causal_abs_err_allmethods_datasize{}'.format(datasize) + '.png'))
    plt.close()


def ate_comparison_pairwise(do_A, ate_true, ate_baselines, ate_mmr, ate_kpv, PATH, datasize, args):
    diff_true_mmr = np.abs(ate_true.squeeze() - ate_mmr.squeeze())
    diff_true_kpv = np.abs(ate_true.squeeze() - ate_kpv.squeeze())

    for method_name, ate_est in ate_baselines:
        diff_true_baseline = np.abs(ate_true.squeeze() - ate_est.squeeze())
        visualise_ATEs(diff_true_mmr, diff_true_baseline,
                       x_name='|true - mmr|',
                       y_name='|true - {}|'.format(method_name),
                       save_loc=os.path.join(PATH, str(date.today()), args.sem),
                       save_name='ate_comparison_mmr_{}'.format(method_name))

    for method_name, ate_est in ate_baselines:
        diff_true_baseline = np.abs(ate_true.squeeze() - ate_est.squeeze())
        visualise_ATEs(diff_true_kpv, diff_true_baseline,
                       x_name='|true - _kpv_ab|',
                       y_name='|true - {}|'.format(method_name),
                       save_loc=os.path.join(PATH, str(date.today()), args.sem),
                       save_name='ate_comparison_kpv_{}'.format(method_name))

    visualise_ATEs(diff_true_kpv, diff_true_mmr,
                   x_name='|true - _kpv_ab|',
                   y_name='|true - mmr|',
                   save_loc=os.path.join(PATH, str(date.today()), args.sem),
                   save_name='ate_comparison_kpv_mmr')




def ate_all(do_A, ate_true, ate_baselines, ate_mmr, ate_kpv, PATH, datasize, args):
    plt.figure()
    for method_name, EY_do_A in ate_baselines:
        plt.plot(do_A, np.squeeze(EY_do_A), label=method_name)
    plt.plot(do_A, np.squeeze(ate_true), label='gt')
    plt.plot(do_A, np.squeeze(ate_mmr), label='mmr')
    plt.plot(do_A, np.squeeze(ate_kpv), label='_kpv_ab')
    plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    plt.savefig(
        os.path.join(PATH, str(date.today()), args.sem, 'ate_allmethods_datasize{}'.format(datasize) + '.png'))
    plt.close()


def get_kpv(datasize, sname, ate_true, fp=None, kpv_func=None, args=None):
    """
    Args:
        fp: take in a string for a filepath. Must be a npz file which contains ate as key
    """
    if fp is not None:
        # file = np.load(fp, allow_pickle=True)
        # print(file)
        ate = np.load(fp, allow_pickle=True)['ate_est']
    else:
        assert kpv_func is not None
        ate = kpv_func(args=args, datasize=datasize, sname=sname)
    causal_mae = np.mean(np.abs(ate.squeeze() - ate_true.squeeze()))

    return causal_mae, ate


def run_mmr(args, datasize, sname):
    return run_rkhs(args=args, datasize=datasize, sname=sname)


def load_blines_from_result(args, fp):
    pass


def load_mmr_from_result(args, fp):
    pass


if __name__ == '__main__':

    PATH = ROOT_PATH + "/results/zoo/" + sname + "/"
    for train_sz in train_sizes:
        for data_seed in data_seeds:
            do_A = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['do_A']
            if args.from_res:
                ate_true_and_baselines = load_blines_from_result(args=args, fp=fp_blines_seed_train_size)
                ate_true, ate_baselines = split_true_baselines(ate_true_and_baselines)
                causal_mae_mmr, err_in_expectation_mmr, ate_mmr = load_mmr_from_result(args, fp=fp_mmr_seed_train_size)
            else:
                ate_true_and_baselines = run_baselines(args, datasize=datasize, sname=sname, data_seed=data_seed)
                ate_true, ate_baselines = split_true_baselines(ate_true_and_baselines)
                causal_mae_mmr, err_in_expectation_mmr, ate_mmr = run_mmr(args=args, datasize=datasize, sname=sname, data_seed=data_seed)

            causal_mae_kpv, ate_kpv = get_kpv(datasize=None, sname=None, fp=args.kpv_result_path, ate_true=ate_true)
        # average_performance

    add_rkhs_performance(causal_mae_rkhs=causal_mae_mmr, err_in_expectation_rkhs=err_in_expectation_mmr, ate_rkhs=ate_mmr, ate_true=ate_true, our_method_name='mmr-proxy', sname=sname, PATH=PATH, args=args)
    add_rkhs_performance(causal_mae_rkhs=causal_mae_kpv, err_in_expectation_rkhs=None, ate_rkhs=ate_kpv, ate_true=ate_true, our_method_name='_kpv_ab', sname=sname, PATH=PATH, args=args)

    # ate_true = np.load(ROOT_PATH + "/data/zoo/" + sname + '/do_A_{}.npz'.format(args.sem))['gt_EY_do_A']

    make_ate_comparison_plots(do_A=do_A, ate_true=ate_true, ate_baselines=ate_baselines, ate_mmr=ate_mmr, ate_kpv=ate_kpv, PATH=PATH, datasize=datasize, args=args)




