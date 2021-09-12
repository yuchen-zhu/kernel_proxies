import numpy as np
import argparse
from PMMR.util import visualise_ATEs, ROOT_PATH, split_into_bins
import os
import matplotlib.pyplot as plt
from datetime import date
import pickle
import seaborn as sns
import pandas as pd


datestr = '2021-04-02'
seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
train_sizes = [1500]

bins = np.array([-0.5,0.5,1.5,2.5])  # edu
labels = np.array([0,1,2])  # edu

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sim_1d_no_x')  # Path to results directory.

parser = argparse.ArgumentParser(description='postprocess results')

parser.add_argument('--sem', default=None, type=str)
parser.add_argument('--discrete', default=True, type=bool, help='indicate discrete actions')
parser.add_argument('--date', default=None, type=str, help='set date: <YYYY-MM-DD> OR "preset"')
parser.add_argument('--seeds', default=seeds, type=list, help='specify seeds of data')
parser.add_argument('--labels', default=None, type=np.array, help='set do_A labels for discrete actions')
parser.add_argument('--bins', default=None, type=np.array, help='set bins for discrete actions')
parser.add_argument('--offset-bool', default=False, type=bool, help='set offset on/off')


args = parser.parse_args()

if args.discrete:
    args.labels = labels
    args.bins = bins

if args.date == 'preset':
    args.date = datestr

if args.date is None:
    args.date = str(date.today())


def calculate_off_set(labels, preds):
    n = len(labels)
    return 1 / n * (np.sum(labels) - np.sum(preds))


def write_latex(kpv_collect_sz, mmr_collect_sz, blines_collect_sz, train_sz):
    """
    for each method, ie _kpv_ab, mmr, or a baseline, the file structure is:

    - method_abs
    --- method_abs_over_seeds_av_each_a
    --- method_abs_over_seeds_av
    --- method_abs_over_seeds_std

    - method_change_abs
    --- method_change_abs_over_seeds_av_each_a
    --- method_change_abs_over_seeds_av
    --- method_change_abs_over_seeds_std

    - method_rel
    --- method_rel_over_seeds_av_each_a
    --- method_rel_over_seeds_av
    --- method_rel_over_seeds_std

    - method_change_rel
    --- method_change_rel_over_seeds_av_each_a
    --- method_change_rel_over_seeds_av
    --- method_change_rel_over_seeds_std
    """

    """
        summary_file = open(
        os.path.join(PATH, str(date.today()), args.sem+'_seed'+str(data_seed), "summary_trainsz{}_nystrom_hparam{}.txt".format(train_sz, args.hparam)),
        "a")
    summary_file.write("al: {}, bl: {}, causal_mae_: {}, causal_std: {}, causal_rel_err: {}\n"
                       "expected_error : {}, mmr_v_supp: {}, mmr_v: {}, mmr_u_supp: {}, mmr_u: {}\n"
                       "mse: {}, mse_supp: {}, lmo: {}\n"
                       "causal_est: {}\n".format(params[0], params[1], causal_effect_mean_abs_err, causal_std,
                                                 causal_rel_err, err_in_expectation, mmr_v_supp, mmr_v, mmr_u_supp, mmr_u, mse_standard, mse_supp, lmo, EYhat_do_A))
    summary_file.close()"""

    dbl = '\\'.replace('\\', '\\\\')
    latex_file = open(
        os.path.join(PATH, 'latex-trainsz{}-{}.txt'.format(train_sz, args.sem)), 'w'
    )
    latex_file.write("\hline\n")
    latex_file.write("  & c-MAE & offset-MAE & deriv-MAE & c-RMAE & offset-RMAE & deriv-RMAE" + dbl + "\n")
    latex_file.close()
    if mmr_collect_sz is not None:
        string = '{} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} {}\n'.format(
            'PMMR',
            mmr_collect_sz[0][1], "$\pm$",
            mmr_collect_sz[0][-2],
            mmr_collect_sz[1][1], "$\pm$",
            mmr_collect_sz[1][-2],
            mmr_collect_sz[2][1], "$\pm$",
            mmr_collect_sz[2][-2],
            mmr_collect_sz[3][1], "$\pm$",
            mmr_collect_sz[3][-2],
            mmr_collect_sz[4][1], "$\pm$",
            mmr_collect_sz[4][-2],
            mmr_collect_sz[5][1], "$\pm$",
            mmr_collect_sz[5][-2],
            dbl)
        latex_file = open(os.path.join(PATH, 'latex-trainsz{}-{}.txt'.format(train_sz, args.sem)), 'a')
        latex_file.write("\hline\n")
        latex_file.write(string)
        latex_file.close()

    if kpv_collect_sz is not None:
        string = '{} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} {}\n'.format(
            'KPV',
            kpv_collect_sz[0][1], "$\pm$",
            kpv_collect_sz[0][-2],
            kpv_collect_sz[1][1], "$\pm$",
            kpv_collect_sz[1][-2],
            kpv_collect_sz[2][1], "$\pm$",
            kpv_collect_sz[2][-2],
            kpv_collect_sz[3][1], "$\pm$",
            kpv_collect_sz[3][-2],
            kpv_collect_sz[4][1], "$\pm$",
            kpv_collect_sz[4][-2],
            kpv_collect_sz[5][1], "$\pm$",
            kpv_collect_sz[5][-2],
            dbl)
        latex_file = open(os.path.join(PATH, 'latex-trainsz{}-{}.txt'.format(train_sz, args.sem)), 'a')
        latex_file.write("\hline\n")
        latex_file.write(string)
        latex_file.close()

    if blines_collect_sz is not None:
        for bline in blines_collect_sz.keys():
            string = '{} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} & {:.3g} {} {:.3g} {}\n'.format(
                bline,
                blines_collect_sz[bline][0][1], "$\pm$", blines_collect_sz[bline][0][-2],
                blines_collect_sz[bline][1][1], "$\pm$", blines_collect_sz[bline][1][-2],
                blines_collect_sz[bline][2][1], "$\pm$", blines_collect_sz[bline][2][-2],
                blines_collect_sz[bline][3][1], "$\pm$", blines_collect_sz[bline][3][-2],
                blines_collect_sz[bline][2][1], "$\pm$", blines_collect_sz[bline][2][-2],
                blines_collect_sz[bline][3][1], "$\pm$", blines_collect_sz[bline][3][-2],
                dbl)
            latex_file = open(os.path.join(PATH, 'latex-trainsz{}-{}.txt'.format(train_sz, args.sem)), 'a')
            latex_file.write("\hline\n")
            latex_file.write(string)
            latex_file.close()

    latex_file = open(os.path.join(PATH, 'latex-trainsz{}-{}.txt'.format(train_sz, args.sem)), 'a')
    latex_file.write("\hline\n")
    latex_file.close()


def process_discrete_res(ate, do_A):
    do_A_uniq = np.unique(do_A)
    do_A_uniq.sort()
    print('uniq: ', do_A_uniq)
    ate_uniq = []
    for a in do_A_uniq:
        inds = (do_A == a)
        ate_a = ate[inds].mean()
        ate_uniq.append(ate_a)
    ate_uniq = np.array(ate_uniq)

    return do_A_uniq, ate_uniq


def process_kpv(do_A, args):
    kpv_collect = {}
    for train_sz in train_sizes:
        kpv_res_over_seeds = []
        kpv_abs_over_seeds = []
        kpv_rel_over_seeds = []
        kpv_change_abs_over_seeds = []
        kpv_change_rel_over_seeds = []
        kpv_deriv_abs_over_seeds = []
        kpv_deriv_rel_over_seeds = []

        for seed in args.seeds:
            # get gt
            gt = (np.load(os.path.join(os.path.dirname(__file__),
                                       'data/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seed)),
                          allow_pickle=True)['gt_EY_do_A']).squeeze()

            # get KPV estimates
            kpv_res = pickle.load(open(os.path.join(PATH, args.date, 'kpv_{}'.format(args.sem),
                                                    'Ewh_results_{}.p'.format(seed)), 'rb'))[
                '{}_{}'.format(seed, train_sz)]['Ew_Haw'].astype(float).values.squeeze()
            # kpv_res = pickle.load(open(os.path.join(PATH, args.date, 'kpv_{}'.format(args.sem),
            #                                         'summary_results_AG_seed{}_scaled_size{}.p'.format(seed, train_sz)), 'rb'))[
            #           train_sz]['Ew_Haw'].astype(float).values.squeeze()

            if args.discrete:
                _, gt = process_discrete_res(ate=gt, do_A=do_A)
                _, kpv_res = process_discrete_res(ate=kpv_res, do_A=do_A)
            # print('***** kpv: ', kpv_res)

            gt_change = gt
            # print('***** gt change: ', gt_change)

            gt_deriv = gt[1:] - gt[:-1]
            # print('***** gt deriv: ', gt_deriv)

            kpv_offset = calculate_off_set(labels=gt, preds=kpv_res)
            # print('kpv_res: '.format(kpv_res))
            kpv_res_change = kpv_res + kpv_offset
            # print('***** kpv_result: ', kpv_res)
            # print('***** kpv_change: ', kpv_res_change)

            kpv_res_deriv = kpv_res[1:] - kpv_res[:-1]
            # print('***** _kpv_ab res deriv: ', kpv_res_deriv)

            kpv_abs_err = np.abs(kpv_res - gt)
            kpv_abs_over_seeds.append(kpv_abs_err)
            kpv_rel_err = np.abs((kpv_res - gt) / gt)
            kpv_rel_over_seeds.append(kpv_rel_err)

            kpv_change_abs_err = np.abs(kpv_res_change - gt_change)
            kpv_change_abs_over_seeds.append(kpv_change_abs_err)
            kpv_change_rel_err = np.abs((kpv_res_change[1:] - gt_change[1:]) / gt_change[1:])
            kpv_change_rel_over_seeds.append(kpv_change_rel_err)

            # print('***** kpv_change_abs_err: ', kpv_change_abs_err)

            kpv_deriv_abs_err = np.abs(kpv_res_deriv - gt_deriv)
            kpv_deriv_abs_over_seeds.append(kpv_deriv_abs_err)
            kpv_deriv_rel_err = np.abs((kpv_res_deriv[1:] - gt_deriv[1:]) / gt_deriv[1:])
            kpv_deriv_rel_over_seeds.append(kpv_deriv_rel_err)

            # print('***** kpv_deriv_abs_err: ', kpv_deriv_abs_err)

            kpv_res_over_seeds.append(kpv_res)

        # process _kpv_ab
        kpv_abs_over_seeds = np.array(kpv_abs_over_seeds)
        kpv_abs_over_seeds_av_each_a = np.mean(kpv_abs_over_seeds, axis=0)
        kpv_abs_over_seeds_av = np.mean(kpv_abs_over_seeds)
        kpv_abs_over_seeds_std = np.std(kpv_abs_over_seeds)
        kpv_abs = (kpv_abs_over_seeds_av_each_a, kpv_abs_over_seeds_av, kpv_abs_over_seeds_std, kpv_abs_over_seeds)

        kpv_change_abs_over_seeds = np.array(kpv_change_abs_over_seeds)
        kpv_change_abs_over_seeds_av_each_a = np.mean(kpv_change_abs_over_seeds, axis=0)
        kpv_change_abs_over_seeds_av = np.mean(kpv_change_abs_over_seeds)
        kpv_change_abs_over_seeds_std = np.std(kpv_change_abs_over_seeds)
        kpv_change_abs = (
        kpv_change_abs_over_seeds_av_each_a, kpv_change_abs_over_seeds_av, kpv_change_abs_over_seeds_std,
        kpv_change_abs_over_seeds)

        kpv_deriv_abs_over_seeds = np.array(kpv_deriv_abs_over_seeds)
        kpv_deriv_abs_over_seeds_av_each_a = np.mean(kpv_deriv_abs_over_seeds, axis=0)
        kpv_deriv_abs_over_seeds_av = np.mean(kpv_deriv_abs_over_seeds)
        kpv_deriv_abs_over_seeds_std = np.std(kpv_deriv_abs_over_seeds)
        kpv_deriv_abs = (kpv_deriv_abs_over_seeds_av_each_a, kpv_deriv_abs_over_seeds_av, kpv_deriv_abs_over_seeds_std,
                         kpv_deriv_abs_over_seeds)

        kpv_rel_over_seeds = np.array(kpv_rel_over_seeds)
        kpv_rel_over_seeds_av_each_a = np.mean(kpv_rel_over_seeds, axis=0)
        kpv_rel_over_seeds_av = np.mean(kpv_rel_over_seeds)
        kpv_rel_over_seeds_std = np.std(kpv_rel_over_seeds)
        kpv_rel = (kpv_rel_over_seeds_av_each_a, kpv_rel_over_seeds_av, kpv_rel_over_seeds_std, kpv_rel_over_seeds)

        kpv_change_rel_over_seeds = np.array(kpv_change_rel_over_seeds)
        kpv_change_rel_over_seeds_av_each_a = np.mean(kpv_change_rel_over_seeds, axis=0)
        kpv_change_rel_over_seeds_av = np.mean(kpv_change_rel_over_seeds)
        kpv_change_rel_over_seeds_std = np.std(kpv_change_rel_over_seeds)
        kpv_change_rel = (
        kpv_change_rel_over_seeds_av_each_a, kpv_change_rel_over_seeds_av, kpv_change_rel_over_seeds_std,
        kpv_change_rel_over_seeds)

        kpv_deriv_rel_over_seeds = np.array(kpv_deriv_rel_over_seeds)
        kpv_deriv_rel_over_seeds_av_each_a = np.mean(kpv_deriv_rel_over_seeds, axis=0)
        kpv_deriv_rel_over_seeds_av = np.mean(kpv_deriv_rel_over_seeds)
        kpv_deriv_rel_over_seeds_std = np.std(kpv_deriv_rel_over_seeds)
        kpv_deriv_rel = (kpv_deriv_rel_over_seeds_av_each_a, kpv_deriv_rel_over_seeds_av, kpv_deriv_rel_over_seeds_std,
                         kpv_deriv_rel_over_seeds)

        kpv_res_over_seeds = np.array(kpv_res_over_seeds)
        kpv_res_over_seeds_av_each_a = np.mean(kpv_res_over_seeds, axis=0)
        kpv_res_over_seeds_std_each_a = np.std(kpv_res_over_seeds, axis=0)
        kpv_res_ = (kpv_res_over_seeds_av_each_a, kpv_res_over_seeds_std_each_a, kpv_res_over_seeds)

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'KPV_RES_OVER SEEDS_TRAINSZ{}: '.format(train_sz), kpv_res_over_seeds)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'KPV_res_AV_TRAINSZ{}:'.format(train_sz), kpv_res_over_seeds_av_each_a)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'THE STANDARD DEV_TRAINSZ{}: '.format(train_sz), kpv_res_over_seeds_std_each_a)

        kpv_collect[train_sz] = (
        kpv_abs, kpv_change_abs, kpv_deriv_abs, kpv_rel, kpv_change_rel, kpv_deriv_rel, kpv_res_)
        print(kpv_collect)
    return kpv_collect


def process_mmr(do_A, args):
    mmr_collect = {}
    for train_sz in train_sizes:
        mmr_res_over_seeds = []
        mmr_abs_over_seeds = []
        mmr_rel_over_seeds = []
        mmr_change_abs_over_seeds = []
        mmr_change_rel_over_seeds = []
        mmr_deriv_abs_over_seeds = []
        mmr_deriv_rel_over_seeds = []

        for seed in args.seeds:
            # get gt
            gt = (np.load(os.path.join(os.path.dirname(__file__),
                                       'data/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seed)),
                          allow_pickle=True)['gt_EY_do_A']).squeeze()

            # get mmr estimates
            # mmr_res = (np.load(os.path.join(PATH, args.date, args.sem+'_seed{}'.format(seed),
            #                                    'mmr_res_trainsz{}.npz'.format(train_sz)))['ate_est']).squeeze()
            mmr_res = (np.load(os.path.join(PATH, args.date, args.sem + '_seed{}'.format(seed),
                                            'mmr_res_trainsz{}.npz'.format(train_sz, args.offset_bool)))[
                'ate_est']).squeeze()

            # print('mmr_res: '.format(mmr_res))
            if args.discrete:
                _, gt = process_discrete_res(ate=gt, do_A=do_A)
                _, mmr_res = process_discrete_res(ate=mmr_res, do_A=do_A)

            # print('***** gt: ', gt)
            gt_change = gt
            # print('***** gt change: ', gt_change)

            gt_deriv = gt[1:] - gt[:-1]
            # print('***** gt deriv: ', gt_deriv)

            mmr_off_set = calculate_off_set(labels=gt, preds=mmr_res)
            mmr_res_change = mmr_res + mmr_off_set
            # print('***** mmr_result: ', mmr_res)
            # print('***** mmr_change: ', mmr_res_change)

            mmr_res_deriv = mmr_res[1:] - mmr_res[:-1]
            # print('***** mmr res deriv: ', mmr_res_deriv)

            mmr_abs_err = np.abs(mmr_res - gt)
            mmr_abs_over_seeds.append(mmr_abs_err)
            mmr_rel_err = np.abs((mmr_res - gt) / gt)
            mmr_rel_over_seeds.append(mmr_rel_err)

            mmr_change_abs_err = np.abs(mmr_res_change - gt_change)
            mmr_change_abs_over_seeds.append(mmr_change_abs_err)
            mmr_change_rel_err = np.abs((mmr_res_change[1:] - gt_change[1:]) / gt_change[1:])
            mmr_change_rel_over_seeds.append(mmr_change_rel_err)

            # print('***** mmr_change_abs_err: ', mmr_change_abs_err)

            mmr_deriv_abs_err = np.abs(mmr_res_deriv - gt_deriv)
            mmr_deriv_abs_over_seeds.append(mmr_deriv_abs_err)
            mmr_deriv_rel_err = np.abs((mmr_res_deriv[1:] - gt_deriv[1:]) / gt_deriv[1:])
            mmr_deriv_rel_over_seeds.append(mmr_deriv_rel_err)

            # print('***** mmr_deriv_abs_err: ', mmr_deriv_abs_err)

            mmr_res_over_seeds.append(mmr_res)

        # process mmr
        mmr_abs_over_seeds = np.array(mmr_abs_over_seeds)
        mmr_abs_over_seeds_av_each_a = np.mean(mmr_abs_over_seeds, axis=0)
        mmr_abs_over_seeds_av = np.mean(mmr_abs_over_seeds)
        mmr_abs_over_seeds_std = np.std(mmr_abs_over_seeds)
        # mmr_abs_over_seeds_std_each_a = np.std(mmr_abs_over_seeds, axis=0)
        mmr_abs = (mmr_abs_over_seeds_av_each_a, mmr_abs_over_seeds_av, mmr_abs_over_seeds_std, mmr_abs_over_seeds)

        mmr_change_abs_over_seeds = np.array(mmr_change_abs_over_seeds)
        mmr_change_abs_over_seeds_av_each_a = np.mean(mmr_change_abs_over_seeds, axis=0)
        mmr_change_abs_over_seeds_av = np.mean(mmr_change_abs_over_seeds)
        mmr_change_abs_over_seeds_std = np.std(mmr_change_abs_over_seeds)
        mmr_change_abs = (
        mmr_change_abs_over_seeds_av_each_a, mmr_change_abs_over_seeds_av, mmr_change_abs_over_seeds_std,
        mmr_change_abs_over_seeds)

        mmr_deriv_abs_over_seeds = np.array(mmr_deriv_abs_over_seeds)
        mmr_deriv_abs_over_seeds_av_each_a = np.mean(mmr_deriv_abs_over_seeds, axis=0)
        mmr_deriv_abs_over_seeds_av = np.mean(mmr_deriv_abs_over_seeds)
        mmr_deriv_abs_over_seeds_std = np.std(mmr_deriv_abs_over_seeds)
        mmr_deriv_abs = (mmr_deriv_abs_over_seeds_av_each_a, mmr_deriv_abs_over_seeds_av, mmr_deriv_abs_over_seeds_std,
                         mmr_deriv_abs_over_seeds)

        # print('!!!!#################### mmr_deriv_abs_over_seeds_av: ', mmr_deriv_abs_over_seeds_av)
        # print('!!!!#################### mmr_deriv_abs_over_seeds_std: ', mmr_deriv_abs_over_seeds_std)

        # print('############## TRAIN SIZE = {} ###################'.format(train_sz))
        # print('-------------------MMR----------------------------')
        # print('-------------------ABS EACH A---------------------')
        # print(mmr_abs[0])
        # print('-------------------MAE----------------------------')
        # print(mmr_abs[1])
        # print('-------------------STD----------------------------')
        # print(mmr_abs[2])

        mmr_rel_over_seeds = np.array(mmr_rel_over_seeds)
        mmr_rel_over_seeds_av_each_a = np.mean(mmr_rel_over_seeds, axis=0)
        mmr_rel_over_seeds_av = np.mean(mmr_rel_over_seeds)
        mmr_rel_over_seeds_std = np.std(mmr_rel_over_seeds)
        mmr_rel = (mmr_rel_over_seeds_av_each_a, mmr_rel_over_seeds_av, mmr_rel_over_seeds_std, mmr_rel_over_seeds)

        mmr_change_rel_over_seeds = np.array(mmr_change_rel_over_seeds)
        mmr_change_rel_over_seeds_av_each_a = np.mean(mmr_change_rel_over_seeds, axis=0)
        mmr_change_rel_over_seeds_av = np.mean(mmr_change_rel_over_seeds)
        mmr_change_rel_over_seeds_std = np.std(mmr_change_rel_over_seeds)
        mmr_change_rel = (
        mmr_change_rel_over_seeds_av_each_a, mmr_change_rel_over_seeds_av, mmr_change_rel_over_seeds_std,
        mmr_change_rel_over_seeds)

        mmr_deriv_rel_over_seeds = np.array(mmr_deriv_rel_over_seeds)
        mmr_deriv_rel_over_seeds_av_each_a = np.mean(mmr_deriv_rel_over_seeds, axis=0)
        mmr_deriv_rel_over_seeds_av = np.mean(mmr_deriv_rel_over_seeds)
        mmr_deriv_rel_over_seeds_std = np.std(mmr_deriv_rel_over_seeds)
        mmr_deriv_rel = (mmr_deriv_rel_over_seeds_av_each_a, mmr_deriv_rel_over_seeds_av, mmr_deriv_rel_over_seeds_std,
                         mmr_deriv_rel_over_seeds)

        mmr_res_over_seeds = np.array(mmr_res_over_seeds)
        mmr_res_over_seeds_av_each_a = np.mean(mmr_res_over_seeds, axis=0)
        mmr_res_over_seeds_std_each_a = np.std(mmr_res_over_seeds, axis=0)
        mmr_res_ = (mmr_res_over_seeds_av_each_a, mmr_res_over_seeds_std_each_a, mmr_res_over_seeds)

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'MMR_RES_OVER SEEDS_TRAINSZ{}: '.format(train_sz), mmr_res_over_seeds)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'MMR_res_AV_TRAINSZ{}:'.format(train_sz), mmr_res_over_seeds_av_each_a)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        #       'THE STANDARD DEV_TRAINSZ{}: '.format(train_sz), mmr_res_over_seeds_std_each_a)

        # print('-------------------REL EACH A---------------------')
        # print(mmr_rel[0])
        # print('-------------------MEAN REL ERR-------------------')
        # print(mmr_rel[1])
        # print('-------------------REL STD------------------------')
        # print(mmr_rel[2])

        mmr_collect[train_sz] = (
        mmr_abs, mmr_change_abs, mmr_deriv_abs, mmr_rel, mmr_change_rel, mmr_deriv_rel, mmr_res_)
        # print(mmr_collect[1000][0][1])
        # raise ValueError
    return mmr_collect


def process_blines(do_A, args):
    # get baselines estimates

    keys = pickle.load(open(os.path.join(PATH, args.date, args.sem + '_seed{}'.format(seeds[0]),
                                         'results_baselines_trainsz{}_seed527_reps1.p'.format(train_sizes[0])),
                            'rb')).keys()

    blines_collect = {}
    for train_sz in train_sizes:
        blines_collect[train_sz] = {}

        blines_res_over_seeds = {}
        blines_abs_over_seeds = {}
        blines_rel_over_seeds = {}
        blines_change_abs_over_seeds = {}
        blines_change_rel_over_seeds = {}
        blines_deriv_abs_over_seeds = {}
        blines_deriv_rel_over_seeds = {}

        for bline in keys:
            if bline == 'gt':
                continue
            blines_res_over_seeds[bline] = []
            blines_abs_over_seeds[bline] = []
            blines_rel_over_seeds[bline] = []
            blines_change_abs_over_seeds[bline] = []
            blines_change_rel_over_seeds[bline] = []
            blines_deriv_abs_over_seeds[bline] = []
            blines_deriv_rel_over_seeds[bline] = []

            for seed in args.seeds:
                # get gt
                gt = (np.load(os.path.join(os.path.dirname(__file__),
                                           'data/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seed)),
                              allow_pickle=True)['gt_EY_do_A']).squeeze()
                print('***** gt: ', gt)

                # get baselines estimates
                with open(os.path.join(PATH, args.date, args.sem + '_seed{}'.format(seed),
                                       'results_baselines_trainsz{}_seed527_reps1.p'.format(train_sz)),
                          'rb') as bline_handle:
                    blines_res = pickle.load(bline_handle)

                bline_res = (blines_res[bline]['causal_est'][0]).squeeze()

                if args.discrete:
                    _, gt = process_discrete_res(ate=gt, do_A=do_A)
                    _, bline_res = process_discrete_res(ate=bline_res, do_A=do_A)

                gt_change = gt
                # print('***** gt change: ', gt_change)

                gt_deriv = gt[1:] - gt[:-1]
                # print('***** gt deriv: ', gt_deriv)

                bline_off_set = calculate_off_set(labels=gt, preds=bline_res)
                bline_res_change = bline_res + bline_off_set

                bline_res_deriv = bline_res[1:] - bline_res[:-1]

                bline_abs_err = np.abs(bline_res - gt)
                bline_change_abs_err = np.abs(bline_res_change - gt_change)
                blines_abs_over_seeds[bline].append(bline_abs_err)
                blines_change_abs_over_seeds[bline].append(bline_change_abs_err)
                bline_rel_err = np.abs((bline_res - gt) / gt)
                bline_change_rel_err = np.abs((bline_res_change[1:] - gt_change[1:]) / gt_change[1:])
                blines_rel_over_seeds[bline].append(bline_rel_err)
                blines_change_rel_over_seeds[bline].append(bline_change_rel_err)

                bline_deriv_abs_err = np.abs(bline_res_deriv - gt_deriv)
                blines_deriv_abs_over_seeds[bline].append(bline_deriv_abs_err)
                bline_deriv_rel_err = np.abs((bline_res_deriv[1:] - gt_deriv[1:]) / gt_deriv[1:])
                blines_deriv_rel_over_seeds[bline].append(bline_deriv_rel_err)

                # print('***** bline_deriv_abs_err: ', bline_deriv_abs_err)

                blines_res_over_seeds[bline].append(bline_res)

        # process blines
        blines = blines_abs_over_seeds.keys()
        for bline in blines:
            blines_collect[train_sz][bline] = []
            bline_abs_over_seeds = np.array(blines_abs_over_seeds[bline])
            bline_abs_over_seeds_av_each_a = np.mean(bline_abs_over_seeds, axis=0)
            bline_abs_over_seeds_av = np.mean(bline_abs_over_seeds)
            bline_abs_over_seeds_std = np.std(bline_abs_over_seeds)
            bline_abs = (
            bline_abs_over_seeds_av_each_a, bline_abs_over_seeds_av, bline_abs_over_seeds_std, bline_abs_over_seeds)

            bline_change_abs_over_seeds = np.array(blines_change_abs_over_seeds[bline])
            bline_change_abs_over_seeds_av_each_a = np.mean(bline_change_abs_over_seeds, axis=0)
            bline_change_abs_over_seeds_av = np.mean(bline_change_abs_over_seeds)
            bline_change_abs_over_seeds_std = np.std(bline_change_abs_over_seeds)
            bline_change_abs = (
            bline_change_abs_over_seeds_av_each_a, bline_change_abs_over_seeds_av, bline_change_abs_over_seeds_std,
            bline_change_abs_over_seeds)

            bline_deriv_abs_over_seeds = np.array(blines_deriv_abs_over_seeds[bline])
            bline_deriv_abs_over_seeds_av_each_a = np.mean(bline_deriv_abs_over_seeds, axis=0)
            bline_deriv_abs_over_seeds_av = np.mean(bline_deriv_abs_over_seeds)
            bline_deriv_abs_over_seeds_std = np.std(bline_deriv_abs_over_seeds)
            bline_deriv_abs = (
            bline_deriv_abs_over_seeds_av_each_a, bline_deriv_abs_over_seeds_av, bline_deriv_abs_over_seeds_std,
            bline_deriv_abs_over_seeds)

            bline_rel_over_seeds = np.array(blines_rel_over_seeds[bline])
            bline_rel_over_seeds_av_each_a = np.mean(bline_rel_over_seeds, axis=0)
            bline_rel_over_seeds_av = np.mean(bline_rel_over_seeds)
            bline_rel_over_seeds_std = np.std(bline_rel_over_seeds)
            bline_rel = (
            bline_rel_over_seeds_av_each_a, bline_rel_over_seeds_av, bline_rel_over_seeds_std, bline_rel_over_seeds)

            bline_change_rel_over_seeds = np.array(blines_change_rel_over_seeds[bline])
            bline_change_rel_over_seeds_av_each_a = np.mean(bline_change_rel_over_seeds, axis=0)
            bline_change_rel_over_seeds_av = np.mean(bline_change_rel_over_seeds)
            bline_change_rel_over_seeds_std = np.std(bline_change_rel_over_seeds)
            bline_change_rel = (
            bline_change_rel_over_seeds_av_each_a, bline_change_rel_over_seeds_av, bline_change_rel_over_seeds_std,
            bline_change_rel_over_seeds)
            # print('*************** bline change', bline_change_rel)
            # print('*************** bline change', bline_change_rel_over_seeds)

            bline_deriv_rel_over_seeds = np.array(blines_deriv_rel_over_seeds[bline])
            bline_deriv_rel_over_seeds_av_each_a = np.mean(bline_deriv_rel_over_seeds, axis=0)
            bline_deriv_rel_over_seeds_av = np.mean(bline_deriv_rel_over_seeds)
            bline_deriv_rel_over_seeds_std = np.std(bline_deriv_rel_over_seeds)
            bline_deriv_rel = (
            bline_deriv_rel_over_seeds_av_each_a, bline_deriv_rel_over_seeds_av, bline_deriv_rel_over_seeds_std,
            bline_deriv_rel_over_seeds)

            bline_res_over_seeds = np.array(blines_res_over_seeds[bline])
            bline_res_over_seeds_av_each_a = np.mean(bline_res_over_seeds, axis=0)
            bline_res_over_seeds_std_each_a = np.std(bline_res_over_seeds, axis=0)
            bline_res_ = (bline_res_over_seeds_av_each_a, bline_res_over_seeds_std_each_a, bline_res_over_seeds)

            blines_collect[train_sz][bline] = (
            bline_abs, bline_change_abs, bline_deriv_abs, bline_rel, bline_change_rel, bline_deriv_rel, bline_res_)
            print(train_sz, bline, blines_collect[train_sz][bline])
    return blines_collect


def make_res_df(**res_dict):
    df_dict = {'method': [], 'action': [], 'ae': [], 'log_ae': []}
    for method in res_dict.keys():
        for sd in range(len(res_dict[method])):
            for action, ae in enumerate(res_dict[method][sd]):
                df_dict['method'].append(method)
                df_dict['action'].append(action)
                df_dict['ae'].append(ae)
                df_dict['log_ae'].append(np.log(ae))

    df = pd.DataFrame(df_dict)
    # print(df.tail(70))
    # raise ValueError

    return df


def make_boxplots(save_name, legend_loc, df, args):
    sns.set_style("whitegrid")
    sns.boxplot(y='log_ae', x='action',
                data=df,
                palette="colorblind",
                hue='method')
    plt.legend(loc=legend_loc, prop={'size': 8})
    plt.savefig(os.path.join(PATH, 'plots', save_name + '.png'), bbox_inches='tight')


def plot_curves_w_intevals(save_name, X, x_name, y_name, legend_loc, legend_name, y_lim, args, trainsize,
                           **curves_dict):
    # a bank of 10 colours:
    # kernelridge tchetgen20 deaner18 kernelridge-adj kernelridge-w pmmr kpv
    # colours = ['crimson', 'purple', 'pink', 'orange', 'gold', 'black', 'green', 'blue', 'grey', 'red']
    if args.sem == 'std':
        colours = ['#F2DCBB', '#BBBBBB', '#8E7F7F', '#D8AC9C', '#9DDFD3', '#D44000', '#FF7A00', 'blue', 'grey', 'red']
    elif args.sem == 'ab':
        colours = ['black', '#F2DCBB', '#8E7F7F', '#9DDFD3', '#D44000', '#FF7A00', 'blue', 'red']
        # ab: kernelridge bases2psls kernelridge-W pmmr kpv
    else:
        colours = ['#F2DCBB', '#BBBBBB', '#8E7F7F', '#D8AC9C', '#9DDFD3', '#D44000', '#FF7A00', 'blue', 'grey', 'red']

    BIGGER_SIZE = 14
    sns.set_style("whitegrid")
    plt.clf()
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig, ax = plt.subplots()
    colour_counter = 0
    for method in curves_dict.keys():
        print('*********Method: ', method)
        y_mean = curves_dict[method][0]
        y_ci = curves_dict[method][1]
        if method == r'$\bf{KPV}$':
            # print('y_ci, KPV: ', y_ci)
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=3, color=colours[colour_counter], zorder=10)
            # markerline, _, _ = plt.stem(X, y_mean, linefmt=colours[colour_counter], use_line_collection=True)
            # markerline.set_markerfacecolor(colours[colour_counter])
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor=colours[colour_counter])
            colour_counter += 1
        elif method == r'$\bf{PMMR}$':
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=3, color=colours[colour_counter], zorder=10)
            # markerline, _, _ = plt.stem(X, y_mean, linefmt=colours[colour_counter], use_line_collection=True)
            # markerline.set_markerfacecolor(colours[colour_counter])
            # plt.stem(X, y_mean, linefmt=colours[colour_counter])
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor=colours[colour_counter])
            # print('y_ci, PMMR: ', y_ci)
            colour_counter += 1
        elif method == 'KernelRidge-W':
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=2, color=colours[colour_counter], zorder=10)
            # markerline, _, _ = plt.stem(X, y_mean, linefmt=colours[colour_counter], use_line_collection=True)
            # markerline.set_markerfacecolor(colours[colour_counter])
            # plt.stem(X, y_mean, linefmt=colours[colour_counter])
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor=colours[colour_counter])
            # print('y_ci, PMMR: ', y_ci)
            colour_counter += 1
        elif method == 'BasesP2SLS':
            ax.plot(X, y_mean, label='Deaner18', alpha=0.8, linewidth=2, color=colours[colour_counter])
            # markerline, _, _ = plt.stem(X, y_mean, linefmt=colours[colour_counter], use_line_collection=True)
            # markerline.set_markerfacecolor(colours[colour_counter])
            # plt.stem(X, y_mean, linefmt=colours[colour_counter])
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor=colours[colour_counter])
            # print('y_ci, PMMR: ', y_ci)
            colour_counter += 1
        elif method == 'LinearP2SLS':
            colour_counter += 1
            continue
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=1.5, color=colours[colour_counter], zorder=10)
            # ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor=colours[colour_counter])
            # print('y_ci, PMMR: ', y_ci)
            colour_counter += 1
        elif method == 'gt':
            if args.discrete:
                ax.plot(X, y_mean, 'gx', color=colours[colour_counter], label=method, zorder=10)
                colour_counter += 1
            else:
                ax.plot(X, y_mean, 'x', color=colours[colour_counter], label='groundtruth', alpha=0.8, linewidth=2,
                        zorder=10)
                colour_counter += 1
        else:
            print(method)
            if method == 'KernelRidge':
                pass
            # elif method == 'Deaner18':
            #     continue
            ax.plot(X, y_mean, color=colours[colour_counter], label=method, alpha=0.8, linewidth=2)
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, color=colours[colour_counter], alpha=0.2)
            colour_counter += 1
    plt.ylim(y_lim)
    if 'edu_IR' in args.sem:
        plt.xlim([-0.1, 2.1])
        plt.ylim([4.5, 5.0])
    elif 'edu_IM' in args.sem:
        plt.xlim([-0.1, 2.1])
        plt.ylim([4.3, 4.85])

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('train size = {}'.format(trainsize))
    if args.sem == 'ab':
        plt.legend(loc=3, prop={'size': 8})
        plt.ylim([-0.4, 0.4])
    plt.savefig(os.path.join(PATH, 'plots', save_name + '.png'), bbox_inches='tight')
    plt.close()

    # generate legend separately
    sns.set_style("white")
    plt.clf()
    # fig, ax = plt.subplots()
    handles, colour_counter, names = [], 0, []
    for method in curves_dict.keys():
        if method == r'$\bf{KPV}$':
            # print('y_ci, KPV: ', y_ci)
            handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
            names.append(method)
            colour_counter += 1
        elif method == r'$\bf{PMMR}$':
            handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
            names.append(method)
            # print('y_ci, PMMR: ', y_ci)
            colour_counter += 1
        elif method == 'LinearP2SLS':
            colour_counter += 1
            continue
            handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
            names.append('Tchetgen-Tchetgen20')
            colour_counter += 1
        elif method == 'gt':
            if args.discrete:
                handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
                names.append(method)
                colour_counter += 1
            else:
                handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
                names.append(method)
                print('HERE!!!!')
                colour_counter += 1
        else:
            print(method)
            if method == 'KernelRidge':
                pass
            # elif method == 'Deaner18':
            #     continue
            handles.append(plt.plot([], [], marker='s', color=colours[colour_counter], ls='none')[0])
            names.append(method)
            colour_counter += 1

    legend = plt.legend(handles, names, loc=legend_loc, prop={'size': 8}, framealpha=1, frameon=True)
    legend_saveloc = os.path.join(PATH, 'plots', legend_name)

    def export_legend(legend, filename=legend_saveloc, expand=[-5, -5, 5, 5]):
        fig = legend.figure
        renderer = fig.canvas.get_renderer()
        fig.draw(renderer=renderer)
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend)


# (save_name, X, x_name, y_name, legend_loc, legend_name, y_lim, args, trainsize, **curves_dict)

def _plot_curves_w_intevals(save_name, X, x_name, y_name, legend_loc, legend_name, y_lim, args, trainsize,
                            **curves_dict):
    BIGGER_SIZE = 14
    sns.set_style("whitegrid")
    plt.clf()
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig, ax = plt.subplots()
    for method in curves_dict.keys():
        y_mean = curves_dict[method][0]
        y_ci = curves_dict[method][1]
        if method == 'KernelRidge-W':
            print('y_ci, KPV: ', y_ci)
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=2, color='red', zorder=10)
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2, facecolor='red')
        elif method == 'KernelRidge':
            print('y_ci, KPV: ', y_ci)
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=2, zorder=10)
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2)
        elif method == 'gt':
            if args.discrete:
                ax.plot(X, y_mean, 'gx', label=method, zorder=10)
            else:
                ax.plot(X, y_mean, 'x', color='black', label='groundtruth', alpha=0.8, linewidth=2, zorder=10)
        elif method == r'$\bf{KPV}$':
            print('y_ci, KPV: ', y_ci)
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=2, zorder=10)
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2)
        elif method == r'$\bf{PMMR}$':
            print('y_ci, KPV: ', y_ci)
            ax.plot(X, y_mean, label=method, alpha=0.8, linewidth=2, zorder=10)
            ax.fill_between(X, y_mean + y_ci, y_mean - y_ci, alpha=0.2)
        else:
            continue
    plt.ylim([-0.5, 0.5])
    if 'edu_IR' in args.sem:
        plt.xlim([-0.1, 2.1])
        plt.ylim([4.5, 5.0])
    elif 'edu_IM' in args.sem:
        plt.xlim([-0.1, 2.1])
        plt.ylim([4.3, 4.85])

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc=legend_loc, prop={'size': 8})
    plt.savefig(os.path.join(PATH, 'plots', save_name + '.png'), bbox_inches='tight')
    plt.close()


def main():

    if not os.path.exists(os.path.join(PATH, 'plots')):
        os.makedirs(os.path.join(PATH, 'plots'), exist_ok=True)
        print('making {}'.format(os.path.join(PATH, 'plots')))

    do_A = (np.load(os.path.join(os.path.dirname(__file__),
                                 'data/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seeds[0])),
                    allow_pickle=True)['do_A']).squeeze()

    print('do_A: ', do_A)

    gt = (np.load(os.path.join(os.path.dirname(__file__),
                               'data/sim_1d_no_x/do_A_{}_seed{}.npz'.format(args.sem, seeds[0])),
                  allow_pickle=True)['gt_EY_do_A']).squeeze()
    if args.discrete:
        do_A = split_into_bins(arr=do_A, labels=args.labels, bins=args.bins)
        do_A.sort()
        print('do_A: ', do_A)
        _, gt = process_discrete_res(ate=gt, do_A=do_A)

    kpv_collect = process_kpv(do_A=do_A, args=args)
    mmr_collect = process_mmr(do_A=do_A, args=args)
    blines_collect = process_blines(do_A=do_A, args=args)
    for train_sz in train_sizes:
        if not args.discrete:
            do_A_to_plot = {'gt': [gt, np.zeros((gt.shape[0],))]}
            abs_over_A_to_plot = {}
            # process blines
            blines = blines_collect[train_sz].keys()
            for bline in blines:
                if (bline == 'Vanilla2SLS') or (bline == 'DeepIV'):
                    continue
                do_A_to_plot[bline] = []
                abs_over_A_to_plot[bline] = []
                do_A_to_plot[bline].append(blines_collect[train_sz][bline][-1][0])
                abs_over_A_to_plot[bline].append(blines_collect[train_sz][bline][0][0])
                ci = 1.96 * blines_collect[train_sz][bline][-1][1] / blines_collect[train_sz][bline][-1][0]
                do_A_to_plot[bline].append(blines_collect[train_sz][bline][-1][1])
                abs_over_A_to_plot[bline].append(np.std(blines_collect[train_sz][bline][0][-1], axis=0))
            pmmr_name = r'$\bf{PMMR}$'
            do_A_to_plot[pmmr_name] = []
            abs_over_A_to_plot[pmmr_name] = []
            do_A_to_plot[pmmr_name].append(mmr_collect[train_sz][-1][0])
            abs_over_A_to_plot[pmmr_name].append(mmr_collect[train_sz][0][0])
            ci_mmr = 1.96 * mmr_collect[train_sz][-1][1] / mmr_collect[train_sz][-1][0]
            do_A_to_plot[pmmr_name].append(mmr_collect[train_sz][-1][1])
            abs_over_A_to_plot[pmmr_name].append(np.std(mmr_collect[train_sz][0][-1], axis=0))

            kpv_name = r'$\bf{KPV}$'
            do_A_to_plot[kpv_name] = []
            abs_over_A_to_plot[kpv_name] = []
            do_A_to_plot[kpv_name].append(kpv_collect[train_sz][-1][0])
            abs_over_A_to_plot[kpv_name].append(kpv_collect[train_sz][0][0])
            ci_kpv = 1.96 * kpv_collect[train_sz][-1][1] / kpv_collect[train_sz][-1][0]
            do_A_to_plot[kpv_name].append(kpv_collect[train_sz][-1][1])
            abs_over_A_to_plot[kpv_name].append(np.std(kpv_collect[train_sz][0][-1], axis=0))

            plot_curves_w_intevals(save_name='do_A_plot_{}_all_m_trainsz{}'.format(args.sem, train_sz), X=do_A,
                                   x_name='A', y_name='E[Y|do(A)]', legend_loc=3,
                                   legend_name='legend_do_A_{}'.format(args.sem), y_lim=[-1., 1.5], trainsize=train_sz,
                                   args=args, **do_A_to_plot)
            plot_curves_w_intevals(save_name='abs_over_A_plot_{}_all_m_trainsz{}'.format(args.sem, train_sz), X=do_A,
                                   x_name='A', y_name='E[Y|do(A)] - E_wh(A,W)', legend_loc=3,
                                   legend_name='legend_abs_over_A_{}'.format(args.sem), y_lim=[0., 2.],
                                   trainsize=train_sz, args=args, **abs_over_A_to_plot)

        else:
            do_A_to_plot = {}
            # process blines
            blines = blines_collect[train_sz].keys()
            for bline in blines:
                if (bline == 'Vanilla2SLS') or (bline == 'DeepIV'):
                    continue
                do_A_to_plot[bline] = blines_collect[train_sz][bline][0][-1]
            pmmr_name = r'$\bf{PMMR}$'
            do_A_to_plot[pmmr_name] = mmr_collect[train_sz][0][-1]

            kpv_name = r'$\bf{KPV}$'
            do_A_to_plot[kpv_name] = kpv_collect[train_sz][0][-1]
            bp_df = make_res_df(**do_A_to_plot)
            make_boxplots(save_name='do_A_boxplot_{}_all_m_trainsz{}'.format(args.sem, train_sz), legend_loc=3,
                          df=bp_df, args=args)

        write_latex(kpv_collect_sz=kpv_collect[train_sz], mmr_collect_sz=mmr_collect[train_sz],
                    blines_collect_sz=blines_collect[train_sz],
                    train_sz=train_sz)


if __name__ == '__main__':
    main()