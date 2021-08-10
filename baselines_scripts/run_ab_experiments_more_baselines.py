import torch
import numpy as np
from baselines_scripts.baselines.all_baselines import Vanilla2SLS, Bases2SLS, KRidge
import os
import tensorflow
# from tabulate import tabulate
from MMR_proxy.util import ROOT_PATH, load_data, data_transform
# random.seed(527)
import argparse
import matplotlib.pyplot as plt
from MMR_proxy.util import visualise_ATEs
from datetime import date
import pickle


parser = argparse.ArgumentParser(description='run baselines')

parser.add_argument('--sem', type=str, help='set which SEM to use data from')
args = parser.parse_args()


def scale_all(train_A, train_Y, train_Z, train_W, test_A, test_Y, test_Z, test_W):
    A_scaled, A_scaler = data_transform(train_A)
    print('mean train Y', np.mean(train_Y))
    Y_scaled, Y_scaler = data_transform(train_Y)
    Z_scaled, Z_scaler = data_transform(train_Z)
    W_scaled, W_scaler = data_transform(train_W)

    test_A_scaled = A_scaler.transform(test_A)
    test_Y_scaled = Y_scaler.transform(test_Y)
    test_Z_scaled = Z_scaler.transform(test_Z)
    test_W_scaled = W_scaler.transform(test_W)

    return A_scaled, Y_scaled, Z_scaled, W_scaled, test_A_scaled, test_Y_scaled, test_Z_scaled, test_W_scaled, A_scaler, Y_scaler, Z_scaler, W_scaler



def eval_model(model, test, tag):
    if tag == "direct":
        y_pred_test = model.predict(test.a.reshape(-1,1))
        mse = float(((y_pred_test.squeeze() - test.y) ** 2).mean())
    elif tag == '2sls':
        y_pred_test = model.predict(np.stack([test.a, test.w], axis=-1))
        mse = float(((y_pred_test.squeeze() - test.y) ** 2).mean())
    elif tag == 'covar':
        y_pred_test = model.predict(test.a.reshape(-1,1), context=np.stack([test.z, test.w], axis=-1))
        mse = float(((y_pred_test.squeeze() - test.y) ** 2).mean())
    elif tag == 'covar_w':
        y_pred_test = model.predict(test.a.reshape(-1,1), context=test.w.reshape(-1,1))
        mse = float(((y_pred_test.squeeze() - test.y) ** 2).mean())
    else:
        raise ValueError('tag must be direct or 2sls or covar or covar_w.')
    return mse


def eval_ate_direct(model_direct, do_A, context, EY_do_A, Y_scaler):
    if context is not None:

        do_A_rep = np.repeat(do_A, [context.shape[0]], axis=-1).reshape(-1, 1)
        context_rep = np.tile(context, [do_A.shape[0], 1])
        EYhat_do_A_sc_out = model_direct.predict(do_A_rep, context_rep).reshape(-1, context.shape[0])
        EYhat_do_A_sc = np.mean(EYhat_do_A_sc_out, axis=-1).reshape(-1, 1)

        # EYhat_do_A_sc = []
        # for A in do_A:
        #     A = np.repeat(A, context.shape[0]).reshape(-1,1)
        #     context = context.reshape(-1,context.shape[-1])
        #     EYhat_do_A_sc_ = np.mean(model_direct.predict(A, context))
        #     EYhat_do_A_sc.append(EYhat_do_A_sc_)
        # EYhat_do_A_sc = np.array(EYhat_do_A_sc)
    else:
        EYhat_do_A_sc = model_direct.predict(do_A.reshape(-1,1))
    print('unscaled ATE: ', EYhat_do_A_sc)
    print('Y_scaler: ', Y_scaler.mean_, Y_scaler.scale_)
    EYhat_do_A = Y_scaler.inverse_transform(EYhat_do_A_sc)
    print('scaled ATE: ', EYhat_do_A)

    mae = np.mean(np.abs(EYhat_do_A.squeeze() - EY_do_A.squeeze()))
    std = np.std(np.abs(EYhat_do_A.squeeze() - EY_do_A.squeeze()))
    mean_rel_err = np.mean(np.abs((EYhat_do_A.squeeze() - EY_do_A.squeeze())/EY_do_A.squeeze()))
    return mae, EYhat_do_A, std, mean_rel_err


def get_causal_effect(model_2sls, do_A, w, Y_scaler):
    """
    :param net: FCNN object
    :param do_A: a numpy array of interventions, size = B_a
    :param w: a torch tensor of w samples, size = B_w
    :return: a numpy array of interventional parameters
    """
    # raise ValueError('have not tested get_causal_effect.')

    do_A_rep = np.repeat(do_A.reshape(-1,1), [w.shape[0]], axis=-1).reshape(-1, 1)
    w_rep = np.tile(w, [do_A.shape[0], 1])
    aw_rep = np.concatenate([do_A_rep, w_rep], axis=-1)
    h_out = model_2sls.predict(aw_rep)

    h_out_a_as_rows = h_out.reshape(-1, w.shape[0])
    ate_est = np.mean(h_out_a_as_rows, axis=-1).reshape(-1, 1)
    ate_est_orig_scale = Y_scaler.inverse_transform(ate_est)

    # EYhat_do_A = []
    # for a in do_A:
    #     a = np.repeat(a, [w.shape[0]]).reshape(-1,1)
    #     # a_tensor = torch.as_tensor(a).float()
    #     w = w.reshape(-1, 1)
    #     aw = np.concatenate([a, w], axis=-1)
    #     mean_h = np.mean(model_2sls.predict(aw)).reshape(-1, 1)
    #     EYhat_do_A.append(mean_h)
    #     print('a = {}, beta_a = {}'.format(np.mean(a), mean_h))

    return ate_est_orig_scale


def eval_ate_2sls(model_2sls, do_A, w, EY_do_A_gt, Y_scaler):
    EYhat_do_A = get_causal_effect(model_2sls=model_2sls, do_A=do_A, w=w, Y_scaler=Y_scaler)
    mae = np.mean(np.abs(EYhat_do_A.squeeze() - EY_do_A_gt.squeeze()))
    std = np.std(np.abs(EYhat_do_A.squeeze() - EY_do_A_gt.squeeze()))
    mean_rel_err = np.mean(np.abs((EYhat_do_A.squeeze() - EY_do_A_gt.squeeze())/EY_do_A_gt.squeeze()))
    return mae, EYhat_do_A, std, mean_rel_err


def plotting(EYhat_do_A, do_A, EY_do_A, PATH, method_name, train_sz, args, seed, data_seed):
    if not os.path.exists(os.path.join(PATH, args.sem+'_seed{}'.format(data_seed))):
        os.makedirs(os.path.join(PATH, args.sem+'_seed{}'.format(data_seed)), exist_ok=True)
    plt.figure()
    plt.plot(do_A, np.squeeze(EYhat_do_A), label='est')
    plt.plot(do_A, EY_do_A, label='gt')
    plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
    plt.savefig(
        os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed), 'causal_effect_estimates_{}_trainsz{}_seed{}'.format(method_name, train_sz, seed) + '.png'))
    plt.close()
    visualise_ATEs(EY_do_A, EYhat_do_A,
                   x_name='E[Y|do(A)] - gt',
                   y_name='beta_A',
                   save_loc=os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed)),
                   save_name='ate_{}_trainsz{}'.format(method_name, train_sz))


def save_model(model, save_path, test, tag, **kwargs):
    if tag == "direct":
        g_pred = model.predict(test.a.reshape(-1,1))
        np.savez(save_path, x=test.a, y=test.y, g_true=test.y, g_hat=g_pred.squeeze(), **kwargs)
    elif tag == '2sls':
        g_pred = model.predict(np.stack([test.a, test.w], axis=-1))
        np.savez(save_path, x=np.stack([test.a, test.w], axis=-1), y=test.y, g_true=test.y, g_hat=g_pred.squeeze(), **kwargs)
    elif tag == 'covar':
        g_pred = model.predict(test.a.reshape(-1,1), context=np.stack([test.z, test.w], axis=-1))
        np.savez(save_path, x=np.stack([test.a, test.z, test.w], axis=-1), y=test.y, g_true=test.y, g_hat=g_pred.squeeze(), **kwargs)
    elif tag == 'covar_w':
        g_pred = model.predict(test.a.reshape(-1,1), context=test.w.reshape(-1,1))
        np.savez(save_path, x=np.stack([test.a, test.w], axis=-1), y=test.y, g_true=test.y, g_hat=g_pred.squeeze(), **kwargs)
    else:
        raise ValueError('tag must be direct or 2sls or covar or covar_w.')


def run_experiment(scenario_name,mid,repid,trainsz, data_seed, num_reps=10, seed=527, training=False, args=None, PATH=None):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)

    # train, dev, test = load_data(ROOT_PATH+'/data/zoo/'+scenario_name+'_{}.npz'.format(trainsz))
    train, dev, test = load_data(ROOT_PATH + '/data/zoo/' + scenario_name + '/main_{}_seed{}.npz'.format(args.sem, data_seed))


    A_scaled, Y_scaled, Z_scaled, W_scaled, test_A_scaled, test_Y_scaled, test_Z_scaled, test_W_scaled, A_scaler, Y_scaler, Z_scaler, W_scaler = \
    scale_all(train_A=train.a[:trainsz], train_Y=train.y[:trainsz], train_Z=train.z[:trainsz], train_W=train.w[:trainsz],
              test_A=test.a[:test_size], test_Y=test.y[:test_size], test_Z=test.z[:test_size], test_W=test.w[:test_size])


    # data for direct methods
    Y_direct, X_direct = Y_scaled.reshape(-1, 1), A_scaled.reshape(-1, 1)
    Z_direct, W_direct = Z_scaled.reshape(Z_scaled.shape[0], -1), W_scaled.reshape(W_scaled.shape[0], -1)
    # test_Y, test_X = test.y.reshape(-1, 1), test.a.reshape(-1, 1)

    # data for 2sls methods
    X_train = np.concatenate((A_scaled.reshape(-1,1), W_scaled.reshape(W_scaled.shape[0], -1)), axis=-1)
    Z_train = np.concatenate((A_scaled.reshape(-1,1), Z_scaled.reshape(Z_scaled.shape[0], -1)), axis=-1)
    Y_2sls, X_2sls, Z_2sls = Y_scaled.reshape(-1, 1), X_train, Z_train

    do_A = np.load(ROOT_PATH + "/data/zoo/" + scenario_name + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['do_A']
    do_A = A_scaler.transform(do_A)
    EY_do_A_gt = np.load(ROOT_PATH + "/data/zoo/" + scenario_name + '/do_A_{}_seed{}.npz'.format(args.sem, data_seed))['gt_EY_do_A']

    means = []
    times = []
    EYhat_do_A_collect = [('gt', EY_do_A_gt)]

    # Not all methods are applicable in all scenarios
    methods_direct = []
    methods_2s = []
    methods_covar = []
    methods_covar_w = []
    methods_lvm = []
    methods_dfiv = []

    # baseline methods
    methods_direct += [("KernelRidge", KRidge())]
    methods_covar += [("KernelRidgeAdj", KRidge())]
    methods_covar_w += [("KernelRidge-W", KRidge())]
    methods_2s += [("LinearP2SLS", Vanilla2SLS())]
    methods_2s += [("Deaner18", Bases2SLS())]
    # methods_2s += [("NN2SLS", NN2SLS())]
    # methods_2s += [("NNP2SLS", NNP2SLS())]
    # methods_2s += [("GMM", GMM(g_model="2-layer", n_steps=20))]
    # methods_2s += [("Poly2SLS", Poly2SLS())]
    # methods_2s += [("AGMM", AGMM())]
    # methods_lvm += [("LVM", LVM())]
    # methods_dfiv += [("DFIV", DFIV())]

    # methods += [("Poly2SLS", Poly2SLS())]
    # methods += [("GMM", GMM(g_model="2-layer", n_steps=20))]
    # methods += [("AGMM", AGMM())]
    # methods_2s += [("DeepIV", DeepIV())]

    results_dict = {'gt': EY_do_A_gt}

    for rep in range(num_reps):
        print('rep = ', rep)
        if training:
            train_data_dict = {'covar': (X_direct, Y_direct, Z_direct, np.concatenate((Z_direct, W_direct), axis=-1)),
                               'covar_w': (X_direct, Y_direct, Z_direct, W_direct),
                               'direct': (X_direct, Y_direct, Z_direct, None),
                               '2sls': (X_2sls, Y_2sls, Z_2sls, None),
                               'lvm': (None, None, None, None)}

            # if rep < repid:
            #     continue
            # elif rep > repid:
            #     break
            # else:
            #     pass
            for methods_list, tag in [(methods_direct, 'direct'), (methods_2s, '2sls'), (methods_covar, 'covar'), (methods_covar_w, 'covar_w')]:
                X, Y, Z, context = train_data_dict[tag]
                # for method_name, method in methods_list[mid:mid+1]:
                for method_name, method in methods_list:
                    if method_name not in results_dict:
                        results_dict[method_name] = {'causal_mae':{}, 'causal_std': {}, 'causal_rel_err': {}, 'causal_est': {}}
                    print("Running " + method_name + " " + str(rep))
                    file_name = "%s_%d_%d.npz" % (method_name, rep, X.shape[0])
                    save_path = os.path.join(PATH, file_name)
                    # print(type(X), type(Y))
                    model, time = method.fit(X, Y, Z, context)
                    # np.save(PATH+"%s_%d_%d_time.npy" % (method_name, rep, train.a.shape[0]), time)

                    EYhat_do_A = None
                    if tag == 'direct' or tag == 'covar' or tag =='covar_w':
                        print('method: {}, tag: {}'.format(method_name, tag))
                        causal_mae, EYhat_do_A, causal_std, causal_rel_err = eval_ate_direct(model_direct=model, do_A=do_A, context=context, EY_do_A=EY_do_A_gt, Y_scaler=Y_scaler)
                    elif tag == '2sls':
                        print('method: {}, tag: {}'.format(method_name, tag))
                        causal_mae, EYhat_do_A, causal_std, causal_rel_err = eval_ate_2sls(model_2sls=model, do_A=do_A, w=W_scaled[:50], EY_do_A_gt=EY_do_A_gt, Y_scaler=Y_scaler)
                    if rep == 0:
                        EYhat_do_A_collect.append((method_name, EYhat_do_A))
                    # save_model(model, save_path, test, tag=tag, causal_mae=causal_mae, ate_est=EYhat_do_A, causal_std=causal_std, causal_rel_err=causal_rel_err)
                    results_dict[method_name]['causal_mae'][rep] = causal_mae
                    results_dict[method_name]['causal_std'][rep] = causal_std
                    results_dict[method_name]['causal_rel_err'][rep] = causal_rel_err
                    results_dict[method_name]['causal_est'][rep] = EYhat_do_A

                    # test_mse = eval_model(model, test, tag=tag)
                    # print('BASELINE COLLECT: ', EYhat_do_A_collect)
                    model_type_name = type(model).__name__
                    print("Causal MAE of %s : %f" % (method_name, causal_mae))
                    plotting(EYhat_do_A=EYhat_do_A, do_A=do_A, EY_do_A=EY_do_A_gt, PATH=PATH, method_name=method_name, train_sz=X.shape[0], args=args, seed=seed, data_seed=data_seed)

                    # performance log file
                    performance_log_file = open(os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed), "method_performance_log_blines_trainsz{}_{}.txt".format(trainsz, scenario_name)), "a")
                    performance_log_file.write(
                        "Method name: {}, rep: {}, causal mae: {}, causal std: {}, causal rel err: {}\n".format(method_name, rep, causal_mae, causal_std, causal_rel_err))
                    performance_log_file.close()

            # make ATE comparison plot for baselines.
            if rep == 0:
                plt.figure()
                for method_name, EY_do_A in EYhat_do_A_collect:
                    plt.plot(do_A, np.squeeze(EY_do_A), label=method_name)
                plt.xlabel('A'), plt.ylabel('EYdoA'), plt.legend()
                plt.savefig(
                    os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed), 'causal_effect_estimates_allmethods_trainsz_1seed{}'.format(X.shape[0]) + '.png'))
                plt.close()

                print('EYhat_do_A_collect 1: ', EYhat_do_A_collect)

            with open(os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed), 'results_baselines_trainsz{}_seed{}_reps{}.p'.format(X_direct.shape[0], seed, num_reps)), 'wb') as handle:
                pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            means2 = []
            times2 = []
            for methods_list in [methods_direct, methods_2s]:
                for method_name, method in methods_list:
                    # print("Running " + method_name +" " + str(rep))
                    file_name = "%s_%d_%d.npz" % (method_name, rep,trainsz)
                    save_path = os.path.join(PATH, file_name)
                    if os.path.exists(save_path):
                        res = np.load(save_path)
                        mse = float(((res['g_hat'] - res['g_true']) ** 2).mean())
        #                print('mse: {}'.format(mse))
                        means2 += [mse]
                    else:
                        print(save_path, ' not exists')
                    time_path = PATH+"%s_%d_%d_time.npy" % (method_name, rep,train.x.shape[0])
                    if os.path.exists(time_path):
                        res = np.load(time_path)
                        times2 += [res]
                    else:
                        print(time_path, ' not exists')
            if len(means2) == len(methods):
                means += [means2]
            if len(times2) == len(methods):
                times += [times2]
    # print('EYhat_do_A_collect 2: ', EYhat_do_A_collect)
    return means,times, EYhat_do_A_collect


def run_baselines(args, trainsz, sname, data_seed):
    """
    Return structure:
        ate_estimates_baselines: list of tuples. [('method_name', ate_estimates_for_method: list),...]
    """
    # result PATH
    PATH = os.path.join(ROOT_PATH, "../results/zoo/", sname + "/")
    os.makedirs(PATH, exist_ok=True)
    os.makedirs(os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed)), exist_ok=True)
    print("created directory {}".format(os.path.join(PATH, str(date.today()), args.sem+'_seed{}'.format(data_seed))))
    performance_log_file = open(os.path.join(PATH, str(date.today()),
                                    args.sem+'_seed{}'.format(data_seed), "method_performance_log_trainsz{}_{}.txt".format(trainsz, sname)), "w")
    performance_log_file.close()

    ate_estimates_baselines = None
    for sid in range(1):
        for repid in range(1):
            means, times, ate_estimates_baselines = run_experiment(sname,0,repid,trainsz, num_reps=1, training=True, args=args, PATH=PATH, data_seed=data_seed)

    return ate_estimates_baselines


if __name__ == "__main__":
    test_size, dev_size = 500, 0
    data_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # data_seeds = np.arange(10)
    sname, train_sizes = 'sim_1d_no_x', [1500]

    for data_seed in data_seeds:
        for trainsz in train_sizes:
            ate_est_baselines = run_baselines(args=args, trainsz=trainsz, sname=sname, data_seed=data_seed)






    # scenarios = np.array(["sim_1d_no_x"])
    # for trainsz in [200,2000]:
    #     for sid in range(1):
    #         # result PATH
    #         PATH = ROOT_PATH + "/results/zoo/" + scenarios[sid] + "/"
    #         os.makedirs(PATH, exist_ok=True)
    #         for mid in range(1):
    #             for repid in range(10):
    #                 run_experiment(scenarios[sid],mid,repid,trainsz, training=True, PATH=PATH, data_seed=data_seed)
    #
    #     rows = []
    #     for s in scenarios:
    #         means,times, ate_estimates_baselines = run_experiment(s,0,0,trainsz,training=False, data_seed=data_seed)
    #         mean = np.mean(means,axis=0)
    #         std = np.std(means,axis=0)
    #         rows += [["{:.3f} $pm$ {:.3f}".format(mean[i],std[i]) for i in range(len(mean))]]
    #         print('time: ',np.mean(times,axis=0),np.std(times,axis=0))
    #
    #     methods = np.array(["DirectNN","Vanilla2SLS","Poly2SLS","GMM","AGMM","DeepIV"])[:,None]
    #     print(methods,np.array(rows).T)
    #     rows = np.hstack((methods,np.array(rows).T))
    #     print('Tabulate Table:')
    #     print(tabulate(np.vstack((np.append([""],scenarios),rows)), headers='firstrow',tablefmt='latex'))

