import os,sys,torch,add_path
import torch.autograd as ag
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scenarios.abstract_scenario import AbstractScenario
from early_stopping import EarlyStopping
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy
from joblib import Parallel, delayed
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH,_sqdist,FCNN, CNN, bundle_az_aw, visualise_ATEs
import argparse
import pandas as pd
from datetime import date
# from matplotlib.pyplot import hexbin

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np

parser = argparse.ArgumentParser(description='parses argument for nn ')
parser.add_argument('--sem', type=str, help='set which SEM to use data from')
parser.add_argument('--reps', type=int, default=3, help='set the number of reps to train for')
args = parser.parse_args()



def run_experiment_nn(sname,datasize,indices=[],seed=527,training=True, args=None, folder=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    elif len(indices)==3:
        lr_id, dw_id,W_id = indices
    # load data
    
    train, dev, test = load_data(ROOT_PATH+"/data/zoo/"+sname+'/main_{}.npz'.format(args.sem), Torch=True)
    Y = torch.cat((train.y, dev.y), dim=0).float()
    AZ_train, AW_train = bundle_az_aw(train.a, train.z, train.w, Torch=True)
    AZ_test, AW_test = bundle_az_aw(test.a, test.z, test.w, Torch=True)
    AZ_dev, AW_dev = bundle_az_aw(dev.a, dev.z, test.w, Torch=True)

    X, Z = torch.cat((AW_train,AW_dev),dim=0).float(), torch.cat((AZ_train, AZ_dev),dim=0).float()
    test_X, test_Y = AW_test.float(),test.y.float()  # TODO: is test.g just test.y?


    n_train = train.a.shape[0]
    # training settings
    n_epochs = 1000  # 1000
    batch_size = 1000 if train.a.shape[0]>1000 else train.a.shape[0]

    # load expectation eval data
    axzy = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['axzy']
    w_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['w_samples']
    y_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric_{}.npz'.format(args.sem))['y_samples']
    y_axz = axzy[:, -1]
    ax = axzy[:, :2]

    # kernel
    kernel = Kernel('rbf', Torch=True)
    a = get_median_inter_mnist(AZ_train)
    a = torch.tensor(a).float()
    # training loop
    lrs = [2e-3, 1e-3, 2e-4,1e-4] # [3,5]
    # decay_weights = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6] # [11,5]
    decay_weights = [1e-12,1e-10,1e-8,1e-6]

    def my_loss(output, target, indices, K):
        d = output - target
        if indices is None:
            W = K
        else:
            W = K[indices[:, None], indices]
            # print((kernel(Z[indices],None,a,1)+kernel(Z[indices],None,a/10,1)+kernel(Z[indices],None,a*10,1))/3-W)
        loss = d.T @ W @ d / (d.shape[0]) ** 2
        return loss[0, 0]

    def conditional_expected_loss(net, ax, w_samples, y_samples, y_axz, x_on):
        if not x_on:
            ax = ax[:, 0:1]
        num_reps = w_samples.shape[1]
        assert len(ax.shape) == 2
        assert ax.shape[1] < 3
        assert ax.shape[0] == w_samples.shape[0]
        print('number of points: ', w_samples.shape[0])

        ax_rep = np.repeat(ax, [num_reps], axis=0)
        assert ax_rep.shape[0] == (w_samples.shape[1] * ax.shape[0])

        w_samples_flat = w_samples.flatten().reshape(-1,1)
        nn_inp_np = np.concatenate([ax_rep, w_samples_flat], axis=-1)
        # print('nn_inp shape: ', nn_inp_np.shape)
        nn_inp = torch.as_tensor(nn_inp_np).float()
        nn_out = net(nn_inp).detach().cpu().numpy()
        nn_out = nn_out.reshape([-1, w_samples.shape[1]])
        y_axz_recon = np.mean(nn_out, axis=1)
        assert y_axz_recon.shape[0] == y_axz.shape[0]
        mean_abs_error = np.mean(np.abs(y_axz - y_axz_recon))

        # for debugging compute the mse between y samples and h
        y_samples_flat = y_samples.flatten()
        mse = np.mean((y_samples_flat - nn_out.flatten())**2)

        return mean_abs_error, mse, y_axz_recon  # mean abs error is E_{A,X,Z~uniform}[E[|Y-h||A,X,Z]], mse is E_{A,X,Z~uniform}[E[(y-h)^2|A,X,Z]], y_axz_recon = E[h|A,X,Z] for the uniformly sampled (a,x,z)'s.

    def fit(x,y,z,dev_x,dev_y,dev_z,a,lr,decay_weight, ax, y_axz, w_samples, n_epochs=n_epochs):
        if 'mnist' in sname:
            train_K = torch.eye(x.shape[0])
        else:
            train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
        if dev_z is not None:
            if 'mnist' in sname:
                dev_K = torch.eye(x.shape[0])
            else:
                dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3
        n_data = x.shape[0]
        net = FCNN(x.shape[1]) if sname not in ['mnist_x', 'mnist_xz'] else CNN()
        es = EarlyStopping(patience=100)  # 10 for small
        optimizer = optim.Adam(list(net.parameters()), lr=lr, weight_decay=decay_weight)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

        test_errs, dev_errs, exp_errs, mse_s = [], [], [], []

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_data)

            for i in range(0, n_data, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # training loop
                def closure():
                    optimizer.zero_grad()
                    pred_y = net(batch_x)
                    loss = my_loss(pred_y, batch_y, indices, train_K)
                    loss.backward()
                    return loss

                optimizer.step(closure)  # Does the update

            if epoch % 5 == 0 and epoch >= 50 and dev_x is not None:  # 5, 10 for small # 5,50 for large
                g_pred = net(test_X)  # TODO: is it supposed to be test_X here? A: yes I think so.
                test_err = ((g_pred-test_Y)**2).mean() # TODO: why isn't this loss reweighted? A: because it is supposed to measure the agreement between prediction and labels.
                if epoch == 50 and 'mnist' in sname:
                    if z.shape[1] > 100:
                        train_K = np.load(ROOT_PATH+'/mnist_precomp/{}_train_K0.npy'.format(sname))
                        train_K = (torch.exp(-train_K/a**2/2)+torch.exp(-train_K/a**2*50)+torch.exp(-train_K/a**2/200))/3
                        dev_K = np.load(ROOT_PATH+'/mnist_precomp/{}_dev_K0.npy'.format(sname))
                        dev_K = (torch.exp(-dev_K/a**2/2)+torch.exp(-dev_K/a**2*50)+torch.exp(-dev_K/a**2/200))/3
                    else:
                        train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
                        dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3

                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                err_in_expectation, mse, _ = conditional_expected_loss(net=net, ax=ax, w_samples=w_samples, y_samples=y_samples, y_axz=y_axz, x_on=False)
                print('test: ', test_err, 'dev: ', dev_err, 'err_in_expectation: ', err_in_expectation, 'mse (alternative): ', mse)
                test_errs.append(test_err)
                dev_errs.append(dev_err)
                exp_errs.append(err_in_expectation)
                mse_s.append(mse)

                scheduler.step(test_err)
                if es.step(dev_err):
                    break
            losses = {'test': test_errs, 'dev': dev_errs, 'exp': exp_errs, 'mse_': mse_s}

        return es.best, epoch, net, losses

    def get_causal_effect(net, do_A, w):
        """
        :param net: FCNN object
        :param do_A: a numpy array of interventions, size = B_a
        :param w: a torch tensor of w samples, size = B_w
        :return: a numpy array of interventional parameters
        """
        net.eval()
        # raise ValueError('have not tested get_causal_effect.')
        EYhat_do_A = []
        for a in do_A:
            a = np.repeat(a, [w.shape[0]]).reshape(-1,1)
            a_tensor = torch.as_tensor(a).float()
            w = w.reshape(-1,1).float()
            aw = torch.cat([a_tensor,w], dim=-1)
            aw_tensor = torch.tensor(aw)
            mean_h = torch.mean(net(aw_tensor)).reshape(-1, 1)
            EYhat_do_A.append(mean_h)
            print('a = {}, beta_a = {}'.format(np.mean(a), mean_h))
        return torch.cat(EYhat_do_A).detach().cpu().numpy()


    if training is True:
        print('training')
        test_err_av = 0
        av_causal_effect_mean_abs_err = 0
        for rep in range(args.reps):
            print('*******REP: {}'.format(rep))
            # save_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep, lr_id, dw_id, AW_train.shape[0]))
            # if os.path.exists(save_path):
            #    continue
            lr, dw = lrs[lr_id], decay_weights[dw_id]
            print('lr, dw', lr, dw)
            t0 = time.time()
            err, _, net, losses = fit(X[:n_train], Y[:n_train], Z[:n_train], X[n_train:], Y[n_train:], Z[n_train:], a, lr, dw,
                              ax=ax, y_axz=y_axz, w_samples=w_samples)
            t1 = time.time()-t0
            # np.save(folder+'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep, lr_id, dw_id, AW_train.shape[0]), t1)
            g_pred = net(test_X).detach().numpy()
            test_err = ((g_pred-test_Y.numpy())**2).mean()
            # np.savez(save_path, err=err.detach().numpy(), lr=lr, dw=dw, g_pred=g_pred, test_err=test_err)

            test_err_av = (test_err_av * rep + test_err) / (rep + 1)

            # make E[Y|A,Z] vs E[h|A,Z]
            _, _, y_axz_recon = conditional_expected_loss(net=net, ax=ax, w_samples=w_samples, y_samples=y_samples, x_on=False, y_axz=y_axz)

            visualise_ATEs(y_axz, y_axz_recon,
                           x_name='E[Y|A,Z]',
                           y_name='E[h|A,Z]',
                           save_loc=folder,
                           save_name='expectation_y_recon_{}_{}_{}_{}.png'.format(rep, lr_id, dw_id, AW_train.shape[0]))
            # TODO: rename this function to visualise_reconstructions as it can be applied to any (scalar) reconstruction.

            # make 3d plot for h
            a_max, a_min = X[:, 0].max().numpy(), X[:, 0].min().numpy()
            a_linspace = np.linspace(a_min, a_max, 10)
            w_max, w_min = X[:, 0].max().numpy(), X[:, 0].min().numpy()
            w_linspace = np.linspace(w_min, w_max, 10)
            aa, ww = np.meshgrid(a_linspace,w_linspace)
            aa, ww = aa.flatten(), ww.flatten()
            aw_inp = torch.as_tensor(np.stack([aa, ww], axis=-1)).float()
            h_out = net(aw_inp).detach().cpu().numpy()
            print('aa, ww, h out: ', aw_inp, h_out)
            h_out_grid = h_out.reshape(10,10)

            plot_df = pd.DataFrame([aw_inp[:,0], aw_inp[:,1], h_out.squeeze()]).T
            plot_df.columns = ['A', 'W', 'h(A,W)']

            # Make the plot
            # fig = plt.figure()
            # hexbin(plot_df['A'], plot_df['W'], C=plot_df['h(A,W)'])
            # plt.show()
            # Plot the surface.
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # surf = ax.plot_surface(aa.reshape(10,10), ww.reshape(10,10), h_out_grid, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
            # plt.savefig(os.path.join(folder, args.sem, 'nn', '_{}_{}_{}_{}'.format(rep, lr_id, dw_id, AW_train.shape[0]) + '.png'))

            # make loss curves
            for (name, ylabel) in [('test', 'test av ||y - h||^2'), ('dev', 'R_V'), ('exp', 'E[y-h|a,z,x]'), ('mse_', 'mse_alternative_sim')]:
                errs = losses[name]
                stps = [50 + i * 5 for i in range(len(errs))]
                plt.figure()
                plt.plot(stps, errs)
                plt.xlabel('epoch')
                plt.ylabel(ylabel)
                plt.savefig(os.path.join(folder, name + '_{}_{}_{}_{}'.format(rep, lr_id, dw_id, AW_train.shape[0]) + '.png'))
                plt.close()

            # do causal effect estimates
            do_A = np.load(ROOT_PATH+"/data/zoo/"+sname+'/do_A_{}.npz'.format(args.sem))['do_A']
            EY_do_A_gt = np.load(ROOT_PATH+"/data/zoo/"+sname+'/do_A_{}.npz'.format(args.sem))['gt_EY_do_A']
            w_sample = train.w
            EYhat_do_A = get_causal_effect(net, do_A=do_A, w=w_sample)
            plt.figure()
            plt.plot([i+1 for i in range(20)], EYhat_do_A, label='est')
            plt.plot([i+1 for i in range(20)], EY_do_A_gt, label='gt')
            plt.xlabel('A')
            plt.ylabel('EYdoA-est')
            plt.savefig(
                os.path.join(folder, 'causal_effect_estimates_{}_{}_{}'.format(lr_id, dw_id, AW_train.shape[0]) + '.png'))
            plt.close()
            print('ground truth ate: ', EY_do_A_gt)
            visualise_ATEs(EY_do_A_gt, EYhat_do_A,
                           x_name='E[Y|do(A)] - gt',
                           y_name='beta_A',
                           save_loc=folder,
                           save_name='ate_{}_{}_{}_{}.png'.format(rep, lr_id, dw_id, AW_train.shape[0]))
            causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt - EYhat_do_A))
            av_causal_effect_mean_abs_err = (av_causal_effect_mean_abs_err * rep + causal_effect_mean_abs_err) / (rep + 1)  # compute the average MAE on causal effect over the reps

            # performance log file
            if rep == (args.reps - 1):
                performance_log_file = open(os.path.join(folder, "performance_log_{}.txt".format(AW_train.shape[0]+AW_dev.shape[0]+AW_test.shape[0])), "a")
                performance_log_file.write("lr:{}, dw:{}, average test mse E[(Y-h(A,W,X))^2]:{}, average causal MAE:{}\n".format(lr, dw, test_err_av, av_causal_effect_mean_abs_err))
                performance_log_file.close()

            # causal_effect_mae_file = open(os.path.join(folder, "ate_mae_{}_{}_{}.txt".format(lr_id, dw_id, AW_train.shape[0])), "a")
            # causal_effect_mae_file.write("mae_rep_{}: {}\n".format(rep, causal_effect_mean_abs_err))
            # causal_effect_mae_file.close()

    else:
        print('test')
        opt_res = []
        times = []
        for rep in range(10):
            res_list = []
            other_list = []
            times2 = []
            for lr_id in range(len(lrs)):
                for dw_id in range(len(decay_weights)):
                    load_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,datasize))
                    if os.path.exists(load_path):
                        res = np.load(load_path)
                        res_list += [res['err'].astype(float)]
                        other_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err'].astype(float)]]
                    time_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,datasize))
                    if os.path.exists(time_path):
                        t = np.load(time_path)
                        times2 += [t]
            res_list = np.array(res_list)
            other_list = np.array(other_list)
            other_list = other_list[res_list>0]
            res_list = res_list[res_list>0]
            optim_id = np.argsort(res_list)[0]# np.argmin(res_list)
            print(rep,'--',other_list[optim_id],np.min(res_list))
            opt_res += [other_list[optim_id][-1]]
        print('time: ', np.mean(times),np.std(times))
        print(np.mean(opt_res),np.std(opt_res))



if __name__ == '__main__': 
    # scenarios = ["step", "sin", "abs", "linear"]
    scenarios = ["sim_1d_no_x"]
    # index = int(sys.argv[1])
    # datasize = int(sys.argv[2])
    # sid,index = divmod(index,21)
    # lr_id, dw_id = divmod(index,7)

    for datasize in [5000]:  # [200, 2000]:
        for s in scenarios:
            folder = os.path.join(ROOT_PATH, "MMR_proxy", "results/zoo", s, str(date.today()), args.sem, "nn")
            os.makedirs(folder, exist_ok=True)
            performance_log_file = open(
                os.path.join(folder, "performance_log_{}.txt".format(datasize)), "w")
            performance_log_file.close()
            for lr_id in range(4):
                for dw_id in range(4):
                    run_experiment_nn(s, datasize, [lr_id, dw_id], args=args, folder=folder)

        for s in scenarios:
            run_experiment_nn(s, datasize, [1, 0], training=False, args=args)
