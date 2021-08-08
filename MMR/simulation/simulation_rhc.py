from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import preprocessing, linear_model
from sklearn.svm import SVR
import numpy as np
from numpy import concatenate as cat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from simulation.util import calculate_correlation_coeff


seed = 0
file_path = os.path.join(os.path.dirname(__file__), 'rhc_for_sim_fitting.npz')
root_path = os.path.join(os.path.dirname(__file__), '..')


def calculate_correlation_coeff_rhc(root_path, save_name, **kwargs):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'rhc')):
        os.mkdir(os.path.join(root_path, 'simulation', 'rhc'))
    PATH = os.path.join(root_path, 'simulation', 'rhc')
    calculate_correlation_coeff(PATH=PATH, save_name=save_name, **kwargs)


def plot_real_data(file_path, root_path):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'rhc')):
        os.mkdir(os.path.join(root_path, 'simulation', 'rhc'))

    data = np.load(file_path)
    u, a, w, z, y, x = data['z1'], data['a'], data['w'], data['z2'], data['y'], data['x']
    D = pd.DataFrame([u[:200,0], a[:200,0], w[:200,0], w[:200,1], z[:200,0], y[:200]]).T
    D.columns = ['U', 'A', 'W1', 'W2', 'Z', 'Y']
    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(os.path.join(root_path, 'simulation', 'rhc', 'orig_pairwise.png')), plt.close()
    calculate_correlation_coeff_rhc(x=x, u=u, z=z, w=w, a=a, y=y, root_path=root_path, save_name='corr_orig')


def load_split_and_standardise(file_path, root_path):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'rhc')):
        os.mkdir(os.path.join(root_path, 'simulation', 'rhc'))
    data = np.load(file_path)
    u, a, w, z, y, x = data['z1'], data['a'], data['w'], data['z2'], data['y'], data['x']


    train, test = {}, {}
    split = int(u.shape[0] * 0.66)
    for name, datalet in [('u', u), ('a', a), ('w', w), ('z', z), ('y', y), ('x', x)]:
        train_data, test_data = datalet[:split], datalet[split:]
        train[name], test[name] = train_data, test_data

    transforms = {}
    tr_data_scaled = {'a': train['a'], 'y': train['y']}
    te_data_scaled = {'a': test['a'], 'y': test['y']}

    for name in ['u', 'w', 'z', 'x']:
        scaler = preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(train[name])
        transforms[name] = scaler
        tr_data_scaled[name] = data_scaled
        te_data_scaled[name] = scaler.transform(test[name])

    D = pd.DataFrame([tr_data_scaled['u'][:200,0], tr_data_scaled['a'][:200,0], tr_data_scaled['w'][:200,0], tr_data_scaled['w'][:200,1], tr_data_scaled['z'][:200,0], tr_data_scaled['y'][:200]]).T
    D.columns = ['U', 'A', 'W1', 'W2', 'Z', 'Y']
    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(os.path.join(root_path, 'simulation', 'rhc', 'scaled_orig_pairwise.png')), plt.close()
    calculate_correlation_coeff_rhc(x=tr_data_scaled['x'],
                                    u=tr_data_scaled['u'],
                                    z=tr_data_scaled['z'],
                                    w=tr_data_scaled['w'],
                                    a=tr_data_scaled['a'],
                                    y=tr_data_scaled['y'],
                                    root_path=root_path, save_name='corr_scaled_orig')

    return tr_data_scaled, te_data_scaled, train, test, transforms


def rt_residual_variance(model, data, label):
    fitted_vals = model.predict(data)
    residual_sq = (fitted_vals.squeeze() - label.squeeze()) ** 2
    assert residual_sq.shape[0] == label.shape[0], residual_sq.shape[1] == label.shape[1]
    rt_mean_residual_sq = np.sqrt(np.mean(residual_sq, axis=0))
    return rt_mean_residual_sq


def fit_structural_eqs(tr_data, te_data, nonlinear_regressor, kwargs4regressor):
    print('fitting structural equations')
    u, a, w, z, y, x = tr_data['u'], tr_data['a'], tr_data['w'], tr_data['z'], tr_data['y'], tr_data['x']
    f_xu = linear_model.LinearRegression()
    f_uxz = linear_model.LinearRegression()
    ux_tr = np.concatenate([u, x], axis=-1)
    ux_te = np.concatenate([te_data['u'], te_data['x']], axis=-1)
    f_uxw = linear_model.LinearRegression()
    f_xu.fit(x, u)
    xu_test_err = f_xu.score(te_data['x'], te_data['u'].squeeze())
    f_uxz.fit(ux_tr, z)
    uxz_test_err = f_uxz.score(ux_te, te_data['z'])
    f_uxw.fit(ux_tr, w)
    uxw_test_err = f_uxw.score(ux_te, te_data['w'])

    f_uzxa = linear_model.LogisticRegression()
    uzx_tr = np.concatenate([u, z, x], axis=-1)
    uzx_te = np.concatenate([te_data['u'], te_data['z'], te_data['x']], axis=-1)
    f_uzxa.fit(uzx_tr, a)
    uzxa_test_err = f_uzxa.score(uzx_te, te_data['a'])

    f_uwaxy = nonlinear_regressor(**kwargs4regressor)
    uwax_tr = np.concatenate([u, w, a, x], axis=-1)
    uwax_te = np.concatenate([te_data['u'], te_data['w'], te_data['a'], te_data['x']], axis=-1)
    f_uwaxy.fit(uwax_tr, y)
    uwaxy_test_err = f_uwaxy.score(uwax_te, te_data['y'])

    models = []
    for model in [(f_xu, x, u), (f_uxz, ux_tr, z), (f_uxw, ux_tr, w), (f_uwaxy, uwax_tr, y)]:
        rmrs = rt_residual_variance(*model)
        models.append((model[0], rmrs))
    models.append(f_uzxa)
    # model order: u, z, w, y, a as outputs
    return models


def generate_data_sample(x_samples_transformed, u_model, z_model, w_model, y_model, a_model, root_path, reps):
    print('generating data samples.')
    if not os.path.exists(os.path.join(root_path, 'simulation', 'rhc')):
        os.mkdir(os.path.join(root_path, 'simulation', 'rhc'))
    if not os.path.exists(os.path.join(root_path, 'data/zoo/rhc_fitted_no_x/')):
        os.mkdir(os.path.join(root_path, 'data/zoo/rhc_fitted_no_x/'))
    if not os.path.exists(os.path.join(root_path, 'data/zoo/rhc_fitted_with_x/')):
        os.mkdir(os.path.join(root_path, 'data/zoo/rhc_fitted_with_x/'))

    x_samples_transformed = np.repeat(x_samples_transformed, reps, axis=0)
    n_samples = x_samples_transformed.shape[0]
    f_u, f_z, f_w, f_y, f_a = u_model[0], z_model[0], w_model[0], y_model[0], a_model
    u_samples = f_u.predict(x_samples_transformed).reshape(-1,1) + u_model[1] * np.random.randn(n_samples, 1)
    z_samples = f_z.predict(cat([u_samples, x_samples_transformed], axis=-1)).reshape(-1, 1) + z_model[1] * np.random.randn(n_samples, 1)
    w_samples = f_w.predict(cat([u_samples, x_samples_transformed], axis=-1)).reshape(-1, 2) + w_model[1] * np.random.randn(n_samples, 2)
    a_samples = f_a.predict(cat([u_samples, z_samples, x_samples_transformed], axis=-1)).reshape(-1,1)
    y_samples = f_y.predict(cat([u_samples, w_samples, a_samples, x_samples_transformed], axis=-1)).reshape(-1, 1) \
                + y_model[1] * np.random.randn(n_samples, 1)

    D = pd.DataFrame([u_samples[:200,0], a_samples[:200,0], w_samples[:200, 0], w_samples[:200,1], z_samples[:200,0], y_samples[:200]]).T
    D.columns = ['U', 'A', 'W1', 'W2', 'Z', 'Y']

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(os.path.join(root_path, 'simulation', 'rhc', 'fitted_pairwise.png')), plt.close()
    calculate_correlation_coeff_rhc(x=x_samples_transformed[:1000],
                                u=u_samples[:1000],
                                z=z_samples[:1000],
                                w=w_samples[:1000],
                                a=a_samples[:1000],
                                y=y_samples[:1000],
                                root_path=root_path, save_name='corr_fitted')

    np.savez(os.path.join(root_path, 'data/zoo/rhc_fitted_no_x/rhc.npz'), u=u_samples, z=z_samples, w=w_samples, a=a_samples, y=y_samples)
    np.savez(os.path.join(root_path, 'data/zoo/rhc_fitted_with_x/rhc.npz'), u=u_samples, x=x_samples_transformed, z=z_samples, w=w_samples, a=a_samples, y=y_samples)

    savez_inp_no_x = {'splits': ['train', 'test', 'dev']}
    savez_inp_with_x = {'splits': ['train', 'test', 'dev']}
    for name, data in [('x', x_samples_transformed), ('z', z_samples), ('w', w_samples), ('a', a_samples), ('y', y_samples), ('u', u_samples)]:
        train, test, dev = data[:int(n_samples * 0.6)], data[int(n_samples * 0.6): int(n_samples * 0.8)], data[int(n_samples * 0.8):]
        savez_inp_with_x['train_{}'.format(name)] = train
        savez_inp_with_x['test_{}'.format(name)] = test
        savez_inp_with_x['dev_{}'.format(name)] = dev
        if not name == 'x':
            savez_inp_no_x['train_{}'.format(name)] = train
            savez_inp_no_x['test_{}'.format(name)] = test
            savez_inp_no_x['dev_{}'.format(name)] = dev
    np.savez(os.path.join(root_path, 'data/zoo/rhc_fitted_no_x/main_.npz'), **savez_inp_no_x)
    np.savez(os.path.join(root_path, 'data/zoo/rhc_fitted_with_x/main_.npz'), **savez_inp_with_x)

    # causal ground truth
    do_A = np.array([0, 1])
    EY_do_A = []
    for a in do_A:
        A_ = np.repeat(a, n_samples).reshape(-1,1)
        Y_do_A = f_y.predict(cat([u_samples, w_samples, A_, x_samples_transformed], axis=-1)).reshape(-1, 1) \
                + y_model[1] * np.random.randn(n_samples, 1)

        eY_do_A = np.mean(Y_do_A)
        EY_do_A.append(eY_do_A)

    EY_do_A = np.array(EY_do_A)
    np.savez(os.path.join(root_path, 'data/zoo/rhc_fitted_with_x/do_A_.npz'),
             do_A = do_A,
             gt_EY_do_A = EY_do_A)


def main():
    plot_real_data(file_path=file_path, root_path=root_path)
    tr_data_scaled, te_data_scaled, train, test, transforms = load_split_and_standardise(file_path=file_path, root_path=root_path)

    f_u_, f_z_, f_w_, f_y_, f_a_ = fit_structural_eqs(tr_data=tr_data_scaled, te_data=te_data_scaled,
                                                      nonlinear_regressor=RandomForestRegressor,
                                                      kwargs4regressor={'max_depth': 2, 'random_state': seed})

    x_samples_transformed = cat([tr_data_scaled['x'], te_data_scaled['x']], axis=0)
    generate_data_sample(x_samples_transformed=x_samples_transformed,
                         u_model=f_u_, z_model=f_z_, w_model=f_w_, y_model=f_y_, a_model=f_a_,
                         root_path=root_path, reps=5)


if __name__ == '__main__':
    main()