import numpy as np
import pingouin as pg
import statsmodels.api as sm
import os
from simulation.simulation_rhc import calculate_correlation_coeff_rhc
from simulation.util import calculate_partial_correlation_coeff, check_r2, extract_1d_var, calculate_correlation_coeff
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

seed = 0

file_name = 'deaner_edu_split_into_azwyx.npz'  # 'rhc_for_sim_fitting.npz'
file_path = os.path.join(os.path.dirname(__file__), file_name)
root_path = os.path.join(os.path.dirname(__file__), '..')
# 'z-IM', 2: 'z-IR', 3: 'z-IG', 4: 'z-IS'


def load_data_from_edu(file_path):
    data = np.load(file_path, allow_pickle=True)
    a = data['a']
    w = np.concatenate([data['w_IM'], data['w_IR'], data['w_IG'], data['w_IS']], axis=-1)
    z = np.concatenate([data['z_IM'], data['z_IR'], data['z_IG'], data['z_IS']], axis=-1)
    y = np.concatenate([data['y_IM'], data['y_IR']], axis=-1)
    print('load {} datapoints'.format(a.shape[0]))
    x = data['x']
    return a, w, z, y, x


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    u, a, w, z, y, x = data['z1'], data['a'], data['w'], data['z2'], data['y'], data['x']
    return u, a, w, z, y, x


def calculate_correlation_coeff_edu(root_path, save_name, **kwargs):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'edu')):
        os.mkdir(os.path.join(root_path, 'simulation', 'edu'))
    PATH = os.path.join(root_path, 'simulation', 'edu')
    calculate_correlation_coeff(PATH=PATH, save_name=save_name, **kwargs)


def calculate_partial_corr_edu(root_path=root_path, save_name='corr_orig', **kwargs):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'edu')):
        os.mkdir(os.path.join(root_path, 'simulation', 'edu'))
    PATH = os.path.join(root_path, 'simulation', 'edu')
    calculate_partial_correlation_coeff(PATH=PATH, save_name=save_name, **kwargs)


def calculate_partial_corr_rhc(root_path=root_path, save_name='corr_orig', **kwargs):
    if not os.path.exists(os.path.join(root_path, 'simulation', 'rhc')):
        os.mkdir(os.path.join(root_path, 'simulation', 'rhc'))
    PATH = os.path.join(root_path, 'simulation', 'rhc')
    calculate_partial_correlation_coeff(PATH=PATH, save_name=save_name, **kwargs)


def check_r2_rf(covar, outcome, covar_test, outcome_test):
    """
    Fits an random forest from covar to outcome, then report R2.
    Args:
        covar - dictionary for covariates.
        outcome - dictionary with a single key for outcome.
    """

    covar_name_var1d, outcome_name_var1d = extract_1d_var(**covar), extract_1d_var(**outcome)
    covar_name_var1d_te, outcome_name_var1d_te = extract_1d_var(**covar_test), extract_1d_var(**outcome_test)

    tr_size, te_size =covar_name_var1d[0][-1].shape[0], covar_name_var1d_te[0][-1].shape[0]

    print('covar shape: ', covar_name_var1d[0][-1].shape)
    y = np.array([var[-1] for var in outcome_name_var1d]).reshape(tr_size, -1)
    X = np.array([var[-1] for var in covar_name_var1d]).reshape((tr_size, -1))

    y_te = np.array([var[-1] for var in outcome_name_var1d_te]).reshape(te_size, -1)
    X_te = np.array([var[-1] for var in covar_name_var1d_te]).reshape(te_size, -1)
    # print('***************', np.mean(y_te), np.std(y_te))

    print('X shape, y shape', X.shape, y.shape)

    param_grid = {
        'bootstrap': [True],
        'max_depth': [3, 6, 9, 12, 15, 18],
        'n_estimators': [5, 10, 15]
    }
    f = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=f, param_grid=param_grid, cv=4, verbose=2)
    grid_search.fit(X,y)
    f = grid_search.best_estimator_
    r2 = f.score(X_te,y_te)
    y_pred = f.predict(X_te)
    mae = np.mean(np.abs(y_pred - y_te))
    print('the r2 for RF fitting {} to {} is {}, mae {}.'.format([name for name in covar.keys()],
                                                      [name for name in outcome.keys()], r2, mae))
    return r2, mae


def random_select_covar(covar_np, seed, size):
    """
    Args:
        covar_np: np array of data. Number of data points x number of covariates
        seed: random seed for np generator
        size: number of covariates to select
    """
    n_data = covar_np.shape[0]

    if size > covar_np.shape[1]:
        raise ValueError('can not choose number of covariates more than the total number')
    np.random.seed(seed)
    indices = np.random.choice(np.arange(covar_np.shape[1]), size=10)
    covar_np = covar_np[:, indices]
    train, test = covar_np[:int(n_data*0.7), :], covar_np[int(n_data*0.7):, :]

    scaler = preprocessing.StandardScaler()
    train_scaled_np = scaler.fit_transform(train)
    test_scaled_np = scaler.transform(test)

    return train_scaled_np, test_scaled_np


def test_subsample_explanability(covar_np, outcome_np, seeds, size):
    assert covar_np.shape[0] == outcome_np.shape[0]
    n_data = outcome_np.shape[0]
    best_r2, best_seed_r2, best_mae, best_seed_mae = None, None, None, None
    for seed in seeds:
        train_scaled_np, test_scaled_np = random_select_covar(covar_np, seed, size)
        train_outcome, test_outcome = outcome_np[:int(n_data * 0.7)], outcome_np[int(n_data * 0.7):]
        covar, covar_test = {'x': train_scaled_np}, {'x': test_scaled_np}
        outcome, outcome_test = {'out': train_outcome}, {'out': test_outcome}
        r2, mae = check_r2_rf(covar, outcome, covar_test, outcome_test)
        if best_r2 is None or r2 > best_r2:
            best_seed_r2, best_r2 = seed, r2
        if best_mae is None or mae < best_mae:
            best_seed_mae, best_mae = seed, mae
    print('best r2: {}, best mae: {}'.format(best_r2, best_mae))
    return best_seed_r2, best_seed_mae


def main_subsample_for_u_edu(file_path):
    a, w, z, y, x = load_data_from_edu(file_path=file_path)

    n_data = a.shape[0]
    a_2d, w_2d, z_2d, y_2d, x_2d = a.reshape(n_data, -1), w.reshape(n_data, -1), z.reshape(n_data, -1), y.reshape(n_data, -1), x.reshape(n_data, -1)

    covar_np = x_2d
    print('covar shape, y shape', covar_np.shape, y_2d.shape)
    best_seed_r2_y, best_seed_mae_y = test_subsample_explanability(covar_np=covar_np, outcome_np=y_2d, seeds=np.arange(40), size=100)



def main_subsample_for_u(file_path):
    u, a, w, z, y, x = load_data(file_path=file_path)

    n_data = u.shape[0]
    u_2d, a_2d, w_2d, z_2d, y_2d, x_2d = u.reshape(n_data, -1), a.reshape(n_data, -1), w.reshape(n_data, -1), z.reshape(n_data, -1), y.reshape(n_data, -1), x.reshape(n_data, -1)

    covar_np = np.concatenate([u_2d, w_2d, z_2d, x_2d], axis=-1)
    print('covar shape, y shape', covar_np.shape, y_2d.shape)
    best_seed_r2_y, best_seed_mae_y = test_subsample_explanability(covar_np=covar_np, outcome_np=y_2d, seeds=np.arange(40), size=100)


def main_edu(file_path):
    a, w, z, y, x = load_data_from_edu(file_path=file_path)
    calculate_correlation_coeff_edu(x=x, u=u, z=z, w=w, a=a, y=y, root_path=root_path, save_name='corr_orig')
    calculate_partial_corr_edu(x=x, z=z, w=w, a=a, y=y, root_path=root_path, save_name='pcorr_orig')

    train, test = {}, {}
    split = int(a.shape[0] * 0.66)
    for name, datalet in [('a', a), ('w', w), ('z', z), ('y', y), ('x', x)]:
        train_data, test_data = datalet[:split], datalet[split:]
        train[name], test[name] = train_data, test_data

    transforms = {}
    tr_data_scaled = {'a': train['a'], 'y': train['y']}
    te_data_scaled = {'a': test['a'], 'y': test['y']}

    for name in ['w', 'z', 'x']:
        scaler = preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(train[name])
        transforms[name] = scaler
        tr_data_scaled[name] = data_scaled
        te_data_scaled[name] = scaler.transform(test[name])

    # original
    covar = {'a': a, 'w': w, 'z': z, 'x': x}
    outcome = {'y': y}
    check_r2(covar, outcome)

    covar = {'w': w, 'z': z, 'x': x}
    outcome = {'a': a}
    check_r2(covar, outcome)

    # standardised
    # data_scaled = {'a': a, 'y': y}
    #
    # for name, var in [('u', u), ('w', w), ('z', z), ('x', x)]:
    #     scaler = preprocessing.StandardScaler()
    #     data_sd = scaler.fit_transform(var)
    #     data_scaled[name] = data_sd
    print('!STANDARDISED DATA!')
    covar = {'a': tr_data_scaled['a'],
             'w': tr_data_scaled['w'],
             'z': tr_data_scaled['z'],
             'x': tr_data_scaled['x']}
    outcome = {'y': tr_data_scaled['y']}
    covar_te = {'a': te_data_scaled['a'],
             'w': te_data_scaled['w'],
             'z': te_data_scaled['z'],
             'x': te_data_scaled['x']}
    outcome_te = {'y': te_data_scaled['y']}
    check_r2(covar, outcome)
    check_r2_rf(covar, outcome, covar_te, outcome_te)

    covar = {'w': tr_data_scaled['w'],
             'z': tr_data_scaled['z'],
             'x': tr_data_scaled['x']}
    outcome = {'a': tr_data_scaled['a']}
    covar_te = {'w': te_data_scaled['w'],
             'z': te_data_scaled['z'],
             'x': te_data_scaled['x']}
    outcome_te = {'a': te_data_scaled['a']}
    check_r2(covar, outcome)
    check_r2_rf(covar, outcome, covar_te, outcome_te)

def main(file_path):

    u, a, w, z, y, x = load_data(file_path=file_path)
    calculate_correlation_coeff_rhc(x=x, u=u, z=z, w=w, a=a, y=y, root_path=root_path, save_name='corr_orig')
    calculate_partial_corr_rhc(x=x, u=u, z=z, w=w, a=a, y=y, root_path=root_path, save_name='pcorr_orig')

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

    # original
    covar = {'u': u, 'a': a, 'w': w, 'z': z, 'x': x}
    outcome = {'y': y}
    check_r2(covar, outcome)

    covar = {'u': u, 'w': w, 'z': z, 'x': x}
    outcome = {'a': a}
    check_r2(covar, outcome)

    # standardised
    # data_scaled = {'a': a, 'y': y}
    #
    # for name, var in [('u', u), ('w', w), ('z', z), ('x', x)]:
    #     scaler = preprocessing.StandardScaler()
    #     data_sd = scaler.fit_transform(var)
    #     data_scaled[name] = data_sd
    print('!STANDARDISED DATA!')
    covar = {'u': tr_data_scaled['u'],
             'a': tr_data_scaled['a'],
             'w': tr_data_scaled['w'],
             'z': tr_data_scaled['z'],
             'x': tr_data_scaled['x']}
    outcome = {'y': tr_data_scaled['y']}
    covar_te = {'u': te_data_scaled['u'],
             'a': te_data_scaled['a'],
             'w': te_data_scaled['w'],
             'z': te_data_scaled['z'],
             'x': te_data_scaled['x']}
    outcome_te = {'y': te_data_scaled['y']}
    check_r2(covar, outcome)
    check_r2_rf(covar, outcome, covar_te, outcome_te)

    covar = {'u': tr_data_scaled['u'],
             'w': tr_data_scaled['w'],
             'z': tr_data_scaled['z'],
             'x': tr_data_scaled['x']}
    outcome = {'a': tr_data_scaled['a']}
    covar_te = {'u': te_data_scaled['u'],
             'w': te_data_scaled['w'],
             'z': te_data_scaled['z'],
             'x': te_data_scaled['x']}
    outcome_te = {'a': te_data_scaled['a']}
    check_r2(covar, outcome)
    check_r2_rf(covar, outcome, covar_te, outcome_te)

if __name__ == "__main__":
    main_subsample_for_u_edu(file_path)