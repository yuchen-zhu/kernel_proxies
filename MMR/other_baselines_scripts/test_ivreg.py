from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np
from MMR.other_baselines_scripts.run_zoo_experiments_more_baselines import scale_all
from MMR_proxy.util import load_data, ROOT_PATH

scenario_name = 'sim_1d_no_x'
data_seed = 100
sem = 'linear033'
trainsz = 500

train, dev, test = load_data(
    ROOT_PATH + '/data/zoo/' + scenario_name + '/main_{}_seed{}.npz'.format(sem, data_seed))

A_scaled, Y_scaled, Z_scaled, W_scaled, test_A_scaled, test_Y_scaled, test_Z_scaled, test_W_scaled, A_scaler, Y_scaler, Z_scaler, W_scaler = \
    scale_all(train_A=train.a[:trainsz].reshape(trainsz, -1), train_Y=train.y[:trainsz].reshape(trainsz, -1),
              train_Z=train.z[:trainsz].reshape(trainsz, -1), train_W=train.w[:trainsz].reshape(trainsz, -1),
              test_A=test.a[:test_size].reshape(test_size, -1), test_Y=test.y[:test_size].reshape(test_size, -1),
              test_Z=test.z[:test_size].reshape(test_size, -1), test_W=test.w[:test_size].reshape(test_size, -1))

lm = IV2SLS(endog=Y_scaled, exog=np.concatenate([A_scaled, W_scaled], axis=-1), instrument=np.concatenate([A_scaled, Z_scaled], axis=-1))

print(dir(lm))