# from models.cnn_models import DefaultCNN

# if __name__ == "__main__":
#     from abstract_baseline import AbstractBaseline
#     from agmm.deep_gmm import DeepGMM
# else:
#     from .abstract_baseline import AbstractBaseline
#     from .agmm.deep_gmm import DeepGMM
import pystan
from baselines_scripts.baselines.abstract_baseline import AbstractBaseline, AbstractLVMBaseline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, \
    StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
import sklearn.metrics.pairwise
from sklearn.neural_network import MLPClassifier
# from baselines.lvm import lvm_code
# from baselines.vae import VAE

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

import keras
from econml.deepiv import DeepIVEstimator

# import statsmodels.sandbox.regression.gmm
# import statsmodels.tools.tools

import os
import time


class SklearnBaseline(AbstractBaseline):
    def _predict(self, x, context):
        return self._model.predict(self.augment(x, context))


class DirectLinearRegression(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)
        direct_regression = sklearn.linear_model.LinearRegression()
        direct_regression.fit(x, y)
        self._model = direct_regression
        return self


class DirectRidge(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(ridge__alpha=np.logspace(-5, 5, 11))
        pipe = Pipeline([('ridge', Ridge())])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)

        direct_regression.fit(x, y)
        self._model = direct_regression
        return self



class DirectPoly(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(poly__degree=range(1, 4),
                      ridge__alpha=np.logspace(-5, 5, 11))
        pipe = Pipeline([('poly', PolynomialFeatures()),
                         ('ridge', Ridge())])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)

        direct_regression.fit(x, y)
        self._model = direct_regression
        return self


class DirectMNIST(AbstractBaseline):
    def __init__(self, n_epochs=6, batch_size=128, lr=0.005):
        super().__init__()
        self._n_epochs = n_epochs
        self._n_batch_size = batch_size
        self._lr = lr

    def _fit(self, x, y, z, context=None):
        model = DefaultCNN(cuda=torch.cuda.is_available())
        model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        model.train()

        x = self.augment(x, context)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        t0 = time.time()
        train = data_utils.DataLoader(data_utils.TensorDataset(x, y),
                                      batch_size=self._n_batch_size,
                                      shuffle=True)
        for epoch in range(self._n_epochs):
            losses = list()
            print("Epoch: ", epoch + 1, "/", self._n_epochs, " batch size: ",
                  self._n_batch_size)
            for i, (x, y) in enumerate(train):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                losses += [loss.data.cpu().numpy()]
                loss.backward()
                optimizer.step()
            print("   train loss", np.mean(losses))
        self._model = model
        return time.time()-t0

    def _predict(self, x, context):
        self._model.eval()
        x = self.augment(x, context)
        x = torch.tensor(x, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        return self._model(x).data.cpu().numpy()


class DirectNN(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(nn__alpha=np.logspace(-5, 5, 5),
                      nn__hidden_layer_sizes=[(10,), (20,), (10, 10), (20, 10),
                                              (10, 10, 10), (20, 10, 5)])
        pipe = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)
        t0 = time.time()
        direct_regression.fit(x, y.flatten())
        self._model = direct_regression
        # print('DirectNN model: ', self._model)
        return time.time()-t0

    def _predict(self, x, context):
        return self._model.predict(self.augment(x, context)).reshape((-1, 1))


class KRidge(SklearnBaseline):
    def _fit(self, x, y, z, context):
        t0 = time.time()
        x = self.augment(x, context)

        params = dict(ridge__alpha=np.logspace(-10,-2,11))
        pipe = Pipeline([('ridge', KernelRidge(kernel='rbf'))])
        kernel_regr = GridSearchCV(pipe, param_grid=params, cv=5)
        kernel_regr.fit(x, y.flatten())
        self._model = kernel_regr
        return time.time()-t0


class Direct_discr(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        t0 = time.time()
        self._model = []
        for a in range(3):
            a_idx = (x == a)
            y_idx = y[a_idx]
            y_do_a = np.mean(y_idx)
            self._model.append(y_do_a)
        self._model = np.array(self._model)

        return time.time()-t0

    def _predict(self, x, context):
        return self._model[x].reshape(-1,1)





class DeepIV(AbstractBaseline):
    def __init__(self, treatment_model=None):
        if treatment_model is None:
            print("Using standard treatment model...")
            self._treatment_model = lambda input_shape: keras.models.Sequential(
                [keras.layers.Dense(8,
                                    activation='relu',
                                    input_shape=input_shape),
                 keras.layers.Dropout(0.17),
                 keras.layers.Dense(4,
                                    activation='relu'),
                 keras.layers.Dropout(0.17),
                 keras.layers.Dense(2,
                                    activation='relu'),
                 keras.layers.Dropout(0.17)])

        else:
            if keras.backend.image_data_format() == "channels_first":
                image_shape = (1, 28, 28)
            else:
                image_shape = (28, 28, 1)

            self._treatment_model = lambda input_shape: keras.models.Sequential([
                keras.layers.Reshape(image_shape, input_shape=input_shape),
                keras.layers.Conv2D(16, kernel_size=(3, 3),
                                    activation='relu'),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.1),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.1)])

    def _fit(self, x, y, z, context=None):
        if context is None:
            # context = np.empty((x.shape[0], 0))
            context = np.zeros((x.shape[0], 1))

        x_dim = x.shape[1]
        z_dim = z.shape[1]
        context_dim = context.shape[1]

        treatment_model = self._treatment_model((context_dim + z_dim,))

        response_model = keras.models.Sequential([keras.layers.Dense(128,
                                                              activation='relu',
                                                              input_shape=(
                                                                  context_dim + x_dim,)),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(64,
                                                              activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(32,
                                                              activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(1)])

        self._model = DeepIVEstimator(n_components=10,
                                      # Number of gaussians in the mixture density networks)
                                      m=lambda _z, _context: treatment_model(
                                          keras.layers.concatenate(
                                              [_z, _context])),
                                      # Treatment model
                                      h=lambda _t, _context: response_model(
                                          keras.layers.concatenate(
                                              [_t, _context])),
                                      # Response model
                                      n_samples=1
                                      )
        t0 = time.time()
        self._model.fit(Y=y, T=x, X=context, Z=z)
        return time.time() - t0

    def _predict(self, x, context):
        if context is None:
            # context = np.empty((x.shape[0], 0))
            context = np.zeros((x.shape[0], 1))

        return self._model.predict(x, context)


class AGMM(AbstractBaseline):
    def _fit(self, x, y, z, context=None):
        _z = self.augment(z, context)
        _x = self.augment(x, context)

        self._model = DeepGMM(n_critics=50, num_steps=100,
                              learning_rate_modeler=0.01,
                              learning_rate_critics=0.1, critics_jitter=True,
                              eta_hedge=0.16, bootstrap_hedge=False,
                              l1_reg_weight_modeler=0.0,
                              l2_reg_weight_modeler=0.0,
                              dnn_layers=[1000, 1000, 1000], dnn_poly_degree=1,
                              log_summary=False, summary_dir='', random_seed=30)
        t0 = time.time()
        self._model.fit(_z, _x, y)
        return time.time()-t0

    def _predict(self, x, context):
        _x = self.augment(x, context)

        return self._model.predict(_x).reshape(-1, 1)


def cv_remove_neg(cv_results):
    cv = int(sum([1 for e in cv_results.keys() if 'split' in e]))
    neg_pos = np.sum([cv_results['split{}_test_score'.format(i)]>=0 for i in range(cv)],axis=0).astype(int)
    print(cv==neg_pos)
    mean_test_score = (cv_results['mean_test_score'])[neg_pos==cv]
    params = np.array(cv_results['params'])[neg_pos==cv]
    print(len(params),len(neg_pos))
    return params[np.argmin(mean_test_score)]


class Poly2SLS(SklearnBaseline):
    def __init__(self, poly_degree=range(1, 4),
                 ridge_alpha=np.logspace(-5, 5, 11)):
        super().__init__()
        self._poly_degree = poly_degree
        self._ridge_alpha = ridge_alpha

    def _fit(self, x, y, z, context=None):
        '''
        Two stage least squares with polynomial basis function.
        - x: treatment
        - y: outcome
        - z: instrument
        - context: additional information
        '''
        params = dict(poly__degree=self._poly_degree,
                      ridge__alpha=self._ridge_alpha)
        pipe = Pipeline([('poly', PolynomialFeatures()),
                         ('ridge', Ridge())])
        stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z, x)
        #cv_results = stage_1.cv_results_
        #stage_1.best_estimator_.set_params(**cv_remove_neg(cv_results))
        x_hat = stage_1.predict(_z)
        print(stage_1.best_params_)
        pipe2 = Pipeline([('poly', PolynomialFeatures()),
                          ('ridge', Ridge())])
        stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
        t0 = time.time()
        stage_2.fit(self.augment(x_hat, context), y)
        print(stage_2.best_params_)
        #cv_results = stage_2.cv_results_
        #stage_2.best_estimator_.set_params(**cv_remove_neg(cv_results))
        self._model = stage_2
        return time.time()-t0


class Bases2SLS(SklearnBaseline):
    def __init__(self, basis_fn=5):
        super().__init__()
        self._poly_degree = 3
        # x_features.append(np.exp(np.sum(np.abs(x - i) ** 2, axis=-1) / (-2) / ()))
        # self._poly_degree = poly_degree
        # self._ridge_alpha = ridge_alpha

    def _fit(self, x, y, z, context=None):
        '''
        Two stage least squares with polynomial basis function.
        - x: treatment
        - y: outcome
        - z: instrument
        - context: additional information
        '''
        split = x.shape[0]//2
        t0 = time.time()
        _z = self.augment(z, context)

        ftmap_w = PolynomialFeatures(self._poly_degree)
        w_fea_1 = ftmap_w.fit_transform(x[:split, 1:])

        ftmap_az = PolynomialFeatures(self._poly_degree)
        az_fea_1 = ftmap_az.fit_transform(_z[:split])
        # _z = az_fea_1

        params = dict(nn__alpha=np.logspace(-5, 5, 5),
                      nn__hidden_layer_sizes=[(10,), (20,), (10, 10), (20, 10),
                                              (10, 10, 10), (20, 10, 5)])
        pipe = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
        _hat = stage_1.fit(az_fea_1, w_fea_1)

        w_fea_hat = stage_1.predict(ftmap_az.transform(_z[split:]))
        print('warning: sk NN expects 1d actions!!!')
        print(stage_1.best_params_)
        stage_2 = Ridge(alpha=w_fea_hat.shape[0]**0.5)

        ftmap_a = PolynomialFeatures(self._poly_degree)
        a_fea = ftmap_a.fit_transform(x[split:, :1])
        aw_fea = np.einsum('ij, ik -> ijk', a_fea, w_fea_hat).reshape(a_fea.shape[0], -1)

        stage_2.fit(aw_fea, y[split:].squeeze())

        self._model = stage_2
        return time.time()-t0


    def _predict(self, x, context):
        # x_features = []
        # for i in range(1, self._basis_fn):
        #     x_features.append(np.sin(x * i) * np.exp(-0.1*np.sum(np.abs(x)**2, axis=-1).reshape(-1,1)))
        #     x_features.append(np.cos(x * i) * np.exp(-0.1*np.sum(np.abs(x)**2, axis=-1).reshape(-1,1)))
        # x_features = np.concatenate(x_features, axis=-1)

        w_fea = PolynomialFeatures(self._poly_degree).fit_transform(x[:, 1:])
        a_fea = PolynomialFeatures(self._poly_degree).fit_transform(x[:, 0:1])
        aw_fea = np.einsum('ij, ik -> ijk', a_fea, w_fea).reshape(a_fea.shape[0], -1)
        preds = self._model.predict(aw_fea)
        return preds


class NNP2SLS(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        params = dict(nn__alpha=np.logspace(-5, 5, 5),
                      nn__hidden_layer_sizes=[(10,), (20,), (10, 10), (20, 10),
                                              (10, 10, 10), (20, 10, 5)])
        pipe = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z, x[:, 1:])
        print('warning: NNP2SLS expects 1d actions!!!')
        #cv_results = stage_1.cv_results_
        #stage_1.best_estimator_.set_params(**cv_remove_neg(cv_results))
        x_hat = stage_1.predict(_z).reshape(-1, x[:, 1:].shape[-1])
        print(stage_1.best_params_)
        pipe2 = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
        t0 = time.time()
        y = y.squeeze()
        # assert len(y.shape) == 1
        stage_2.fit(self.augment(np.concatenate([x[:, 0:1], x_hat], axis=-1), context), y)
        print(stage_2.best_params_)
        #cv_results = stage_2.cv_results_
        #stage_2.best_estimator_.set_params(**cv_remove_neg(cv_results))
        self._model = stage_2
        return time.time()-t0


class NN2SLS(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        params = dict(nn__alpha=np.logspace(-5, 5, 5),
                      nn__hidden_layer_sizes=[(10,), (20,), (10, 10), (20, 10),
                                              (10, 10, 10), (20, 10, 5)])
        pipe = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z, x)
        print('warning: NNP2SLS expects 1d actions!!!')
        #cv_results = stage_1.cv_results_
        #stage_1.best_estimator_.set_params(**cv_remove_neg(cv_results))
        x_hat = stage_1.predict(_z)
        print(stage_1.best_params_)
        pipe2 = Pipeline([('standard', StandardScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
        t0 = time.time()
        y = y.squeeze()
        # assert len(y.shape) == 1
        stage_2.fit(self.augment(x_hat, context), y)
        print(stage_2.best_params_)
        #cv_results = stage_2.cv_results_
        #stage_2.best_estimator_.set_params(**cv_remove_neg(cv_results))
        self._model = stage_2
        return time.time()-t0


class Vanilla2SLS(Poly2SLS):
    def display(self):
        weights = self.arr2str(self._model.coef_)
        bias = self.arr2str(self._model.intercept_)
        print("%s*x + %s" % (weights, bias))

    def _fit(self, x, y, z, context=None):
        '''
        Vanilla two stage least squares.
        - x: treatment
        - y: outcome
        - z: instrument
        - context: additional information
        '''
        split = x.shape[0]//2
        stage_1 = LinearRegression()
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z[:split], x[:split, 1:])
        x_hat = stage_1.predict(_z[split:])

        stage_2 = LinearRegression()
        t0 = time.time()
        stage_2.fit(np.concatenate([x[split:, :1], self.augment(x_hat, context)], axis=-1), y[split:])

        self._model = stage_2
        return time.time()-t0

class LVMLinear(AbstractLVMBaseline):
    def  _fit(self, a, y, z, w):

        assert all([var.ndim <= 2 for var in [a, y, z, w]])
        assert([var.shape[0] == a.shape[0] for var in [a, y, z, w]])

        data = {}
        for name, var in [('a', a), ('y', y), ('z', z), ('w', w)]:
            if var.ndim == 1:
                var = var.reshape(-1,1)
                data[name] = var
            elif var.ndim == 2:
                data[name] = var
        data['N'] = a.shape[0]

        sm = pystan.StanModel(model_code=lvm_code)
        fitted_sm = sm.sampling(data=data, iter=2000, chains=2, verbose=True)
        self._model = fitted_sm
        return self

    def _predict(self, a):
        la = self._model.extract(permuted=True)
        # a_uz = np.mean(la['a_uz'], axis=0)
        a_uw = np.mean(la['a_uw'], axis=0)
        # a_ua = np.mean(la['a_ua'], axis=0)
        a_uy = np.mean(la['a_uy'], axis=0)
        # a_za = np.mean(la['a_za'], axis=0)
        a_ay = np.mean(la['a_ay'], axis=0)
        a_wy = np.mean(la['a_wy'], axis=0)

        # c_z = np.mean(la['c_z'], axis=0)
        c_w = np.mean(la['c_w'], axis=0)
        # c_a = np.mean(la['c_a'], axis=0)
        # c_y = np.mean(la['c_y'], axis=0)

        # sigma_z_Sq = np.mean(la['sigma_z_Sq'], axis=0)
        # sigma_w_Sq = np.mean(la['sigma_w_Sq'], axis=0)
        # sigma_a_Sq = np.mean(la['sigma_a_Sq'], axis=0)
        # sigma_y_Sq = np.mean(la['sigma_y_Sq'], axis=0)

        u = la['u']
        EY_do_a = []
        for a_ in a:
            a_s = np.repeat(a_, u.shape[0]).reshape(-1,1)
            assert len(a_s.shape) == 2
            assert (a_s.shape[0] == u.shape[0])
            assert (a_s.shape[1] == u.shape[1])
            w = a_uw * u + c_w
            EY_do_a_ = np.mean(a_uy * u + a_ay * a_s + a_wy * w, axis=0)
            EY_do_a.append((EY_do_a_))

        return np.array(EY_do_a)


class LVMVAE(AbstractLVMBaseline):
    def _fit(self, a, y, z, w):
        u_dim = 1
        self.u_dim = u_dim

        assert all([var.ndim <= 2 for var in [a, y, z, w]])
        assert ([var.shape[0] == a.shape[0] for var in [a, y, z, w]])

        data = {}
        for name, var in [('a', a), ('y', y), ('z', z), ('w', w)]:
            if var.ndim == 1:
                var = var.reshape(-1, 1)
                data[name] = var
            elif var.ndim == 2:
                data[name] = var

        vae = VAE(hidden_1=4, dim_y=y.ndim, dim_u=u_dim, dim_a=a.ndim, dim_w=w.ndim, dim_z=z.ndim, C_init=0.5, epochs=100, batch_size=64, lr=0.01)
        fitted_vae = vae.fit(**data)
        self._model = fitted_vae
        return self

    def _predict(self, a):

        u = torch.as_tensor(np.random.normal(0,1,(200, self.u_dim))).float()
        w = self._model.decode_w(u)
        assert w.ndim == 2

        if a.ndim == 1:
            a = a.reshape(-1,1)
        a_rep = np.repeat(a, u.shape[0], axis=1).reshape(-1,1)
        u_rep = np.repeat(np.expand_dims(u, axis=0), a.shape[0], axis=0).reshape(u.shape[0] * a.shape[0], -1)
        w_rep = np.repeat(np.expand_dims(w, axis=0), a.shape[0], axis=0).reshape(w.shape[0] * a.shape[0], -1)
        y_rep = self._model.decode_y(u=u_rep, a=a_rep, w=w_rep)
        EY_do_A = np.mean(y_rep.reshape(-1, u.shape[0]), axis=-1)

        return EY_do_A


class GMMfromStatsmodels(AbstractBaseline):
    def _fit(self, x, y, z, context=None):
        z = self.augment(z, context)
        z = statsmodels.tools.tools.add_constant(z, prepend=False)

        x = self.augment(x, context)
        x = statsmodels.tools.tools.add_constant(x, prepend=False)

        resultIV = statsmodels.sandbox.regression.gmm.IVGMM(y, x, z).fit(
            optim_args={"disp": True, "gtol": 1e-08, "epsilon": 1e-10,
                        "maxiter": 250}, maxiter=1,
            inv_weights=np.eye(z.shape[1]))
        print(resultIV.model.gmmobjective(resultIV.params, np.eye(z.shape[1])))
        # print(resultIV.model.momcond_mean(resultIV.params))

        self._model = resultIV

    def display(self):
        weights = self.arr2str(self._model.params[:-1])
        bias = self.arr2str(self._model.params[-1])
        print("%s*x + %s" % (weights, bias))

    def _predict(self, x, context):
        x = self.augment(x, context)
        x = statsmodels.tools.tools.add_constant(x, prepend=False)

        return self._model.predict(x)


class Featurizer(object):
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            return torch.from_numpy(self._transform(X.data.cpu().numpy()))
        else:
            return self._transform(X)

    def is_initialized(self):
        return self._n_features is not None

    def n_features(self):
        if self.is_initialized():
            return self._n_features
        else:
            raise ValueError("Need to call transform first")


class VanillaFeatures(Featurizer):
    def __init__(self, add_constant=True):
        self._add_constant = add_constant

    def _transform(self, X):
        self._n_features = X.shape[1] + int(self._add_constant)
        if self._add_constant:
            return np.append(X, np.ones_like(X[:, 0:1]), axis=1)
        else:
            return X


class PolyFeatures(Featurizer):
    def __init__(self, degree=2):
        self._scaler = self._scaler = Pipeline([('pre_scale', MinMaxScaler()),
                                                ('poly',
                                                 PolynomialFeatures(degree)),
                                                (
                                                    'after_scale',
                                                    MinMaxScaler())])
        self._n_features = None

    def _transform(self, X):
        if self.is_initialized():
            return self._scaler.transform(X)
        else:
            r = self._scaler.fit_transform(X)
            self._n_features = self._scaler.named_steps[
                'poly'].n_output_features_
            return r


class GaussianKernelFeatures(Featurizer):
    def __init__(self, n_kernel_fcts=10):
        self._n_kernel_fcts = n_kernel_fcts
        self._n_features = None

    def _transform(self, X):
        if not self.is_initialized():
            # fit a (spherical) Gaussian mixture model to estimate kernel params
            gmix = GaussianMixture(n_components=self._n_kernel_fcts,
                                   covariance_type="spherical", max_iter=100,
                                   random_state=0)

            gmix.fit(X)

            kernels = []
            for k in range(self._n_kernel_fcts):
                kernels.append(
                    (np.atleast_2d(gmix.means_[k]), gmix.precisions_[k]))

            self._n_features = self._n_kernel_fcts

        transformed = list()
        for kernel in kernels:
            gamma = None if X.shape[1] > 10 else kernel[
                1]  # only use precision for low-dim X
            shift = sklearn.metrics.pairwise.rbf_kernel(X, kernel[0], gamma)
            transformed += [shift]

        transformed = np.hstack(transformed)
        return transformed


class GMM(AbstractBaseline):
    models = {
        "linear": lambda input_dim: torch.nn.Linear(input_dim, 1),
        "2-layer": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 1)
        ),
        "mnist": lambda input_dim: DefaultCNN(cuda=torch.cuda.is_available())
    }

    def __init__(self, g_model="linear", f_feature_mapping=None,
                 g_feature_mapping=None, n_steps=1, g_epochs=200):
        '''
        Generalized methods of moments.
        - g_model: Model to estimate for g
        - f_feature_mapping: mapping of raw instruments z
        - g_feature_mapping: mapping of raw features x
        - norm: additional information
        '''
        super().__init__()

        if f_feature_mapping is None:
            self.f_mapping = VanillaFeatures()
        else:
            self.f_mapping = f_feature_mapping

        if g_feature_mapping is None:
            self.g_mapping = VanillaFeatures(add_constant=False)
        else:
            self.g_mapping = g_feature_mapping

        if g_model in self.models:
            self._g = self.models[g_model]
        else:
            raise ValueError("g_model has invalid value " + str(g_model))
        self._optimizer = None
        self._n_steps = n_steps
        self._g_epochs = g_epochs

    def display(self):
        for name, param in self._model.named_parameters():
            print(name, self.arr2str(param.data.cpu().numpy()))

    def fit_g_minibatch(self, train, loss):
        losses = list()
        for i, (x_b, y_b, z_b) in enumerate(train):
            if torch.cuda.is_available():
                x_b = x_b.cuda()
                y_b = y_b.cuda()
                z_b = z_b.cuda()
            loss_val = self._optimizer.step(lambda: loss(x_b, y_b, z_b))
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_g_batch(self, x, y, z, loss):
        _ = self._optimizer.step(lambda: loss(x, y, z))

    def _fit(self, x, y, z, context=None):
        z = self.augment(z, context)
        z = self.f_mapping.transform(z)
        x = self.augment(x, context)
        x = self.g_mapping.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()

        n_samples = x.size(0)
        x_dim, z_dim = x.size(1), z.size(1)

        g_model = self._g(x_dim)
        if torch.cuda.is_available():
            g_model = g_model.cuda()
        g_model.float()
        self._optimizer = torch.optim.Adam(g_model.parameters(), lr=0.01)
        weights = torch.eye(z_dim)
        if torch.cuda.is_available():
            weights = weights.cuda()
        self._model = g_model

        def loss(x_b, y_b, z_b):
            moment_conditions = z_b.mul(y_b - g_model(x_b))
            moms = moment_conditions.mean(dim=0, keepdim=True)
            loss = torch.mm(torch.mm(moms, weights), moms.t())
            self._optimizer.zero_grad()
            loss.backward()
            return loss

        batch_mode = "mini" if n_samples > 5000 else "full"
        t0 = time.time()
        train = data_utils.DataLoader(data_utils.TensorDataset(x, y, z),
                                      batch_size=128, shuffle=True)

        for step in range(self._n_steps):
            print("GMM step %d/%d" % (step + 1, self._n_steps))
            if step > 0:
                # optimize weights
                with torch.no_grad():
                    moment_conditions = z.mul(y - g_model(x))
                    covariance_matrix = torch.mm(moment_conditions.t(),
                                                 moment_conditions) / n_samples
                    weights = torch.as_tensor(
                        np.linalg.pinv(covariance_matrix.cpu().numpy(),
                                       rcond=1e-9))
                    if torch.cuda.is_available():
                        weights = weights.cuda()

            for epoch in range(self._g_epochs):
                if batch_mode == "full":
                    self.fit_g_batch(x, y, z, loss)
                else:
                    print("g epoch %d / %d" % (epoch + 1, self._g_epochs))
                    self.fit_g_minibatch(train, loss)
            self._model = g_model
        return time.time()-t0

    def _predict(self, x, context):
        x = self.augment(x, context)
        x = self.g_mapping.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        return self._model(x).data.cpu().numpy()


def main():
    def quick_scenario(n=1000, train=True):
        z = np.random.normal(size=(n, 2))
        context = np.zeros((n, 1))
        intercept = 0.0
        slope = 0.2
        g_true = lambda x: np.maximum(slope * x + intercept,
                                      slope * x / 0.2 + intercept)
        epsilon, eta = np.random.normal(size=(n, 1)), np.random.normal(
            size=(n, 1))
        x = z[:, 0:1] + z[:, 1:] + epsilon * 2.0
        y = g_true(x) + epsilon * 7.0 + eta / np.sqrt(2)
        y_true = g_true(x)
        return x, y, y_true, z, context

    np.random.seed(1)
    torch.manual_seed(1)

    x, y, _, z, context = quick_scenario()
    x_t, y_t_observed, y_t, _, context_t = quick_scenario(train=False)

    def eval(model):
        y_pred = model.predict(x_t, context_t)
        return ((y_pred - y_t) ** 2).mean()

    def save(model):
        os.makedirs("quick_scenario", exist_ok=True)
        y_pred = model.predict(x_t, context_t)
        np.savez("quick_scenario/" + type(model).__name__, x=x_t,
                 y=y_t_observed, g_true=y_t, g_hat=y_pred)

    for method in [DirectPoly(), DirectLinearRegression(),
                   GMM(f_feature_mapping=PolyFeatures(),
                       g_feature_mapping=PolyFeatures()), Vanilla2SLS(),
                   Poly2SLS()]:
        model = method.fit(x, y, z, context)

        print("Test MSE of %s: %f" % (type(model).__name__, eval(model)))




if __name__ == "__main__":
    main()
