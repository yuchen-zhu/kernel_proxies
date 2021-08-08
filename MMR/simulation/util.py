import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def extract_1d_var(**kwargs):
    name_var1d = []
    for var_name, var in kwargs.items():
        if len(var.shape) == 2:
            for i in range(var.shape[-1]):
                name_var1d.append((var_name+str(i), var[:, i]))
        elif len(var.shape) == 1:
            name_var1d.append((var_name, var))
        else:
            raise ValueError('var has to be 1d or 2d array.')
    return name_var1d


def calculate_correlation_coeff(PATH, save_name, **kwargs):
    """
    calculates the pairwise total correlation coefficient between variables and save them as a csv file.

    Args:
        PATH: path to directory of saving the output file. str
        save_name: name with which to save the output file. str
        **kwargs: input variables for which we want to calculate the pairwise correlation.
            e.g. kwargs = {'a': np.1darray, 'b': np.2darray} will calculate the pairwise correlation among
            the 1d variable a and each dimension of the 2d variable b. Variable data must be either 1d or 2d numpy arrays.
    """

    name_var1d = extract_1d_var(**kwargs)

    name_list, var1d_list = [], []
    for var1d_name, var1d in name_var1d:
        name_list.append(var1d_name)
        var1d_list.append(var1d)
    df = pd.DataFrame(var1d_list).T
    df.columns = name_list
    corr = df.corr(method='pearson')
    corr.to_csv(os.path.join(PATH, save_name + '.csv'), sep=' ', mode='a')


def calculate_partial_correlation_coeff(PATH, save_name, **kwargs):
    """
    calculates the pairwise PARTIAL correlation coefficient between variables and save them as a csv file.

    Args:
        PATH: path to directory of saving the output file. str
        save_name: name with which to save the output file. str
        **kwargs: input variables for which we want to calculate the pairwise correlation.
            e.g. kwargs = {'a': np.1darray, 'b': np.2darray} will calculate the PARTIAL correlation among
            the 1d variable a and each dimension of the 2d variable b. Variable data must be either 1d or 2d numpy arrays.
    """
    name_var1d = extract_1d_var(**kwargs)

    name_list, var1d_list = [], []
    for var1d_name, var1d in name_var1d:
        name_list.append(var1d_name)
        var1d_list.append(var1d)
    df = pd.DataFrame(var1d_list).T
    df.columns = name_list
    corr = df.pcorr()
    corr.to_csv(os.path.join(PATH, save_name + '.csv'), sep=' ', mode='a')


def check_r2(covar, outcome):
    """
    Fits a linear regression from covar to outcome using OLS, then report R2.
    Args:
        covar - dictionary for covariates.
        outcome - dictionary with a single key for outcome.
    """

    covar_name_var1d, outcome_name_var1d = extract_1d_var(**covar), extract_1d_var(**outcome)

    y = np.array([var[-1] for var in outcome_name_var1d]).squeeze()
    X = np.array([var[-1] for var in covar_name_var1d]).squeeze().T
    print(X.shape, y.shape)
    f = sm.OLS(y, X).fit()
    print('the r2 for fitting {} to {} is {}.'.format([name for name in covar.keys()],
                                                      [name for name in outcome.keys()], f.rsquared))
    return f.rsquared

