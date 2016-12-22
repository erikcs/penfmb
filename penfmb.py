import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings

from arch.bootstrap import StationaryBootstrap

def _fmb(df_portfolios, df_factors, intercept=True):
    """ Simple 2 stage OLS
    returns the first stage and second stage coefficients
    """
    n_pf = df_portfolios.shape[1]
    mean_pf = df_portfolios.mean(axis=0)
    mean_pf.name = "mean"

    if isinstance(df_factors, pd.DataFrame):
        columns = df_factors.columns
    else:
        columns = [df_factors.name]

    if intercept:
        df_factors = sm.add_constant(df_factors)
        slice = 1
    else:
        slice = 0

    tsres = map(
            lambda col:
            sm.OLS(df_portfolios.iloc[:, col], df_factors).fit(),
            range(n_pf))
    tsbeta = pd.DataFrame(map(
            lambda x:
            tsres[x].params[slice:].values,
            range(n_pf)), columns=columns, index=mean_pf.index)

    xsres = sm.OLS(mean_pf, sm.add_constant(tsbeta)).fit()
    loadings = pd.concat([tsbeta, mean_pf], axis=1)

    # first stage object, second stage object, first stage coefficients
    return tsres, xsres, loadings

def _normalise(X, y):
    Xbar = X.mean(axis=0)
    Xstd = X.std(axis=0)
    ybar = y.mean(axis=0)
    ystd = y.std()
    X_norm = (X - Xbar) / Xstd
    y_cent = y - ybar

    return X_norm, y_cent, Xstd, Xbar, ybar, ystd

def _soft_threshold(x, value):
    return np.sign(x) * np.maximum(np.abs(x) - value, 0)

def _l1_weight(x, d):
    return 1 / np.abs(x).sum(axis=0)**d

def _coordinate_descent(X, y, alpha=0, alpha_weights=None, tol=1e-5, maxiter=1000):
    """ Runs coordinate descent on the objective

    1\2n||y - w0 - Xw||^2_2 + P(alpha, x, w)

    where

    P = alpha sum_{i=1}^{p} |w_i| * alpha_weights[i]

    i.e. Lasso where each l1 penalty coordinate is weighted according to
    `alpha_weights`

    Fitting is done on normalized data, but coefficients are returned
    on original scale.

    Parameters
    ----------
    X : ndarray, shape (n_observations, n_features)
    y : ndarray, shape (n_observations,)

    Returns
    -------
    w_rescaled: array, shape (n_features, )
        Estimated coefficients on original scale

    intercept : int
        Estimated (unpenalized) intercept on original scale

    w : array, shape (n_features)
        Estimated coefficients, normalized

    Notes
    -----
    Gives the exact same result as running
    glmnet(X, y, alpha = 1, lambda = alpha, penalty.factor = 1 / colSums(abs(X)) ^ 4))
    in R

    Reference:
    Hastie, T., Tibshirani, R., & Wainwright, M. (2015). Statistical learning
    with sparsity: the lasso and generalizations. CRC Press.

    (will only be applied to tiny `X` and `y` so should not be necessary to do
    in Cython)
    """
    n, p = X.shape
    w = np.zeros(p)
    wp = w.copy()
    X, y, Xstd, Xbar, ybar, ystd = _normalise(X, y)

    if alpha_weights is None:
        alpha_weights = np.ones(p)

    r = y - (X * w).sum(axis=1)
    for n_iter in range(maxiter):

        for i in range(p):
            x = X[:, i]
            w_ols = w[i] + np.dot(x, r) / n
            w[i] = _soft_threshold(w_ols, alpha * alpha_weights[i])
            r += wp[i] * X[:, i]
            r -= w[i] * X[:, i]

        converged = np.linalg.norm(wp - w) < tol
        if converged:
            break
        wp = w.copy()

    if not converged:
        warnings.warn("Coordinate descent did not converge. You might want "
                      "to increase `maxiter` or decrease `tol`")

    w_rescaled = w / Xstd
    intercept = ybar - np.sum(Xbar * w_rescaled)

    return w_rescaled, intercept, w

def _penfmb(df_loadings, alpha, d, tol, maxiter, alpha_weights=_l1_weight):
    X = df_loadings.iloc[:, 0:-1].values
    y = df_loadings.iloc[:, -1].values

    alpha_weights = alpha_weights(X, d)

    coefs, intercept, _ = _coordinate_descent(X, y, alpha, alpha_weights,
                                              tol, maxiter)

    idx = df_loadings.columns.tolist()[:-1]
    idx.insert(0, 'const')

    return pd.Series(np.hstack((intercept, coefs)), index=idx)

def _get_alpha(tsres):
    """Returns the average standard devation of the first state residuals
    """
    return np.mean(map(lambda x:
                     np.std(tsres[x].resid), range(len(tsres))))

class PenFMB():
    """The penalized Fama-MacBeth estimator, estimate

    ||y - w0 - Xw||^2_2 + P(alpha, x, w)

    where

    P = alpha sum_{i=1}^{p} |w_i| * 1 / ||z_i||^d_1

    z_i is the first stage estimate ('betas')

    Reference:
    S. Bryzgalova 'Spurious Factors in Linear Asset Pricing Models' (2015)

    Parameters
    ----------
    intercept : bool, default True
        Fit an intercept in the first stage regression

    d : int, default 4
        The curvature tuning parameter in the penalty

    alpha : float, default None
        The level tuning parameter, if not provided, set to the average of
        the standard deviation of the residuals from the first stage

    tol : float, optional

    maxiter : int, optional

    nboot : int, optional, default 500
        The number of stationary bootstrap iterations to estimate shrinkage rate

    block_length : int, optional, default 100
        Expected block length for the stationary bootstrap

    Attributes
    ----------
    coef_ : array
        The estimated second stage penalized coefficients
        and bootstrapped shrinkage rate (based on a stationary bootstrap
        in the first stage)
    """
    def __init__(self, intercept=True, d=4, alpha=None, tol=1e-5, maxiter=1000,
                 nboot=500, block_length=100):
        self.intercept = intercept
        self.d = d
        self.alpha = alpha
        self.maxiter = maxiter
        self.tol = tol
        self.nboot = nboot
        self.block_length = block_length

    def fit(self, df_portfolios, df_factors):
        """ Fit the estimator

        Parameters
        -----------
        df_portfolios : DataFrame
            Time series of portfolios (test assets)

        df_factors : DataFrame or Series
            Time series of the factors
        """
        tsres, _ , loadings = _fmb(df_portfolios, df_factors, self.intercept)
        self._tsres = tsres
        self.loadings = loadings

        if self.alpha is None:
            self.alpha = _get_alpha(self._tsres)

        self._xsres = _penfmb(loadings, self.alpha, self.d, self.tol, self.maxiter)
        self._xsres.name = 'coef'

        sbs = StationaryBootstrap(self.block_length, df_portfolios, df_factors)

        bsxsres = []
        for data in sbs.bootstrap(self.nboot):
            tsres , _, bloadings = _fmb(data[0][0].reset_index(drop=True),
                                        data[0][1].reset_index(drop=True),
                                        self.intercept)
            bsxsres.append(_penfmb(bloadings, _get_alpha(tsres), self.d,
                                   self.tol, self.maxiter))

        bsxsres = pd.DataFrame(bsxsres)
        self._srate = 1.0*(bsxsres == 0).sum(axis=0) / bsxsres.shape[0]
        self._srate.name = 'shrinkage rate'
        # self._se = bsxsres.std(axis=0)
        # self._se.name = 'standard error'

        return self

    @property
    def coefs_(self):
        return pd.DataFrame([self._xsres, self._srate]).T
