import numpy as np 
import torch

from neuropop.utils import compute_varexp, bin1d, resample_data
from neuropop.split_data import split_traintest


def ridge_regression(X, Y, lam=0):
    """predict Y from X using regularized linear regression using torch arrays
    *** subtract mean from X and Y before predicting
    Prediction:
    >>> Y_pred = X @ A
    Parameters
    ----------
    X : 2D array, input data (n_samples, n_features)
    Y : 2D array, data to predict (n_samples, n_predictors)
    Returns
    --------
    A : 2D array - prediction matrix 1 (n_predictors, rank)
    """
    CXX = (X.T @ X + lam * np.eye(X.shape[1], dtype="float32")) / X.shape[0]
    CXY = (X.T @ Y) / X.shape[0]
    A = torch.linalg.solve(CXX, CXY).T
    return A

def reduced_rank_regression(X, Y, rank=None, lam=0):
    """predict Y from X using regularized reduced rank regression using torch arrays
    *** subtract mean from X and Y before predicting
    if rank is None, returns A and B of full-rank (minus one) prediction
    Prediction:
    >>> Y_pred = X @ B @ A.T
    Parameters
    ----------
    X : 2D array, input data, float32 torch tensor (n_samples, n_features)
    Y : 2D array, data to predict, float32 torch tensor (n_samples, n_predictors)
    rank : int (optional, default None)
        rank to compute reduced rank regression for
    lam : float (optional, default 0)
        regularizer
    Returns
    --------
    A : 2D array - prediction matrix 1 (n_predictors, rank)
    B : 2D array - prediction matrix 2 (n_features, rank)
    """
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)

    # make covariance matrices
    CXX = (X.T @ X + lam * torch.eye(X.shape[1], device=X.device)) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    # s, u = eigh(CXX.cpu().numpy())
    u, s = torch.svd_lowrank(CXX, q=rank)[:2]
    CXXMH = (u * (s + lam) ** -0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH
    # do svd of prediction projection
    # model = PCA(n_components=rank).fit(M)
    # c = model.components_.T
    # s = model.singular_values_
    s, c = torch.svd_lowrank(M, q=rank)[1:]
    A = M @ c
    B = CXXMH @ c
    return A, B


def linear_prediction(X, Y, rank=None, lam=0, allranks=True, itrain=None, itest=None, tbin=None, device=torch.device("cpu")):
    """predict Y from X using regularized regression
    *** user needs to subtract mean from X and Y before predicting ***
    
    if rank is None, performs ridge regression, otherwise performs reduced rank regression
    
    Prediction:
    >>> Y_pred_test = X_test @ B @ A.T
    Parameters
    ----------
    X : 2D array, input data, float32 (n_samples, n_features)
    Y : 2D array, data to predict, float32 (n_samples, n_predictors)
    rank : int (optional, default None)
        rank up to which to compute reduced rank regression for
    lam : float (optional, default 0)
        regularizer
    allranks : bool (optional, default True)
        compute variance explained at all ranks
    itrain: 1D int array (optional, default None)
        times in train set
    itest: 1D int array (optional, default None)
        times in test set
    tbin: int (optional, default None)
        also compute variance explained in bins of tbin
    Returns
    --------
    Y_pred_test : 2D array - prediction of Y with max rank (len(itest), n_features)
    varexp : 1D array - variance explained across all features (rank,)
    itest: 1D int array
        times in test set
    A : 2D array - prediction matrix 1 (n_predictors, rank)
    B : 2D array - prediction matrix 2 (n_features, rank)
    varexpf : 1D array - variance explained per feature (rank, n_features)
    corrf : 1D array - correlation with Y per feature (rank, n_features)

    """
    n_t, n_feats = Y.shape
    if itrain is None and itest is None:
        itrain, itest = split_traintest(n_t)
    itrain, itest = itrain.flatten(), itest.flatten()
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    if rank is not None:
        min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
        rank = min(min_dim, rank)
        A, B = reduced_rank_regression(
            X[itrain], Y[itrain], rank=rank, lam=lam
        )
    else:
        A = ridge_regression(X[itrain], Y[itrain], lam=lam)
        B = None
        allranks = False
        rank = 1

    corrf = np.zeros((rank, n_feats))
    varexpf = np.zeros((rank, n_feats))
    varexp = np.zeros((rank, 2)) if (tbin is not None and tbin > 1) else np.zeros((rank, 1))
    Y_pred_test = np.zeros((len(itest), n_feats))
    for r in range(0 if allranks else rank-1, rank):
        if B is not None:
            Y_pred_test = X[itest] @ B[:, : r + 1] @ A[:, : r + 1].T
        else:
            Y_pred_test = X[itest] @ A.T
        Y_test_var = (Y[itest] ** 2).mean(axis=0)
        corrf[r] = ((Y[itest] * Y_pred_test).mean(axis=0) / 
                    (Y_test_var ** 0.5 * Y_pred_test.std(axis=0))).cpu().numpy()
        residual = ((Y[itest] - Y_pred_test) ** 2).mean(axis=0)
        varexpf[r] = (1 - residual / Y_test_var).cpu().numpy()
        varexp[r, 0] = (1 - residual.mean() / Y_test_var.mean()).cpu().numpy()
        if tbin is not None and tbin > 1:
            varexp[r, 1] = compute_varexp(bin1d(Y[itest], tbin).flatten(), bin1d(Y_pred_test, tbin).flatten()).cpu().numpy()
    if not allranks:
        varexp, varexpf, corrf = varexp[-1:], varexpf[-1:], corrf[-1:]
    if B is not None:
        B = B.cpu().numpy()
    return (Y_pred_test.cpu().numpy(), varexp.squeeze(), itest, 
            A.cpu().numpy(), B, varexpf.squeeze(), corrf.squeeze())

def prediction_wrapper(X, Y, tcam=None, tneural=None, U=None, spks=None, delay=0, tbin=None, rank=32, device=torch.device('cuda')):
    """ predict neurons or neural PCs Y and compute varexp for Y and/or spks"""
    
    X -= X.mean(axis=0)
    X /= X[:,0].std(axis=0)

    if tcam is not None and tneural is not None:
        X_ds = resample_data(X, tcam, tneural, crop='linspace')
    else:
        X_ds = X

    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1],:], (-delay,1))))
    else:
        X_ds = np.vstack((X_ds[delay:], np.tile(X_ds[[-1],:], (delay,1))))
        Ys = Y
    
    Y_pred_test, ve_test, itest, A, B = linear_prediction(X_ds, Ys, rank=rank, lam=1e-6, tbin=tbin, device=device)[:5]
    varexp = ve_test
    # return Y_pred_test at specified rank
    Y_pred_test = X_ds[itest] @ B[:,:rank] @ A[:,:rank].T

    # single neuron prediction
    if U is not None and spks is not None:
        spks_pred_test = Y_pred_test @ U.T 
        spks_test = spks[:, itest-delay].T
        varexp_neurons = np.nan * np.zeros((len(spks), 2 if tbin is not None and tbin>1 else 1))
        varexp_neurons[:,0] = compute_varexp(spks_test, spks_pred_test)
        if tbin is not None and tbin > 1:
            spks_test_bin = bin1d(spks_test, tbin)
            spks_pred_test_bin = bin1d(spks_pred_test, tbin)
            varexp_neurons[:,1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
        spks_pred_test0 = spks_pred_test.copy()
        
        return varexp.squeeze(), varexp_neurons.squeeze(), spks_pred_test0, itest
    else:
        return varexp.squeeze(), None, None, itest


def CCA(x1, x2, lam=1):
    from sklearn.decomposition import TruncatedSVD as SVD
    n_comp = np.min(x1.shape)-1
    model1 = SVD(n_components = min(1000, n_comp)).fit(x1)
    n_comp = np.min(x2.shape)-1
    model2 = SVD(n_components = min(1000, x2.shape[1]-1)).fit(x2)

    U0 = model1.components_
    V0 = U0 @ x1.T

    print(U0.shape, V0.shape)

    U1 = model2.components_
    V1 = U1 @ x2.T

    S0 = np.sum(V0**2, axis=1)**.5
    V0 = V0/S0[:, np.newaxis]

    S1 = np.sum(V1**2, axis=1)**.5
    V1 = V1/S1[:, np.newaxis]

    lam = lam * x1.shape[-1]

    VVT = V0 @ V1.T
    W0 = U0.T * S0/(S0**2 + lam)**.5
    W1 = U1.T * S1/(S1**2 + lam)**.5

    CC = W0 @ (VVT @ W1.T)
    CC = CC + CC.T

    model3 = SVD(n_components = min(100, CC.shape[0]-1)).fit(CC)

    u = model3.components_
    print(u.shape)
    v = u @ CC.T
    v = v / np.sum(v**2, axis=1)[:, np.newaxis]**.5
    u = u @ (U0.T /(S0**2 + lam)**.5) @ U0
    v = v @ (U1.T /(S1**2 + lam)**.5) @ U1

    return u,v