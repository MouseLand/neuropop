import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD as SVD


def CCA(x1, x2, lam=1):
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

def reduced_rank_regression(Y, X, lam = .000, nPCs=-1):
    CXX = (X.T @ X)/X.shape[0]
    if nPCs<0:
        nPCs = min(X.shape[0], X.shape[1])
    model = SVD(n_components=nPCs).fit(CXX)
    U = model.components_
    sv = model.singular_values_

    V = (U @ X.T) / sv[:, np.newaxis]**.5
    X = U * sv[:, np.newaxis]**.5


    CXXMH = U @ (U.T / (sv + lam)**.5)
    CYX = (Y @ X.T)/X.shape[0]

    M = CYX @ CXXMH

    d,ss,c = np.linalg.svd(M, full_matrices = False)
    b = CXXMH @ c @ V
    a = d * ss

    return a, b
