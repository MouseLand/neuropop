"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def fit_asymptote(x, y, xall, fitexp=False):
    from sklearn.linear_model import LinearRegression
    ''' fit y = alpha + beta / sqrt(x)'''
    xi = x.copy()**-0.5
    if xi.ndim < 2:
        xi = xi[:,np.newaxis]
        xall = xall[:,np.newaxis]
    reg = LinearRegression().fit(xi, y)
    beta = reg.coef_
    alpha = reg.intercept_
    r2 = reg.score(xi, y)
    if not fitexp:
        ypred = alpha + np.dot(xall**-0.5, beta)
        par = [alpha]
        for b in beta:
            par.append(b)
        return par, r2, ypred
    if xi.shape[1]==1:
        par0 = [alpha, beta[0], 0.5]
        f = asymp
    else:
        par0 = [alpha, beta[0], beta[1], 0.5, 0.5]
        f = asymp2
    par, mcov = curve_fit(f, x, y, par0)

    if xi.shape[1]==1:
        ypred = asymp(x, par[0], par[1], par[2])
    else:
        ypred = asymp2(x.T, par[0], par[1], par[2], par[3], par[4])
    r2 = np.corrcoef(ypred, y)[0,1]
    print(par, r2)
    if xi.shape[1]==1:
        ypred = asymp(xall, par[0], par[1], par[2])
    else:
        ypred = asymp2(xall.T, par[0], par[1], par[2], par[3], par[4])
    return par, r2**2, ypred

def asymp(x, alpha, beta, t1):
    y = alpha + beta / x**t1
    return y

def asymp2(x, alpha, beta, gamma, t1, t2):
    y = alpha + beta / x[0]**t1 + gamma / x[1]**t2
    return y

def discrimination_threshold(P, x):
    P = (P + 1-P[::-1])/2
    par0 = np.array([5])
    par, mcov = curve_fit(logistic, x, P, par0)
    p75 = - np.log(1/0.75 - 1) * par[0]
    return p75, logistic(x, par)

# psychometric function
def logistic(x, beta):
    return 1. / (1 + np.exp( -x / beta ))

def get_powerlaw(ss, trange):
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred

def shuff_cvPCA(X, nshuff=10):
    ''' X is 2 x stimuli x neurons '''
    nc = min(1024, X.shape[1])

    nr = X.shape[0]

    ss=np.zeros((nshuff,nc))
    for k in range(nshuff):
        rperm = np.random.rand(X.shape[1])
        iflip = rperm > 0.5
        #X0 = np.float64(X.copy())
        X0 = np.roll(X, k, axis=0)

        for t in range(nr):
            X0[t,iflip] = X[(t+1)%nr,iflip]
            #X0[1,iflip] = X[0,iflip]

        ss[k]=cvPCA(X0)
    return ss

def repscvPCA(A,B, nshuff=10):
    NC, NN = A.shape
    ss = np.zeros((nshuff, NC))
    for n in range(nshuff):
        ss[n] = scvPCA(A,B)
    return ss

def scvPCA(A, B):
    """ A, B are neurons x stimuli, NC is # of eigenvalues to return """
    NC, NN = A.shape

    rperm = np.random.permutation(NN)

    A1 = A[:,rperm[:NN//2]]
    B1 = B[:,rperm[:NN//2]]

    A2 = A[:,rperm[NN//2:]]
    B2 = B[:,rperm[NN//2:]]

    covAB = A1 @ B1.T
    u,s,v = np.linalg.svd(covAB, full_matrices=False)
    covAB2 = A2 @ B2.T
    e_AB = np.sum(u  * (covAB2 @ u), axis=0)
    return e_AB

def cvPCA(X):
    ''' X is 2 x stimuli x neurons '''
    nr = X.shape[0]
    pca = PCA(n_components=min(1024, X.shape[1])).fit(X[0])
    #u = pca.components_.T
    #sv = pca.singular_values_
    #xproj = X[0].T @ (u / sv)

    xproj = pca.components_.T
    cproj0 = X[-2] @ xproj
    cproj1 = X[-1] @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss

def SVCA(X):
    from sklearn.decomposition import PCA
    # compute power law
    # SVCA
    #X -= X.mean(axis=1)[:,np.newaxis]

    NN,NT = X.shape

    # split cells into test and train
    norder = np.random.permutation(NN)
    nhalf = int(norder.size/2)
    ntrain = norder[:nhalf]
    ntest = norder[nhalf:]

    # split time into test and train
    torder = np.random.permutation(NT)
    thalf = int(torder.size/2)
    ttrain = torder[:thalf]
    ttest = torder[thalf:]
    #if ntrain.size > ttrain.size:
    #    cov = X[np.ix_(ntrain, ttrain)].T @ X[np.ix_(ntest, ttrain)]
    #    u,sv,v = svdecon(cov, k=min(1024, nhalf-1))
    #    u = X[np.ix_(ntrain, ttrain)] @ u
    #    u /= (u**2).sum(axis=0)**0.5
    #    v = X[np.ix_(ntest, ttrain)] @ v
    #    v /= (v**2).sum(axis=0)**0.5
    #else:
    cov = X[np.ix_(ntrain, ttrain)] @ X[np.ix_(ntest, ttrain)].T
    u = PCA(n_components=min(1024, nhalf-1), svd_solver='randomized').fit_transform(cov)
    u /= (u**2).sum(axis=0)**0.5
    v = cov.T @ u
    v /= (v**2).sum(axis=0)**0.5

    strain = u.T @ X[np.ix_(ntrain,ttest)]
    stest = v.T @ X[np.ix_(ntest,ttest)]

    # covariance k is uk.T * F * G.T * vk / npts
    scov = (strain * stest).mean(axis=1)
    varcov = (strain**2 + stest**2).mean(axis=1) / 2

    return scov, varcov
