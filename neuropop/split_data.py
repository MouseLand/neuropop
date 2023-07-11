"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np 
from scipy.interpolate import interp1d
import torch

def split_traintest(n_t, frac=0.25, pad=3, split_time=False):
    """this returns deterministic split of train and test in time chunks
    
    Parameters
    ----------
    n_t : int
        number of timepoints to split
    frac : float (optional, default 0.25)
        fraction of points to put in test set
    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment
    split_time : bool (optional, default False)
        split train and test into beginning and end of experiment
    Returns
    --------
    itrain: 2D int array
        times in train set, arranged in chunks
    
    itest: 2D int array
        times in test set, arranged in chunks
    """
    #usu want 10 segs, but might not have enough frames for that
    n_segs = int(min(10, n_t/4)) 
    n_len = int(np.floor(n_t/n_segs))
    inds_train = np.linspace(0, n_t - n_len - 5, n_segs).astype(int)
    if not split_time:
        l_train = int(np.floor(n_len * (1-frac)))
        inds_test = inds_train + l_train + pad
        l_test = np.diff(np.stack((inds_train, inds_train + l_train)).T.flatten()).min() - pad
    else:
        inds_test = inds_train[:int(np.floor(n_segs*frac))]
        inds_train = inds_train[int(np.floor(n_segs*frac)):]
        l_train = n_len - 10
        l_test = l_train
    itrain = (inds_train[:,np.newaxis] + np.arange(0, l_train, 1, int))
    itest = (inds_test[:,np.newaxis] + np.arange(0, l_test, 1, int))
    return itrain, itest


def split_batches(tcam, tneural, frac=0.25, pad=3, split_time=False,
                  itrain=None, itest=None):
    """this returns deterministic split of train and test in time chunks for neural and cam times
    
    Parameters
    ----------
    n_t : int
        number of timepoints to split
    tcam : 1D array
        times of camera frames
    tneural : 1D array
        times of neural frames
    frac : float (optional, default 0.25)
        fraction of points to put in test set
    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment
    split_time : bool (optional, default False)
        split train and test into beginning and end of experiment
    itrain: 2D int array
        times in train set, arranged in chunks
    
    itest: 2D int array
        times in test set, arranged in chunks
    
    Returns
    --------
    itrain: 1D int array
        times in train set, arranged in chunks
    
    itest: 1D int array
        times in test set, arranged in chunks
    itrain_cam: 2D int array
        times in cam frames in train set, arranged in chunks
    itest_cam: 2D int array
        times in cam frames in test set, arranged in chunks
    """
    
    if itrain is None or itest is None:
        itrain, itest = split_traintest(len(tneural), frac=frac, pad=pad, split_time=split_time)
    inds_train, inds_test = itrain[:,0], itest[:,0]
    l_train, l_test = itrain.shape[-1], itest.shape[-1]
    
    # find itrain and itest in cam inds
    f = interp1d(tcam, np.arange(0, len(tcam)), kind='nearest', axis=-1,
                fill_value='extrapolate', bounds_error=False)

    inds_cam_train = f(tneural[inds_train]).astype('int')
    inds_cam_test = f(tneural[inds_test]).astype('int')

    l_cam_train = int(np.ceil(np.diff(tneural).mean() / np.diff(tcam).mean() * l_train))
    l_cam_test = int(np.ceil(np.diff(tneural).mean() / np.diff(tcam).mean() * l_test))

    # create itrain and itest in cam inds
    itrain_cam = (inds_cam_train[:,np.newaxis] + np.arange(0, l_cam_train, 1, int))
    itest_cam = (inds_cam_test[:,np.newaxis] + np.arange(0, l_cam_test, 1, int))
    
    itrain_cam = np.minimum(len(tcam)-1, itrain_cam)
    itest_cam = np.minimum(len(tcam)-1, itest_cam)

    # inds for downsampling itrain_cam and itest_cam
    itrain_sample = f(tneural[itrain.flatten()]).astype(int)
    itest_sample = f(tneural[itest.flatten()]).astype(int)
    
    # convert to indices in itrain_cam and itest_cam
    it = np.zeros(len(tcam), 'bool')
    it[itrain_sample] = True
    itrain_sample = it[itrain_cam.flatten()].nonzero()[0]
    
    it = np.zeros(len(tcam), 'bool')
    it[itest_sample] = True
    itest_sample = it[itest_cam.flatten()].nonzero()[0]

    return itrain, itest, itrain_cam, itest_cam, itrain_sample, itest_sample

def split_data(X, Y, tcam, tneural, frac=0.25, delay=-1, split_time=False, device=torch.device('cuda')):
    # ensure keypoints and timestamps are same length
    tc, ttot = len(tcam), len(X)
    inds = np.linspace(0, max(ttot,tc)-1, min(ttot,tc)).astype(int)
    X = X[inds] if ttot > tc else X 
    tcam = tcam[inds] if tc > ttot else tcam
    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1],:], (-delay,1))))
        Xs = X
    elif delay > 0:
        Xs = np.vstack((X[delay:], np.tile(X[[-1],:], (delay,1))))
        Ys = Y
    else:
        Xs = X 
        Ys = Y
    splits = split_batches(tcam, tneural, frac=frac, 
                            split_time=split_time)
    itrain, itest, itrain_cam, itest_cam, itrain_sample, itest_sample = splits
    X_train = torch.from_numpy(Xs[itrain_cam]).float().to(device)
    Y_train = torch.from_numpy(Ys[itrain]).float().to(device)
    X_test = torch.from_numpy(Xs[itest_cam]).float().to(device)
    Y_test = torch.from_numpy(Ys[itest]).float().to(device).reshape(-1, Y.shape[-1])
    
    itrain_sample_b = torch.zeros(itrain_cam.size, dtype=bool, device=device)
    itrain_sample_b[itrain_sample] = True
    itest_sample_b = torch.zeros(itest_cam.size, dtype=bool, device=device)
    itest_sample_b[itest_sample] = True
    itrain_sample_b = itrain_sample_b.reshape(itrain_cam.shape)
    itest_sample_b = itest_sample_b.reshape(itest_cam.shape)
    
    itest -= delay

    return X_train, X_test, Y_train, Y_test, itrain_sample_b, itest_sample_b, itrain_sample, itest_sample, itrain, itest
