import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d

def bin1d(X, bin_size, axis=0):
    """ mean bin over axis of data with bin bin_size """
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = Xb[:size[axis]//bin_size*bin_size].reshape((size[axis]//bin_size, bin_size, -1)).mean(axis=1)
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X
    
def compute_varexp(y_true, y_pred):
    """ variance explained of y_true by y_pred across axis=0
    
    """
    y_var = ((y_true - y_true.mean(axis=0)) ** 2).mean(axis=0)
    residual = ((y_true - y_pred) ** 2).mean(axis=0)
    varexp = 1 - residual / y_var
    return varexp 

def resample_frames(data, torig, tout):
    """
    Resample data from times torig at times tout.
    data is (n_samples, n_features). The data is filtered using a gaussian filter before resampling.
    
    Parameters
    ----------
    data : 2D array, input data (n_samples, n_features)
    torig : 1D-array, original times
    tout : 1D-array, times to resample to
    Returns
    --------
    dout : ND-array
        data resampled at tout
    """
    fs = torig.size / tout.size  # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs / 4), axis=0)
    f = interp1d(torig, data, kind="linear", axis=0, fill_value="extrapolate")
    dout = f(tout)
    return dout

def resample_data(data, tcam, tneural, crop='linspace'):
    """
    Resample data from camera times tcam at times tneural
    sometimes there are fewer camera timestamps than frames, so data is cropped
    data is (n_samples, n_features). The data is filtered using a gaussian filter before resampling.
    
    Parameters
    ----------
    data : 2D array, input data (n_samples, n_features)
    tcam : 1D-array, original times
    tneural : 1D-array, times to resample to
    Returns
    --------
    data_resampled : ND-array
        data resampled at tout
    """
    ttot = len(data)
    tc = len(tcam)
    if crop=='end':
        d = data[:tc]
    elif crop=='start':
        d = data[ttot-tc:]
    elif crop=='linspace':
        d = data[np.linspace(0,ttot-1, tc).astype(int)]
    else:
        d = data[(ttot-tc)//2:(ttot-tc)//2+tc]
    data_resampled = resample_frames(d, tcam, tneural)
    return data_resampled


def KLDiv_discrete(P, Q, binsize=200):
    # Q is the null distribution; P and Q are 2D distributions
    
    x_bins = np.append(np.arange(0, np.amax(P[:,0]), binsize), np.amax(P[:,0]))
    y_bins = np.append(np.arange(0, np.amax(P[:,1]), binsize), np.amax(P[:,1]))

    this_KL = 0
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            Qx = (np.sum((Q[:,0] >= x_bins[i]) & (Q[:,0] < x_bins[i+1]) & \
                        (Q[:,1] >= y_bins[j]) & (Q[:,1] < y_bins[j+1]))) / len(Q)
            Px = (np.sum((P[:,0] >= x_bins[i]) & (P[:,0] < x_bins[i+1]) & \
                        (P[:,1] >= y_bins[j]) & (P[:,1] < y_bins[j+1]))) / len(P)
            if (Px == 0) | (Qx == 0): # no points in test or null distrib -- can't have log(0), or /0
                continue

            this_KL += Px * np.log(Px / Qx)
    
    return this_KL
