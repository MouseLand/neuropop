"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.stats import zscore

def nanmedian_filter(x, win=7):
    """ nanmedian filter array along last axis"""
    nt = x.shape[-1]
    # pad so that x will be divisible by win
    pad = (win - (nt + 2*(win//2)) % win) % win
    xpad = np.pad(x, (win//2, win//2+win+pad), mode='edge')
    xmed = np.zeros_like(x)
    for k in range(win):
        xm = np.nanmedian(xpad[k:k-win].reshape(-1, win), axis=-1)
        xmed[...,k::win] = xm[:len(np.arange(k,nt,win))]
    return xmed

def filter_outliers(x, y, filter_window = 15, baseline_window = 50, max_spike = 50, max_diff = 50):

    # remove frames with large jumps
    x_diff = np.abs(np.append(np.zeros(1,), np.diff(x)))
    y_diff = np.abs(np.append(np.zeros(1,), np.diff(y)))
    replace_inds = np.logical_or(x_diff > max_diff, y_diff > max_diff)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    
    # remove frames with large deviations from baseline
    x_baseline = nanmedian_filter(x, baseline_window)
    y_baseline = nanmedian_filter(y, baseline_window)
    replace_inds = np.logical_or(np.abs(x - x_baseline) > max_spike, np.abs(y - y_baseline) > max_spike)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    replace_inds = np.isnan(x)
    
    # filter x and y
    x_filt = nanmedian_filter(x, filter_window)
    y_filt = nanmedian_filter(y, filter_window)
    
    # this in theory shouldn't add more frames
    replace_inds = np.logical_or(replace_inds, np.isnan(x_filt))
    ireplace = np.nonzero(replace_inds)[0]

    # replace outlier frames with median
    if len(ireplace) > 0:
        # good indices
        iinterp = np.nonzero(np.logical_and(~replace_inds, ~np.isnan(x_filt)))[0]
        x[replace_inds] = np.interp(ireplace, iinterp, x_filt[iinterp])
        y[replace_inds] = np.interp(ireplace, iinterp, y_filt[iinterp])

    if 0:
        # replace overall outlier deflections from baseline
        x_baseline = x.mean() #nanmedian_filter(x, baseline_window)
        y_baseline = y.mean() #nanmedian_filter(y, baseline_window)
        max_spike = x.std() * 5, y.std() * 5
        replace_inds = np.logical_or(np.abs(x - x_baseline) > max_spike[0], 
                                    np.abs(y - y_baseline) > max_spike[1])
        x[replace_inds] = x.mean()#_baseline[replace_inds]
        y[replace_inds] = y.mean()#_baseline[replace_inds]
    
    return x,y 


def keypoints_features(xy):
    xy_vel = (np.diff(xy, axis=0)**2).sum(axis=-1)**0.5
    xy_vel = np.vstack((xy_vel[[0]], xy_vel))
    xy_rad = ((xy - xy.mean(axis=0))**2).sum(axis=-1)**0.5
    xy_dists = compute_dists(xy)
    return xy, xy_vel, xy_rad, xy_dists

def compute_dists(xy):
    xy_dists = ((xy[:,:,np.newaxis] - xy[:,np.newaxis])**2).mean(axis=-1)
    upper_triangular = np.triu_indices(xy_dists.shape[-1], 1)
    xy_dists = xy_dists[:, upper_triangular[0], upper_triangular[1]]
    return xy_dists

def gabor_wavelet(sigma, f, ph, n_pts=201, is_torch=False):
    x = np.linspace(0, 2*np.pi, n_pts+1)[:-1].astype('float32')
    cos = np.cos
    sin = np.sin
    exp = np.exp
    xc = x - x.mean()
    cosine = cos(ph + f * xc)
    gaussian = exp(-(xc**2) / (2*sigma**2))
    G = gaussian * cosine
    G /= (G**2).sum()**0.5
    return G

def get_gabor_transform(data, freqs=np.geomspace(1, 10, 5)):
    """ data is time points by features """
    n_time, n_features = data.shape
    n_widths = len(freqs)
    gabor_transform = np.zeros((n_time, 2*n_widths, n_features), 'float32')    
    for k,f in enumerate(freqs):
        gw0 = gabor_wavelet(1,f,0)
        gw1 = gabor_wavelet(1,f,np.pi/2)
        for j in range(n_features):
            filt0 = np.convolve(zscore(data[:,j]), gw0, mode='same')
            filt1 = np.convolve(zscore(data[:,j]), gw1, mode='same')  
            gabor_transform[:,2*k,j] = filt0
            gabor_transform[:,2*k+1,j] = (filt0**2 + filt1**2)**0.5
    return gabor_transform