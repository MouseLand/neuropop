"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np 
import torch
from torch.nn.functional import conv1d
from utils import compute_varexp

def causal_filter(X, swave, tlag, remove_start=False, device=torch.device('cuda')):
    """ filter matrix X (n_channels, (n_batches,) n_time) with filters swave
    
    returns Xfilt (n_out, n_batches*n_time)
    """
    if X.ndim < 3:
        X = X.unsqueeze(1)
    NT = X.shape[-1]
    nt =  swave.shape[1]
    # reshape X for input to be (n_channels*n_batches, 1, n_time)
    Xfilt = conv1d(X.reshape(-1, X.shape[-1]).unsqueeze(1), 
                   swave.unsqueeze(1), padding=nt+tlag)
    Xfilt = Xfilt[..., :NT]
    Xfilt = Xfilt[..., nt:] if remove_start else Xfilt
    Xfilt = Xfilt.reshape(X.shape[0], X.shape[1], swave.shape[0], -1)
    Xfilt = Xfilt.permute(0,2,1,3)
    Xfilt = Xfilt.reshape(X.shape[0]*swave.shape[0], X.shape[1], -1)
    return Xfilt

def fit_causal_prediction(X_train, X_test, swave, lam = 1e-3, tlag=1, device=torch.device('cuda')):
    """ predict X in the future with exponential filters"""
    # fit on train data
    Xfilt = causal_filter(X_train, swave, tlag)    
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    NT = X_train.shape[1] * X_train.shape[2]
    nff = Xfilt.shape[0]
    CC = (Xfilt @ Xfilt.T)/NT + lam * torch.eye(nff, device = device)
    CX = (Xfilt @ X_train.reshape(-1,NT).T) / NT
    B = torch.linalg.solve(CC, CX)    

    # performance on test data
    Xfilt = causal_filter(X_test, swave, tlag, remove_start=True)    
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    ypred = B.T @ Xfilt
    nt = swave.shape[1]
    ve = compute_varexp(X_test[:,:,nt:].reshape(X_test.shape[0],-1).T, ypred.T)
    return ve, ypred, B

def future_prediction(X, Ball, swave, device=torch.device('cuda')):
    """ create future prediction """
    tlag = Ball.shape[-1]
    Xfilt = causal_filter(X, swave, tlag, remove_start=True)
    vef = np.zeros((X.shape[0], tlag))
    nt = swave.shape[1]
    Xpred = np.zeros((X.shape[0], X.shape[1], X.shape[2]-nt, tlag))
    for k in range(tlag):
        Xfilt0 = Xfilt[:,:,tlag-k:].reshape(Xfilt.shape[0], -1)
        B = torch.from_numpy(Ball[:,:,k]).to(device)
        ypred = (B.T @ Xfilt0)
        ve = compute_varexp(X[:,:,nt:-(tlag-k)].reshape(X.shape[0],-1).T, 
                                        ypred.T)
        ypred = ypred.reshape(X.shape[0], X.shape[1], -1)
        vef[:,k] = ve.cpu().numpy()
        Xpred[:,:,:-(tlag-k),k] = ypred.cpu().numpy()
    return vef, Xpred

def predict_future(x, keypoint_labels=None, get_future=True, lam=1e-3, device=torch.device('cuda')):
    """ predict keypoints or latents in future
    
    x is (n_time, n_keypoints) and z-scored per keypoint
    
    """
    nt = 128
    sigs = torch.FloatTensor(2**np.arange(0,8,1)).unsqueeze(-1)
    swave = torch.exp( - torch.arange(nt) / sigs).to(device)
    swave = torch.flip(swave, [1])
    swave = swave / (swave**2).sum(1, keepdim=True)**.5

    tlags = np.arange(1, 501, 1)
    tlags = np.append(tlags, np.arange(525, 2000, 25))

    X = torch.from_numpy(x.T).float().to(device)

    itrain, itest = split_traintest(len(x), frac=0.25, pad=nt)

    X_train = X[:,itrain]
    X_test = X[:,itest]

    n_kp = X_train.shape[0]
    n_tlags = len(tlags)
    vet = np.zeros((n_kp, n_tlags), 'float32')
    Ball = np.zeros((swave.shape[0]*n_kp, n_kp, n_tlags), 'float32')
    for k,tlag in enumerate(tlags):
        ve, ypred, B = fit_causal_prediction(X_train, X_test, swave, tlag=tlag, lam=lam)
        vet[:,k] = ve.cpu().numpy()
        Ball[:,:,k] = B.cpu().numpy()
        
    if get_future:
        vef, ypred = future_prediction(X_test, Ball[:,:,:500], swave)
    else:
        ypred = None

    if keypoint_labels is not None:
        # tile for X and Y
        kp_labels = np.tile(np.array(keypoint_labels)[:,np.newaxis], (1,2)).flatten()

        areas = ['eye', 'whisker', 'nose']
        vet_area = np.zeros((len(areas), vet.shape[1]))
        for j in range(len(areas)):
            ak = np.array([k for k in range(len(kp_labels)) if areas[j] in kp_labels[k]])
            vet_area[j] = vet[ak].mean(axis=0)
    else:
        vet_area = None

    return vet, vet_area, tlags, ypred, itest[:, nt:]