"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np 
from neuropop.linear_prediction import linear_prediction

def peer_prediction(spks, xpos, ypos, dum=400, tbin=4):
    ineu1 = np.logical_xor((xpos%dum)<dum/2 , (ypos%dum) < dum/2)
    #ineu1 = np.random.rand(len(spks)) > 0.5
    ineu2 = np.logical_not(ineu1)
    n_components = 128
    Vn = []
    for ineu in [ineu1, ineu2]:
        Vn.append(PCA(n_components=n_components, copy=False).fit_transform(spks[ineu].T))
    varexp = np.zeros(2)
    varexp_neurons = np.zeros((spks.shape[0], 2))
    for k,ineu in enumerate([ineu1, ineu2]):
        V_pred_test,varexpk,itest = linear_prediction(Vn[(k+1)%2], Vn[k%2], rank=128, lam=1e-1, tbin=tbin)[:3]
        varexp += varexpk[-1]
        U = spks[ineu] @ Vn[k]
        U /= (U**2).sum(axis=0)**0.5
        spks_pred_test = V_pred_test @ U.T
        spks_test = spks[ineu][:,itest].T
        varexp_neurons[ineu, 0] = compute_varexp(spks_test, spks_pred_test)
        spks_test_bin = bin1d(spks_test, tbin)
        spks_pred_test_bin = bin1d(spks_pred_test, tbin)
        varexp_neurons[ineu, 1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
    # average variance explained for two halves
    varexp /= 2
    return varexp, varexp_neurons