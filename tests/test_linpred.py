import numpy as np
import torch
from neuropop import linear_prediction

device = torch.device("cpu")

def test_linpred(test_file):
    dat = np.load(test_file)
    spks = dat["spks"]
    tcam, tneural = dat["tcam"], dat["tneural"]
    beh = dat["beh"]

    ### fit linear model from behavior to neural activity
    ### predict activity from behavior
    ve, _, spks_pred, itest = linear_prediction.prediction_wrapper(beh, spks.T, rank=32,
                                                                    tcam=tcam, tneural=tneural,
                                                                    lam=1e-2, delay=-1, device=device)
    assert ve[-1] > 0.025

    ve, _, spks_pred, itest = linear_prediction.prediction_wrapper(beh, spks.T, rank=None,
                                                                    tcam=tcam, tneural=tneural,
                                                                    lam=1e-2, delay=-1, device=device)
    assert ve > 0.025