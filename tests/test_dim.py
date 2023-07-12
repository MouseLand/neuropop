import numpy as np
import torch
from neuropop import dimensionality

def test_nnpred(test_file):
    dat = np.load(test_file)
    spks = dat["spks"]
    scov, varcov = dimensionality.SVCA(spks)
    alpha, py = dimensionality.get_powerlaw(scov, trange=np.arange(10,len(scov)))
    assert alpha > 1. and alpha < 1.25
