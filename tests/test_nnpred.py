import numpy as np
import torch
from neuropop import nn_prediction

device = torch.device("cpu")

def test_nnpred(test_file):
    dat = np.load(test_file)
    sv, V = dat["sv"], dat["V"]
    tcam, tneural = dat["tcam"], dat["tneural"]
    beh = dat["beh"]
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ### fit nonlinear model from behavior to neural activity
    Vfit = V.copy() * sv
    pred_model = nn_prediction.PredictionNetwork(n_in=beh.shape[-1], n_kp=22, n_out=Vfit.shape[-1], )
    y_pred_all, ve_all, itest = pred_model.train_model(beh, Vfit, tcam, tneural, delay=-1,
                                                            learning_rate=1e-3, n_iter=30,
                                                        device=device, verbose=True)
    assert ve_all > 0.4


