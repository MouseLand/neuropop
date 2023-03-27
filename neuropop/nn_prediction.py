import sys, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neuropop.filtering import gabor_wavelet
from neuropop.utils import bin1d, compute_varexp
from neuropop.split_data import split_data

def network_wrapper(x, Y, tcam, tneural, U, spks, delay=-1,
                        verbose=False, per_pt=False, device=torch.device('cuda')):
    x = (x - x.mean(axis=0)) / x[:,0].std(axis=0)
    
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed(0)
    model = PredictionNetwork(n_in=x.shape[-1], n_out=Y.shape[-1]).to(device)

    y_pred_test, ve_test, itest = model.train_model(x, Y, tcam, tneural, delay=delay, 
                                                verbose=verbose, device=device)
                                                
    y_pred_test = y_pred_test.reshape(-1, Y.shape[-1])
    varexp = np.zeros(2)
    varexp_neurons = np.zeros((len(spks), 2))
    varexp[0] = ve_test
    Y_test_bin = bin1d(Y[itest.flatten()], 4)
    Y_pred_test_bin = bin1d(y_pred_test, 4)
    varexp[1] = 1 - ((Y_test_bin - Y_pred_test_bin)**2).mean() / ((Y_test_bin)**2).mean()
    print(f'all kp, varexp {varexp[0]:.3f}; tbin=4: {varexp[1]:.3f}')
    spks_pred_test = y_pred_test @ U.T
    spks_test = spks[:,itest.flatten()].T
    varexp_neurons[:,0] = compute_varexp(spks_test, spks_pred_test)
    spks_test_bin = bin1d(spks_test, 4)
    spks_pred_test_bin = bin1d(spks_pred_test, 4)
    varexp_neurons[:,1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
    spks_pred_test0 = spks_pred_test.T.copy()

    # predict using each variable
    if per_pt:
        varexp_per_pt = np.nan * np.zeros((x.shape[1], 2))
        varexp_neurons_per_pt = np.nan * np.zeros((len(spks), x.shape[1], 2))
        for k in enumerate(x.shape[1]):
            np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed(0)
            model = PredictionNetwork(n_in=1, n_out=Y.shape[-1]).to(device)

            y_pred_test, ve_test, itest = model.train_model(x[:, k],
                                                        Y, tcam, tneural, 
                                                      delay=delay, device=device)
            y_pred_test = y_pred_test.reshape(-1, Y.shape[-1])
            varexp_per_pt[k,0] = ve_test      
            varexp_per_pt[k,1] = compute_varexp(bin1d(Y[itest.flatten()], 4).flatten(), 
                                                bin1d(y_pred_test, 4).flatten())
            spks_pred_test = y_pred_test @ U.T
            spks_test = spks[:,itest.flatten()].T
            varexp_neurons_per_pt[:,k,0] = compute_varexp(spks_test, spks_pred_test)
            spks_test_bin = bin1d(spks_test, 4)
            spks_pred_test_bin = bin1d(spks_pred_test, 4)
            varexp_neurons_per_pt[:,k,1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
            print(f'{k}, varexp {ve_test:.3f}, {varexp_neurons_per_pt[:,k,0].mean():.3f}')
            return varexp, varexp_neurons, spks_pred_test0, itest, varexp_per_pt, varexp_neurons_per_pt
    else:
        return varexp, varexp_neurons, spks_pred_test0, itest, model

class Core(nn.Module):
    """ linear -> conv1d -> relu -> linear -> relu = latents for KPN model"""
    def __init__(self, n_in=28, n_kp=None, n_filt=10, kernel_size=201, 
                 n_layers=1, n_med=50, n_latents=256, 
                 identity=False, same_conv=True, 
                 relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.n_in = n_in
        self.n_kp = n_in if n_kp is None or identity else n_kp
        self.n_filt = (n_filt//2) * 2 # must be even for initialization
        self.relu_latents = relu_latents
        self.relu_wavelets = relu_wavelets
        self.same_conv = same_conv
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.features = nn.Sequential()

        # combine keypoints into n_kp features
        if identity:
            self.features.add_module('linear0', nn.Identity(self.n_in))
        else:
            self.features.add_module('linear0', nn.Sequential(nn.Linear(self.n_in, self.n_kp),
                                                              ))
        # initialize filters with gabors
        f = np.geomspace(1, 10, self.n_filt//2).astype('float32')
        gw0 = gabor_wavelet(1, f[:,np.newaxis], 0, n_pts=kernel_size)
        gw1 = gabor_wavelet(1, f[:,np.newaxis], np.pi/2, n_pts=kernel_size)
        if self.same_conv:
            # compute n_filt wavelet features of each one => n_filt * n_kp features
            self.features.add_module('wavelet0', nn.Conv1d(1, self.n_filt, kernel_size=kernel_size,
                                                        padding=kernel_size//2, bias=False))
            self.features[-1].weight.data = torch.from_numpy(np.vstack((gw0, gw1))).unsqueeze(1)
        else:
            self.features.add_module('wavelet0', nn.Conv1d(self.n_kp, self.n_kp, kernel_size=kernel_size,
                                                        padding=kernel_size//2, bias=False, groups=self.n_kp))
            self.features[-1].weight.data = torch.tile(torch.from_numpy(gw0[[1]]).unsqueeze(1), 
                                                        (self.n_kp, 1, 1))
            self.n_filt = 1
        for n in range(1, n_layers):
            n_in = self.n_kp * self.n_filt if n==1 else n_med
            self.features.add_module(f'linear{n}', nn.Sequential(nn.Linear(n_in, 
                                                                            n_med),
                                                                 ))

        # latent linear layer
        n_med = n_med if n_layers > 1 else self.n_filt * self.n_kp
        self.features.add_module('latent', nn.Sequential(nn.Linear(n_med, n_latents),
                                                        ))
        
    def wavelets(self, x):
        """ compute wavelets of keypoints through linear + conv1d + relu layer """
        # x is (n_batches, time, features)
        out = self.features[0](x.reshape(-1, x.shape[-1]))
        out = out.reshape(x.shape[0], x.shape[1], -1).transpose(2,1)
        # out is now (n_batches, n_kp, time)
        if self.same_conv:
            out = out.reshape(-1, out.shape[-1]).unsqueeze(1)
            # out is now (n_batches * n_kp, 1, time)
            out = self.features[1](out)
            # out is now (n_batches * n_kp, n_filt, time)
            out = out.reshape(-1, self.n_kp * self.n_filt, out.shape[-1]).transpose(2,1)
            out = out.reshape(-1, self.n_kp * self.n_filt)
        else:
            out = self.features[1](out)
            out = out.transpose(-1,-2)
        if self.relu_wavelets:
            out = F.relu(out)
        
        # if n_layers > 1, go through more linear layers
        for n in range(1, self.n_layers):
            out = self.features[n+1](out)
            out = F.relu(out)
        return out
                                              
    def forward(self, x=None, wavelets=None):
        """ x is (n_batches, time, features)
            sample_inds is (sub_time) over batches
        """
        if wavelets is None:
            wavelets = self.wavelets(x)
        wavelets = wavelets.reshape(-1, wavelets.shape[-1])
        
        # latent layer
        latents = self.features[-1](wavelets)
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        if self.relu_latents:
            latents = F.relu(latents)
        latents = latents.reshape(-1, latents.shape[-1])
        return latents

class Readout(nn.Module):
    """ linear layer from latents to neural PCs or neurons """
    def __init__(self, n_animals=1, n_latents=256, n_layers=1, 
                n_med=128, n_out=128):
        super().__init__()
        self.n_animals = n_animals
        self.linear = nn.Sequential()
        self.bias = nn.Parameter(torch.zeros(n_out))
        if n_animals==1:
            for j in range(n_layers):
                n_in = n_latents if j==0 else n_med 
                n_outc = n_out if j==n_layers-1 else n_med 
                self.linear.append(nn.Linear(n_in, n_outc))
                if n_layers > 1 and j < n_layers-1:
                    self.linear.append(nn.ReLU())
        else:
            # no option for n_layers > 1
            for n in range(n_animals):
                self.linear.append(nn.Linear(n_latents, n_out))
        self.bias.requires_grad = False

    def forward(self, latents, animal_id=0):
        if self.n_animals==1:
            return self.linear(latents) + self.bias
        else:
            return self.linear[animal_id](latents) + self.bias

class PredictionNetwork(nn.Module):
    """ predict from behavior to neural PCs / neural activity model """
    def __init__(self, n_in=28, n_kp=None, n_filt=10, kernel_size=201, n_core_layers=2,
                 n_latents=256, n_out_layers=1, n_out=128, n_med=50, n_animals=1, same_conv=True,
                 identity=False, relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.core = Core(n_in=n_in, n_kp=n_kp, n_filt=n_filt, kernel_size=kernel_size, 
                         n_layers=n_core_layers, n_med=n_med, n_latents=n_latents, same_conv=same_conv,
                         identity=identity, relu_wavelets=relu_wavelets, relu_latents=relu_latents)
        self.readout = Readout(n_animals=n_animals, n_latents=n_latents, n_layers=n_out_layers, 
                                n_out=n_out)

    def forward(self, x, sample_inds=None, animal_id=0):
        latents = self.core(x)
        if sample_inds is not None:
            latents = latents[sample_inds]
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        y_pred = self.readout(latents, animal_id=animal_id)
        return y_pred, latents
    
    def train_model(self, X_dat, Y_dat, tcam_list, tneural_list, 
                        delay=-1, smoothing_penalty=0.5, 
                    n_iter=300, learning_rate=5e-4, annealing_steps=2,
                    weight_decay=1e-4, device=torch.device('cuda'), 
                    split_time=False, verbose=False):
        """ train behavior -> neural model using multiple animals """

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        ### make input data a list if it's not already
        not_list = False
        if not isinstance(X_dat, list):
            not_list = True
            X_dat, Y_dat, tcam_list, tneural_list = [X_dat], [Y_dat], [tcam_list], [tneural_list]
        
        ### split data into train / test and concatenate
        arrs = [[],[],[],[],[],[],[],[],[],[]]
        for i, (X, Y, tcam, tneural) in enumerate(zip(X_dat, Y_dat, tcam_list, tneural_list)):
            dsplits = split_data(X, Y, tcam, tneural, delay=delay, split_time=split_time, device=device)
            for d,a in zip(dsplits, arrs):
                a.append(d)
        X_train, X_test, Y_train, Y_test, itrain_sample_b, itest_sample_b, itrain_sample, itest_sample, itrain, itest = arrs
        n_animals = len(X_train)
        
        tic = time.time()
        ### determine total number of batches across all animals to sample from
        n_batches = [0]
        n_batches.extend([X_train[i].shape[0] for i in range(n_animals)]) 
        n_batches = np.array(n_batches)
        c_batches = np.cumsum(n_batches)
        n_batches = n_batches.sum()   

        anneal_epochs = n_iter - 50*np.arange(1, annealing_steps+1)

        ### optimize all parameters with SGD
        for epoch in range(n_iter):
            self.train()
            if epoch in anneal_epochs:
                if verbose:
                    print('annealing learning rate')
                optimizer.param_groups[0]['lr'] /= 10.
            np.random.seed(epoch)
            rperm = np.random.permutation(n_batches)
            train_loss = 0
            for nr in rperm:
                i = np.nonzero(nr >= c_batches)[0][-1]
                n = nr - c_batches[i]
                y_pred = self.forward(X_train[i][n].unsqueeze(0), 
                            itrain_sample_b[i][n], 
                            animal_id=i)[0]
                loss = ((y_pred - Y_train[i][n].unsqueeze(0))**2).mean()
                loss += smoothing_penalty * (torch.diff(self.core.features[1].weight)**2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= n_batches

            # compute test loss and test variance explained
            if epoch%20==0 or epoch==n_iter-1:
                ve_all, y_pred_all = [], []
                self.eval()
                with torch.no_grad():
                    pstr = f'epoch {epoch}, '
                    for i in range(n_animals):
                        y_pred = self.forward(X_test[i], itest_sample_b[i].flatten(), animal_id=i)[0]
                        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                        tl = ((y_pred - Y_test[i])**2).mean()
                        ve = 1 - tl / ((Y_test[i] - Y_test[i].mean(axis=0))**2).mean()
                        y_pred_all.append(y_pred.cpu().numpy())
                        ve_all.append(ve.item())
                        if n_animals==1:
                            pstr += f'animal {i}, train loss {train_loss:.4f}, test loss {tl.item():.4f}, varexp {ve.item():.4f}, '
                        else:
                            pstr += f'varexp{i} {ve.item():.4f}, '
                pstr += f'time {time.time()-tic:.1f}s'
                if verbose:
                    print(pstr)
        
        if not_list:
            return y_pred_all[0], ve_all[0], itest[0]
        else:
            return y_pred_all, ve_all, itest

def train_model_test(model, X, Y, tcam, tneural, sgd=False, lam=1e-3, 
                   n_iter=600, learning_rate=5e-4, fix_model=True,
                   smoothing_penalty=1.0,
                   weight_decay=1e-4, device=torch.device('cuda')):

    dsplits = split_data(X, Y, tcam, tneural, device=device)
    X_train, X_test, Y_train, Y_test, itrain_sample_b, itest_sample_b, itrain_sample, itest_sample, itrain, itest = dsplits
            
    tic = time.time()

    n_batches = X_train.shape[0]
    if sgd:
        model.train()
        if fix_model:
            for param in model.parameters():
                param.requires_grad = False 
            model.test_classifier.weight.requires_grad = True
            model.test_classifier.bias.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True 
            
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(n_iter):
            model.train()
            np.random.seed(epoch)
            rperm = np.random.permutation(n_batches)
            train_loss = 0
            for n in rperm:
                y_pred, latents = model(X_train[n].unsqueeze(0), 
                                    itrain_sample_b[n], 
                                    test=True)
                loss = ((y_pred - Y_train[n].unsqueeze(0))**2).mean()
                loss += smoothing_penalty * (torch.diff(model.features[1].weight)**2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= n_batches
            if epoch%20==0 or epoch==n_iter-1:
                ve_all = []
                y_pred_all = []
                with torch.no_grad():
                    model.eval()
                    pstr = f'epoch {epoch}, '
                    y_pred = model(X_test, itest_sample_b.flatten(), test=True)[0]
                    tl = ((y_pred - Y_test)**2).mean()
                    ve = 1 - tl / (Y_test**2).mean()
                    #ve = ve.item()
                    y_pred = y_pred.cpu().numpy()
                    #y_pred_all.append(y_pred.cpu().numpy())
                    #ve_all.append(ve.item())
                    pstr += f'train loss {train_loss:.4f}, test loss {tl.item():.4f}, varexp {ve.item():.4f}'
                    print(pstr)

    else:
        itrain = itrain.reshape(n_batches, -1)
        l_train = itrain.shape[-1]
        with torch.no_grad():
            model.eval()
            for n in range(n_batches):
                y_pred, latents = model(X_train[n].unsqueeze(0), 
                                    itrain_sample_b[n], 
                                    test=True)
                if n==0:
                    n_latents = latents.shape[-1]
                    latents_train = np.ones((itrain.size, n_latents+1), 'float32')
                latents_train[n*l_train : (n+1)*l_train, :n_latents] = latents.cpu().numpy()
            latents_test = np.ones((itest.size, n_latents+1), 'float32')
            latents_test[:,:n_latents] = model(X_test, 
                                            itest_sample_b.flatten(), 
                                            test=True)[1].cpu().numpy().reshape(-1, n_latents)

            Y_train = Y_train.cpu().numpy()
            Y_test = Y_test.cpu().numpy().reshape(-1, Y_test.shape[-1])
            Y_train = Y_train.reshape(-1, Y_train.shape[-1])          
            A = np.linalg.solve(latents_train.T @ latents_train + lam * np.eye(n_latents+1),
                                latents_train.T @ Y_train)
            y_pred = latents_test @ A 
            tl = ((y_pred - Y_test)**2).mean()
            ve = 1 - tl / (Y_test**2).mean()
            model.test_classifier.weight.data = torch.from_numpy(A[:n_latents].T).float().to(device)
            model.test_classifier.bias.data = torch.from_numpy(A[-1]).float().to(device)
            print(ve)

    return y_pred, ve, itest
