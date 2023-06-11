# neuropop
analysis tools for large-scale neural recordings

Copyright (C) 2023 Howard Hughes Medical Institute Janelia Research Campus, Carsen Stringer and Marius Pachitariu

**This code is licensed under GPL v3 (no redistribution without credit, and no redistribution in private repos, see [license](LICENSE) for more details).**

## references

[[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6642054/) Stringer, C.\*, Pachitariu, M.\*, Steinmetz, N., Carandini, M., & Harris, K. D. (2019). High-dimensional geometry of population responses in visual cortex. *Nature, 571*(7765), 361-365.

[[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/) Stringer, C.\*, Pachitariu, M.\*, Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science, 364*(6437), eaav7893.

[[3]](https://www.biorxiv.org/content/10.1101/2022.11.03.515121v1) Syeda, A., Zhong, L., Tung, R., Long, W., Pachitariu, M.\*, & Stringer, C.\* (2022). Facemap: a framework for modeling neural activity based on orofacial tracking. *bioRxiv*.


## [dimensionality.py](neuropop/dimensionality.py)

This module contains code for dimensionality estimation methods for neural data:

* Cross-validated PCA (described in [[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6642054/)) is for estimating the dimensionality of neural stimulus responses where each stimulus is shown at least twice. Divide your data into two repeats -- a matrix of 2 x stimuli x neurons, and input it into the function `cvPCA` to obtain the cross-validated eigenspectrum. Note that each repeat can be the average of several stimulus responses (e.g. 5-10 each). You can then use the `get_powerlaw` function to estimate the exponent of the decay of the eigenspectrum. If you use these functions please cite [[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6642054/).

* Shared variance components analysis (described in [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/)) is for estimating the dimensionality of neural activity that is shared across neurons (excluding single neuron variability). This method divides the neurons in half and the timepoints in half into training and testing. Then it computes the principal components of the covariance between the two neural halves on the training timepoints, and then computing the variance of those components on the testing timepoints. Take your neural data as a matrix of neurons x time and input it into the function `SVCA` to obtain the variance of each component of the covariance matrix on the test set `scov` (this had a powerlaw decay of ~1.1 in [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/)). The function also returns the average variance of each component in each neural half `varcov`. If you use this function please cite [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/).

## [peer_prediction.py](neuropop/peer_prediction.py)

Prediction of one half of neurons from the other half of neurons, in order to estimate an upper bound for the amount of predictable variance in the neural population. If you use this function, please cite [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/).

## [linear_prediction.py](neuropop/linear_prediction.py)

This module contains code for ridge regression and regularized reduced rank regression, particularly for predicting from behavior to neural activity. The main function is `prediction_wrapper`, if `rank` is None then ridge regression is performed, otherwise reduced rank regression is performed. This function assumes you have pytorch with GPU support, otherwise set `device=torch.device('cpu')`. CCA is also implemented without GPU support. If you use these functions please cite [[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6525101/).

## [nn_prediction.py](neuropop/nn_prediction.py)

This module contains code for non-linear prediction of neural activity from behavior, as described in [[3]](https://www.biorxiv.org/content/10.1101/2022.11.03.515121v1). The main function is `network_wrapper`, this function assumes you have pytorch with GPU support, otherwise set `device=torch.device('cpu')`, and it also assumes you have taken the principal components of the data `U`, e.g.:

```
# z-score neural activity
spks -= spks.mean(axis=1)[:, np.newaxis]
std = ((spks**2).mean(axis=1) ** 0.5)[:, np.newaxis]
std[std == 0] = 1
spks /= std

# compute principal components
from sklearn.decomposition import PCA
Y = PCA(n_components=128).fit_transform(spks.T)
U = spks @ Y
U /= (U**2).sum(axis=0) ** 0.5

# predict Y from behavior variables x (z-score x if using keypoints)
# tcam are camera/behavior timestamps, tneural are neural timestamps
varexp, varexp_neurons, spks_pred_test0, itest, model = network_wrapper(x, Y, tcam, tneural, U, spks, delay=-1, verbose=True)
```

If you use these functions please cite [[3]](https://www.biorxiv.org/content/10.1101/2022.11.03.515121v1).

## [future_prediction.py](neuropop/future_prediction.py)

This contains functions for predicting behavioral or neural variables into the future using ridge regression with exponential basis functions (see Figure 1 in [[3]](https://www.biorxiv.org/content/10.1101/2022.11.03.515121v1)). If you use these functions please cite [[3]](https://www.biorxiv.org/content/10.1101/2022.11.03.515121v1).

## requirements

This package relies on the following excellent packages (recommended to cite these in your work as well if you use them):
* numpy
* scipy
* scikit-learn
* pytorch
