import numpy as np
import os
from scipy.io import loadmat

def createPaths(basePath, runTag, dataPathOnly=False):
    # Create paths for figures and data
    figPath = os.path.join(basePath, 'figures', runTag)
    dataPath = os.path.join(basePath, 'data', runTag)

    if not dataPathOnly:
        if not os.path.exists(figPath):
            os.makedirs(figPath)
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    return figPath, dataPath

def load_R_matrices(prefix, regimeTag, dataPath):
    # Load in hoeigenpair datat from MATLAB
    eval_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_evals.mat")
    evec_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_Rs.mat")

    all_evals = eval_data['ho_evals']
    all_evecs = evec_data['ho_Rs']

    # Want to make evals and evecs lists of arrays
    evecs = []
    for j in range(all_evals.shape[0]):
        this_evals = []
        this_evecs = []
        for i in range(all_evals.shape[1]):
            if j == 0:
                this_evals.append(np.diag(all_evals[j, i]).flatten())
                this_evecs.append(all_evecs[j, i])
            else:
                this_evals.append(all_evals[j, i].flatten())
                that_evec = []
                for k in range(all_evecs[j, i].shape[1]):
                    that_evec.append(np.squeeze(all_evecs[j, i][0, k]))
                this_evecs.append(np.array(that_evec).T)

        evecs.append(this_evecs)

    # Evecs is a [order] size list of lists with [number of time domains] size where each element is a [stateSize, arbitrary size] np.array of evecs where each column is an evec

    # Need to convert to a times x order-1 x state size matrix
    # But we want the first order to be the third order CGT eigenpairs
    Rs = np.zeros((len(evecs[0]), len(evecs)-1, evecs[0][0].shape[0]))  # times x order x state size
    for order in range(len(evecs)-1):
        for time in range(len(evecs[order])):
            Rs[time, order, :] = evecs[order+1][time][:,0] # Take the first eigenvector (for maximum eigenvalue)

    return Rs


def load_case(i, dataPath):
    """Helper function to load a single .mat file."""
    data = loadmat(os.path.join(dataPath, f'case{i}.mat'))
    return {'X': data['X'], 'ts': data['ts'], 'ra': data['ra']}

def load_case_tag(i, dataPath, tag):
    """Helper function to load a single .mat file."""
    data = loadmat(os.path.join(dataPath, f'case{i}_{tag}.mat'))
    return {'X': data['X'], 'ts': data['ts'], 'ra': data['ra']}
