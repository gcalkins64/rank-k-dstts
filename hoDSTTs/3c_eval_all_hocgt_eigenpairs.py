# Author: Grace Calkins
# Date: 08/15/2025
# Purpose: Load in eigenpairs from MATLAB and plot and compare them
from JP_stts_aerocapture.inputs.inputs import uranus_inputs_JP_aero_stts, uranus_inputs_JP_aero_stts_drag_mod, uranus_inputs_JP_aero_stts_larger_perts_fast
from JP_stts_aerocapture.plotters import *
from JP_stts_aerocapture.util.data_manager import createPaths
from scipy.io import loadmat
import itertools
from tqdm import tqdm

def load_eigenpair_data(prefix, regimeTag, dataPath):
    # Load in hoeigenpair datat from MATLAB
    eval_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_evals.mat")
    evec_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_Rs.mat")

    all_evals = eval_data['ho_evals']
    all_evecs = evec_data['ho_Rs']

    # Want to make evals and evecs lists of arrays
    evals, evecs = [], []
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

        evals.append(this_evals)
        evecs.append(this_evecs)

    # Evals is a [order] size list of lists with [number of time domains] size where each element is a [arbitrary size] np.array of evals
    # Evecs is a [order] size list of lists with [number of time domains] size where each element is a [stateSize, arbitrary size] np.array of evecs where each column is an evec

    return evals, evecs


def main():
    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'

    # runTag = 'JP_dstts_nominal'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts()
    # runTag = 'JP_dstts_nominal_drag_mod'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_drag_mod()
    runTag = 'JP_dstts_nominal_fast'
    ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_larger_perts_fast()
    figPath, dataPath = createPaths(basePath, runTag)

    # Plot max eval, number of evals found, and eigenvector for each time domain and CGT type
    # HOCGTs - t0tk, tktkp1, tktf, Regime
    # sCGTs - t0tk, tktkp1, tktf, Regime
    # S for density, position, and velocity
    # qCGTs
    ## Energy - t0tk, tktkp1, tktf, Regime
    ## Apoapsis - tktf

    # prefixes = ["HOCGT", "sCGT", "energy_qCGT", "apoRad_qCGT"] #, 'rOnly_sCGT', 'pos_qCGT']
    prefixes = ["HOCGT"] #, 'rOnly_sCGT', 'pos_qCGT']

    for prefix in prefixes:
        if prefix == "apoRad_qCGT":
            regimeTags = ["tktf", 't0tf']
            # regimeTags = ['t0tf']
        else:
            # regimeTags = ["tktkp1", "t0tk", "tktf", "regime", 't0tf']
            regimeTags = ["t0tk"]
            # regimeTags = ["tktkp1", "t0tk", 't0tf']
        for regimeTag in tqdm(regimeTags):
            evals, evecs = load_eigenpair_data(prefix, regimeTag, dataPath)
            # Plot max eval
            plotMaxEval(evals, prefix, regimeTag, figPath)
            # Plot top n eval
            plotNEval(evals, prefix, regimeTag, figPath, 2)
            # Plot the difference between the top two eigenvalues
            plotEvalDiff(evals, prefix, regimeTag, figPath)
            # Plot number of evals
            plotNumEvals(evals, prefix, regimeTag, figPath)
            # Plot max eigenvector
            plotMaxEigenvector(evecs, prefix, regimeTag, figPath)
            # Plot top eigenvectors
            plotTopEvecs(evecs, prefix, regimeTag, figPath)
            # plt.close('all')
            plt.show()

            # If we are t0/tf for the regime tag, print the max eigenvalue and eigenvector
            if regimeTag in ['t0tf']:
                print(f"Prefix: {prefix}, Regime: {regimeTag}")
                for order, (this_evals, this_evecs) in enumerate(zip(evals, evecs)):
                    print(f"  Order: {order+1}")
                    for time_idx, (eval_array, evec_array) in enumerate(zip(this_evals, this_evecs)):
                        max_eval_idx = np.argmax(np.abs(eval_array))
                        max_eval = eval_array[max_eval_idx]
                        max_evec = this_evecs[time_idx][:, max_eval_idx]
                        print(f"    Time idx: {time_idx}, Max Eval: {max_eval}")
                        print(f"    Max Evec: {max_evec}")

if __name__ == "__main__":
    main()