import numpy as np
from scipy.io import loadmat, savemat
from scipy.integrate import solve_ivp
from dev_dstt_gmm.plotters import *
from dev_dstt_gmm.util.data_manager import createPaths
from dev_dstt_gmm.inputs.inputs import uranus_inputs_IPPW
from dev_dstt_gmm.util.stt import getSTTs_general, getNominalTrajLnDens, reshape_STT, STTstoDSTTs, getPerturbedTrajSTTsNonDimTkTkp1, getPerturbedTrajXDSTTsNonDimTkTkp1
from dev_dstt_gmm.util.nonDim import *
from dev_dstt_gmm.util.hocgts import *
from dev_dstt_gmm.util.tensor_toolbox import compute_eigenpairs_multiguess
from dev_dstt_gmm.util.hocgts import getHOCGTs_nonSym, getSCGTs_nonSym, getQCGTs_nonSym, get_Eta_Tensors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from tqdm import trange
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import pickle


def main():
    # We want to compute STTs up to third order, and then compute DSTTs, hoDSTTs, and sDSTTs selecting position / velocity / fpa
    computeSTTsFlag = False
    computeTEsFlag = True
    loadRsFlag = False
    computeDSTTsFlag = False

    order = 3  # STT order to use for propagation (2nd order DSTTs)

    # ------------------------------------------------------------------
    # Set up paths and inputs
    # ------------------------------------------------------------------
    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'
    runTag = 'test_sshopm_python'
    figPath, dataPath = createPaths(basePath, runTag)

    ts, x0, params, init_cov, proc_cov = uranus_inputs_IPPW()

    stateSize = len(x0)
    dT = ts[1] - ts[0]
    Tf = ts[-1]

    tk_tkp1_list = [[t, t + dT] for t in np.arange(0, Tf, dT)]


    # ------------------------------------------------------------------
    # Compute reference nominal trajectory and STTs
    # ------------------------------------------------------------------
    if computeSTTsFlag:
        print("Computing nominal trajectory and reference STTs...")
        Xs_d = getNominalTrajLnDens(ts, x0, params).T  # (T, n) dimensional
        Xs_nd = np.array([convertToNonDimDens(Xs_d[:,i], params)
                          for i in range(Xs_d.shape[1])])  # (T, n) non-dim

        # Reference STTs (tk -> tkp1, full trajectory)
        Xs_nd_ref, Xs_ref, STTs_ref = getSTTs_general(
            tk_tkp1_list, x0, order, params, params['eomTag'], mode='tktkp1'
        )
        np.savez(f"{dataPath}/STTs_tktkp1_reference.npz",
                 STTs=STTs_ref, Xs_nd=Xs_nd_ref, Xs_dim=Xs_ref, allow_pickle=True)

        # Compute reference nonsym HOCGTs
        get_and_save_HOCGT(STTs_ref, 'HOCGT', 'tktkp1', runTag, dataPath, params, order,
                           savePrefix=f'nominal')

    else:
        STTs_ref = np.load(f"{dataPath}/STTs_tktkp1_reference.npz", allow_pickle=True)['STTs']
        Xs_ref = np.load(f"{dataPath}/STTs_tktkp1_reference.npz", allow_pickle=True)['Xs_dim']
        Xs_nd_ref = np.load(f"{dataPath}/STTs_tktkp1_reference.npz", allow_pickle=True)['Xs_nd']


    # ------------------------------------------------------------------
    # Compute eigenpairs
    # ------------------------------------------------------------------
    # DO THIS IN MATLAB
    if computeTEsFlag:
        # OVERALL
        all_regimes = []  # rows = HOCGT type, cols = regimes

        # Initialize structure (like MATLAB cell array)
        # Ensure enough rows exist
        for _ in range(2):
            all_regimes.append([])

        # Python (0-based indexing)
        all_regimes[0].append(np.arange(0, len(ts)-1, 1))  # HOCGTs from tk, tk+1
        # all_regimes[1].append([0])  # SCGTs from tk, tk+1 - not included for now

        prefixes = ["nominal_HOCGT"] #, "nominal_sCGT"]
        regimeTags = ["tktkp1"]  # t0tk

        for ii in range(len(prefixes)):  # loop over type of HOCGTs
            prefix = prefixes[ii]

            for jj in range(len(regimeTags)): # loop over regime types
                if all_regimes[ii][jj].size:
                    regimes = all_regimes[ii][jj]
                    regimeTag = regimeTags[jj]

                    print(f"{prefix}, {regimeTag}")

                    HOCGTS_nonsym_all = load_nonsymmetric_tensors(dataPath, regimes, f"{prefix}_{regimeTag}", 'HOCGT', runTag, order)
                    HOCGTs_sym_all = symmetrize_tensors(HOCGTS_nonsym_all, order)
                    save_sym_tensors(HOCGTS_nonsym_all, dataPath, regimes, f"SYM_{prefix}_{regimeTag}", 'HOCGT', order)
                    [ho_lambdas, ho_evecs, ho_flags, ho_all_lambdas, ho_all_evecs] = compute_eigenpairs_multiguess(HOCGTs_sym_all, stateSize, order, verbFlag=True)
                    plot_eigenvalues(ho_lambdas, regimes, figPath, f"hoevals_{prefix}_{regimeTag}", order, ho_flags)
                    plot_eigenvectors(ho_evecs, regimes, stateSize, figPath, f"hoevecs_{prefix}_{regimeTag}", order, ho_flags)
                    plot_unique_eigenvalues(order, regimes, ho_all_lambdas, figPath, runTag, regimeTag)
                    plot_n_ho_evals(2, order, regimes, ho_all_lambdas, figPath, runTag, regimeTag)
                    plot_n_ho_evecs(2, order, regimes, ho_all_evecs, figPath, runTag, regimeTag, stateSize)
                    save_eigenvectors(ho_evecs, dataPath, 'ho', f"{prefix}_{regimeTag}")
                    save_eigenvalues(ho_lambdas, dataPath, 'ho', f"{prefix}_{regimeTag}")
                    save_HOeigenvectors(ho_all_evecs, dataPath, 'ho', f"{prefix}_{regimeTag}")
                    save_HOeigenvalues(ho_all_lambdas, dataPath, 'ho', f"{prefix}_{regimeTag}")


    # ------------------------------------------------------------------
    # Load in eigenpairs and construct R matrices
    # ------------------------------------------------------------------
    if loadRsFlag:
        # Print out the top eigenvector from ts to tf
        evals, evecs, HOCGTs = load_eigenpair_data(f'HOCGT_t0tk', dataPath, [0], order)
        Sevals, Sevecs, sCGTs = load_eigenpair_data(f'sCGT_t0tk', dataPath, [0], order)
        print("HOCGT max eigenvector")
        for oo in range(order):
            max_idx = np.argmax(evals[oo][0])
            print(f"O{oo+2} max evec: {evecs[oo][0][:, max_idx]}")

        print("sCGT max eigenvector")
        for oo in range(order):
            max_idx = np.argmax(Sevals[oo][0])
            print(f"O{oo + 2} max evec: {Sevecs[oo][0][:, max_idx]}")

        evals, evecs, HOCGTs = load_eigenpair_data(f'HOCGT_tktkp1', dataPath, ts, order)
        Sevals, Sevecs, sCGTs = load_eigenpair_data(f'sCGT_tktkp1', dataPath, ts, order)

        Rs, hoRs, sRs = getAllRs(order, stateSize, evals, evecs, Sevals, Sevecs)
        np.savez(f"{dataPath}/R_matrices_tktkp1_reference.npz", Rs=Rs, hoRs=hoRs, sRs=sRs, allow_pickle=True)

    DSTTs_ref, hoDSTTs_ref, sDSTTs_ref = [], [], []
    if computeDSTTsFlag:
        DSTTs_ref = STTstoDSTTs(ts[1:], STTs_ref, Rs, order, stateSize)
        hoDSTTs_ref = STTstoDSTTs(ts[1:], STTs_ref, hoRs, order, stateSize)
        sDSTTs_ref = STTstoDSTTs(ts[1:], STTs_ref, sRs, order, stateSize)

        np.savez(f"{dataPath}/DSTTs_tktkp1_reference.npz",
                 DSTTs=DSTTs_ref, Xs_nd=Xs_nd_ref, Xs_dim=Xs_ref, allow_pickle=True)
        np.savez(f"{dataPath}/hoDSTTs_tktkp1_reference.npz",
                 DSTTs=hoDSTTs_ref, Xs_nd=Xs_nd_ref, Xs_dim=Xs_ref, allow_pickle=True)
        np.savez(f"{dataPath}/sDSTTs_tktkp1_reference.npz",
                 DSTTs=sDSTTs_ref, Xs_nd=Xs_nd_ref, Xs_dim=Xs_ref, allow_pickle=True)

    else:
        DSTTs_ref = np.load(f"{dataPath}/DSTTs_tktkp1_reference.npz", allow_pickle=True)['DSTTs']
        hoDSTTs_ref = np.load(f"{dataPath}/hoDSTTs_tktkp1_reference.npz", allow_pickle=True)['DSTTs']
        sDSTTs_ref = np.load(f"{dataPath}/sDSTTs_tktkp1_reference.npz", allow_pickle=True)['DSTTs']


if __name__ == "__main__":
    main()