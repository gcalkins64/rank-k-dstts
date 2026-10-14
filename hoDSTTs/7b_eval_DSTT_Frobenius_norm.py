# Author: Grace Calkins
# Date: 08/15/2025
# Purpose: Evaluate the difference between different STTs and DSTTs using the Frobenius norm

import re
import numpy as np
from scipy.io import loadmat
from JP_stts_aerocapture.inputs.inputs import uranus_inputs_JP_aero_stts, uranus_inputs_JP_aero_stts_drag_mod, uranus_inputs_JP_aero_stts_larger_perts_fast
from JP_stts_aerocapture.plotters import *
from JP_stts_aerocapture.util.data_manager import createPaths, load_R_matrices
from JP_stts_aerocapture.util.hocgts import get_CGT_R_matrix, getCGTEigenpairsMulti
from JP_stts_aerocapture.util.stt import reshape_DSTT, reshape_STT

def get_frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor.flatten()**2))

def main():
    plotDSTTsFlag = False
    plotNormalizedFroNormFlag = True

    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'
    ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts()
    runTag = 'JP_dstts_nominal'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_drag_mod()
    # runTag = 'JP_dstts_nominal_drag_mod'
    # runTag = 'JP_dstts_nominal_fast'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_larger_perts_fast()
    figPath, dataPath = createPaths(basePath, runTag)
    stateSize = len(x0)

    # Load in all STTs
    STTs_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)['STTs']
    STTs_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)['STTs']
    STTs_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)['STTs']
    STTs_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)['STTs']
    all_STTs = {}
    for regimeTag in ["t0tk", "tktkp1", "tktf", "regime"]:
        if regimeTag == "t0tk":
            all_STTs[regimeTag] = STTs_t0tk[1:]  # Chop off first STT since it's identity
        elif regimeTag == "tktkp1":
            all_STTs[regimeTag] = STTs_tktkp1[1:]  # Chop off first STT since it's identity
        elif regimeTag == "tktf":
            all_STTs[regimeTag] = STTs_tktf[1:]  # Chop off first STT since it's identity
        elif regimeTag == "regime":
            all_STTs[regimeTag] = STTs_regime[1:]  # Chop off first STT since it's identity


    # Load in Rs_data from  f"{dataPath}/Rs_data.npz", **Rs_data
    Rs_data = np.load(f"{dataPath}/Rs_data.npz", allow_pickle=True)

    # Load in DSTTs
    all_DSTTs = {}
    prefixes = ["CGT", "CGT_2", "CGT_3", "CGT_4", "CGT_5", "CGT_6", "CGT_7", "HOCGT", "sCGT", "energy_qCGT", "apoRad_qCGT"] #, 'rOnly_sCGT', 'pos_qCGT']
    for prefix in prefixes:
        match = re.fullmatch(r"CGT_(\d+)", prefix)
        regimeTags = ["t0tk", "tktkp1", "tktf", "regime", 't0tf']
        if prefix == "apoRad_qCGT":
            regimeTags = ["tktf", "t0tf"]
        if match:
            regimeTags = ["t0tk", "tktkp1", "tktf", "t0tf"]
        for regimeTag in regimeTags:
            DSTTs = np.load(f"{dataPath}/DSTTs_{prefix}_{regimeTag}.npz", allow_pickle=True)['DSTTs']
            all_DSTTs[f"{prefix}_{regimeTag}"] = DSTTs

    # Compute normalized Frobenius norm error between STTs and DSTTs
    frobenius_norms = {}
    for prefix in prefixes:
        match = re.fullmatch(r"CGT_(\d+)", prefix)
        regimeTags = ["t0tk", "tktkp1", "tktf", "regime"]
        if prefix == "apoRad_qCGT":
            regimeTags = ["tktf"]
        if match:
            regimeTags = ["t0tk", "tktkp1", "tktf"]
        for regimeTag in regimeTags:
            Rs = Rs_data[f"{prefix}_{regimeTag}"]
            t_len = len(ts)-1
            norms = np.zeros((t_len, 2))
            match = re.fullmatch(r"CGT_(\d+)", prefix)
            latent_dim = int(match.group(1)) if match else 1
            for tt in range(t_len):
                this_STT = all_STTs[regimeTag][tt,:] if regimeTag != 'regime' else all_STTs['t0tk'][tt,:]
                STTs = reshape_STT(this_STT, 3, len(x0))
                DSTTs = reshape_DSTT(all_DSTTs[f"{prefix}_{regimeTag}"][tt,:], 3, len(x0), latentDim=latent_dim)

                # Project DSTTs back into original basis
                if not match:
                    DSTT2_proj = np.einsum('i,j,k->ijk', np.squeeze(DSTTs[1]), np.squeeze(Rs[tt, 0, :]), np.squeeze(Rs[tt, 0, :]))  # Rotate back into original basis
                    DSTT3_proj = np.einsum('i,j,k,l->ijkl', np.squeeze(DSTTs[2]), np.squeeze(Rs[tt, 1, :]), np.squeeze(Rs[tt, 1, :]), np.squeeze(Rs[tt, 1, :]))
                else:
                    DSTT2_proj = np.einsum('ijk,jl,km->ilm', DSTTs[1], Rs[tt], Rs[tt])
                    DSTT3_proj = np.einsum('ijkl,jm,kn,lo->imno', DSTTs[2], Rs[tt], Rs[tt], Rs[tt])

                norms[tt, 0] = get_frobenius_norm(STTs[1] - DSTT2_proj) / get_frobenius_norm(STTs[1])
                norms[tt, 1] = get_frobenius_norm(STTs[2]- DSTT3_proj) / get_frobenius_norm(STTs[2])

            frobenius_norms[f"{prefix}_{regimeTag}"] = norms

    # Save off the Frobenius norms
    np.savez(f"{dataPath}/frobenius_norms.npz", **frobenius_norms)

    # Plot the normalized Frobenius norm for each time regime
    regimeTags = ["t0tk", "tktkp1", "tktf", "regime"]
    plot_prefixes = ["CGT", "CGT_2", "CGT_3", "CGT_4", "CGT_5", "CGT_6", "HOCGT", "sCGT", "energy_qCGT",
                "apoRad_qCGT"]  # Don't want to plot the CGT_7 because it is the same as the STT
    for regimeTag in regimeTags:
        norms_plot = [[],[]]  # length two list where each sublist is an order and each element of the sublist corresponds to a different prefix
        for prefix in plot_prefixes:
            match = re.fullmatch(r"CGT_(\d+)", prefix)

            # Skip CGT_x for 'regime'
            if match and regimeTag == "regime":
                continue

            # Skip apoRad_qCGT except for 'tktf'
            if prefix == "apoRad_qCGT" and regimeTag != "tktf":
                continue

            # Otherwise, process normally
            norms = frobenius_norms[f"{prefix}_{regimeTag}"]
            norms_plot[0].append(norms[:, 0])
            norms_plot[1].append(norms[:, 1])
        labels = plot_prefixes[:-1] if regimeTag != 'tktf' else plot_prefixes
        labels = [label for label in labels if not (re.fullmatch(r"CGT_(\d+)", label) and regimeTag == "regime")]

        ts_plot = ts[:-1]
        print(f"Plotting normalized Frobenius norm for regimeTag: {regimeTag}")
        if plotNormalizedFroNormFlag:
            plotNormalizedFroNorm(norms_plot, ts_plot, labels, f"{regimeTag}", f"{runTag}_{regimeTag}", figPath)
            # plt.show()

if __name__ == "__main__":
    main()
    plt.show()