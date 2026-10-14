# Author: Grace Calkins
# Date: 08/15/2025
# Purpose: Get DSTTs from STTs

import re
import numpy as np
from scipy.io import loadmat
from JP_stts_aerocapture.inputs.inputs import uranus_inputs_JP_aero_stts, uranus_inputs_JP_aero_stts_drag_mod, uranus_inputs_JP_aero_stts_larger_perts_fast
from JP_stts_aerocapture.plotters import *
from JP_stts_aerocapture.util.data_manager import createPaths, load_R_matrices
from JP_stts_aerocapture.util.stt import STTstoDSTTs, STTstoDSTTsLargerLatent
from JP_stts_aerocapture.util.hocgts import get_CGT_R_matrix, getCGTEigenpairsMulti
from tqdm import tqdm

def main():
    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'

    runTag = 'JP_dstts_nominal'
    ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts()
    # runTag = 'JP_dstts_nominal_drag_mod'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_drag_mod()
    # runTag = 'JP_dstts_nominal_fast'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_larger_perts_fast()
    figPath, dataPath = createPaths(basePath, runTag)

    order = 3
    stateSize = len(x0)

    # Load in all STTs
    STTs_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)['STTs']
    Xs_nd_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)['Xs_nd']
    STTs_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)['STTs']
    Xs_nd_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)['Xs_nd']
    STTs_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)['STTs']
    Xs_nd_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)['Xs_nd']
    STTs_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)['STTs']
    Xs_nd_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)['Xs_nd']
    STTs_t0tF = np.load(f"{dataPath}/STTs_t0tf.npz", allow_pickle=True)['STTs']
    Xs_nd_t0tF = np.load(f"{dataPath}/STTs_t0tf.npz", allow_pickle=True)['Xs_nd']

    regime_data = np.load(f"{dataPath}/regime_times.npz", allow_pickle=True)
    phase2StartTime = regime_data['phase2StartTime']
    phase3StartTime = regime_data['phase3StartTime']
    # Get inidices of phase start times from ts
    phase2StartInd = np.where(ts == phase2StartTime)[0][0]
    phase3StartInd = np.where(ts == phase3StartTime)[0][0]

    # Load in all directions
    Rs_data = {}
    # prefixes = ["HOCGT", "sCGT", "energy_qCGT", "apoRad_qCGT", 'rOnly_sCGT', 'pos_qCGT']
    prefixes = ["HOCGT", "sCGT", "energy_qCGT", "apoRad_qCGT"] #, 'rOnly_sCGT', 'pos_qCGT']

    for prefix in prefixes:
        if prefix == "apoRad_qCGT":
            regimeTags = ["tktf", "t0tf"]
            # regimeTags = ["t0tf"]
        else:
            regimeTags = ["t0tk", "tktkp1", "tktf", "regime", 't0tf']
            # regimeTags = ["t0tk", "tktkp1", 't0tf']
        for regimeTag in regimeTags:
            Rs = load_R_matrices(prefix, regimeTag, dataPath)
            # Build Rs matrix for regime STTs
            if regimeTag == "regime":
                # Regime Rs are just the regime Rs replicated throughout the three phases
                Rs_out = np.zeros((len(ts)-1, 2, stateSize))
                Rs_out[:phase2StartInd-1,:,:] = np.tile( Rs[0,:,:], (phase2StartInd-1, 1, 1))  # Phase 1
                Rs_out[phase2StartInd-1:phase3StartInd-1,:,:] = np.tile( Rs[1,:,:], (phase3StartInd-1 - (phase2StartInd-1), 1, 1))  # Phase 2
                Rs_out[phase3StartInd-1:,:,:] = np.tile( Rs[2,:,:], (len(ts)-1 - (phase3StartInd-1), 1, 1))  # Phase 3
                Rs = Rs_out

            Rs_data[f"{prefix}_{regimeTag}"] = Rs

    # Make regular second order CGT R matrix where the
    regimeTags = ["t0tk", "tktkp1", "tktf", "regime", 't0tf']
    # regimeTags = ["t0tk", "tktkp1", 't0tf']
    for regimeTag in regimeTags:
        Rs = get_CGT_R_matrix("HOCGT", regimeTag, dataPath)

        if regimeTag == 'regime':
            Rs_out = np.zeros((len(ts) - 1, 2, stateSize))
            Rs_out[:phase2StartInd - 1, :, :] = np.tile(Rs[0, :, :], (phase2StartInd - 1, 1, 1))  # Phase 1
            Rs_out[phase2StartInd - 1:phase3StartInd - 1, :, :] = np.tile(Rs[1, :, :], (
            phase3StartInd - 1 - (phase2StartInd - 1), 1, 1))  # Phase 2
            Rs_out[phase3StartInd - 1:, :, :] = np.tile(Rs[2, :, :],
                                                        (len(ts) - 1 - (phase3StartInd - 1), 1, 1))  # Phase 3
            Rs = Rs_out

        Rs_data[f"CGT_{regimeTag}"] = Rs
    prefixes.append("CGT")

    # We also want a DSTT with a latent dimension of 2
    latent_dims = [2,3,4,5,6,7]
    for regime in regimeTags:
        if regime == "t0tk":
            STTs = STTs_t0tk[1:,:]
        elif regime == "tktkp1":
            STTs = STTs_tktkp1[1:,:]
        elif regime == "tktf":
            STTs = STTs_tktf[1:,:]
        elif regime == "regime":
            STTs = STTs_regime[1:,:]
        elif regime == 't0tf':
            STTs = STTs_t0tF[1:,:]
        for latent_dim in latent_dims:
            Rs_larger = getCGTEigenpairsMulti(STTs, latent_dim, stateSize)
            Rs_data[f"CGT_{latent_dim}_{regime}"] = Rs_larger

    for latent_dim in latent_dims:
        prefixes.append(f"CGT_{latent_dim}")

    # Save off this Rs_data
    print(f"Saving {dataPath}/Rs_data.npz")
    np.savez(f"{dataPath}/Rs_data.npz", **Rs_data)

    # Loop and get all DSTTs and save
    all_DSTTs = {}
    for prefix in tqdm(prefixes):
        match = re.fullmatch(r"CGT_(\d+)", prefix)
        regimeTags = ["t0tk", "tktkp1", "tktf", "regime", 't0tf']
        # regimeTags = ["t0tk", "tktkp1", 't0tf']
        if prefix == "apoRad_qCGT":
            regimeTags = ["tktf", "t0tf"]
            # regimeTags = ["t0tf"]
        if match:
            regimeTags = ["t0tk", "tktkp1", "tktf", 't0tf']
        for regimeTag in regimeTags:
            if regimeTag == "t0tk":
                thisTs = ts[1:]
                STTs = STTs_t0tk[1:,:]  # Skip the first identity STT
            elif regimeTag == "tktkp1":
                thisTs = ts[1:]
                STTs = STTs_tktkp1[1:,:]  # Skip the first identity STT
            elif regimeTag == "tktf":
                thisTs = ts[1:]
                STTs = STTs_tktf[1:,:]  # Skip the first identity STT
            elif regimeTag == "regime":
                # Regime DSTTs are t0/tk STTs with the Rs replicated throughout the three phases
                thisTs = ts[1:]
                STTs = STTs_t0tk[1:,:]
            elif regimeTag == 't0tf':
                thisTs = [ts[0]]
                STTs = STTs_t0tF[1:,:]

            Rs = Rs_data[f"{prefix}_{regimeTag}"]
            if not match:
                DSTTs = STTstoDSTTs(thisTs, STTs, Rs, order, stateSize)
            else:
                DSTTs = STTstoDSTTsLargerLatent(thisTs, STTs, Rs, order, stateSize)
            all_DSTTs[f"{prefix}_{regimeTag}"] = DSTTs

            # Plot DSTTs
            this_latent_dim = int(match.group(1)) if match else 1
            plotDSTTs(x0, thisTs, DSTTs, f'{prefix}: {regimeTag}', latent_dim=this_latent_dim, figPath=figPath, runTag=f"{prefix}_{regimeTag}")
            if len(thisTs) < len(Rs):
                thisTs = ts
            plot_Rs(Rs, thisTs, f'{prefix}: {regimeTag}', figPath, f"{prefix}_{regimeTag}")
            plt.close('all')
            # plt.show()

            print(f"Saving {dataPath}/DSTTs_{prefix}_{regimeTag}.npz")
            np.savez(f"{dataPath}/DSTTs_{prefix}_{regimeTag}.npz", DSTTs=DSTTs)

    # # Get the difference between the rOnly_sCGTs and pos_qCGTs DSTTs and plot for each time regime
    # # regimeTags = ["t0tk", "tktkp1", "tktf", "regime", 't0tf']
    # regimeTags = ["t0tk", "tktkp1", 't0tf']
    # for regimeTag in regimeTags:
    #     DSTT_diff = all_DSTTs[f"rOnly_sCGT_{regimeTag}"] - all_DSTTs[f"pos_qCGT_{regimeTag}"]
    #     abs_DSTT_diff = np.abs(all_DSTTs[f"rOnly_sCGT_{regimeTag}"]) - np.abs(all_DSTTs[f"pos_qCGT_{regimeTag}"])
    #     if regimeTag == "t0tk":
    #         thisTs = ts[1:]
    #     elif regimeTag == "tktkp1":
    #         thisTs = ts[1:]
    #     elif regimeTag == "tktf":
    #         thisTs = ts[1:]
    #     elif regimeTag == "regime":
    #         thisTs = ts[1:]
    #     elif regimeTag == 't0tf':
    #         thisTs = [ts[0]]
    #     plotDSTTs(x0, thisTs, DSTT_diff, f'r-sDSTT - r-qDSTT: {regimeTag}', figPath=figPath, runTag=f"sCGT_qCGT_diff_{regimeTag}")
    #     plotDSTTs(x0, thisTs, abs_DSTT_diff, f'|r-sDSTT - r-qDSTT|: {regimeTag}', figPath=figPath, runTag=f"abs_sCGT_qCGT_diff_{regimeTag}")


if __name__ == "__main__":
    main()
    plt.show()