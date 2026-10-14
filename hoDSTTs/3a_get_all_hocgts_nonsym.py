# Author: Grace Calkins
# Date: 08/15/2025
# Purpose: Compute all HOCGTs, sCGTs, qCGTs for the aerocapture trajectory (nonsymmetrical) to pass into MATLAB to symmetrize and solve the tensor eigenpair problem

from JP_stts_aerocapture.inputs.inputs import uranus_inputs_JP_aero_stts, uranus_inputs_JP_aero_stts_drag_mod, uranus_inputs_JP_aero_stts_larger_perts_fast
from JP_stts_aerocapture.util.hocgts import getHOCGTs_nonSym, getSCGTs_nonSym, getQCGTs_nonSym, get_Eta_Tensors
from JP_stts_aerocapture.plotters import seabornSettings
from JP_stts_aerocapture.util.data_manager import createPaths
from JP_stts_aerocapture.util.aerocapture import apoRad_func, finalPos_func, energy_func, pos_func
import os
from scipy.io import savemat
import numpy as np

def get_and_save_HOCGT(STTs, cgtType, timeDomain, runTag, dataPath, params, Smat=None, Xs=None, q_func=None, savePrefix=None):
    HOCGTs_nonsym_list = []
    for ii in range(len(STTs) - 1):
        if cgtType == 'HOCGT':
            HOCGT_nonsym = getHOCGTs_nonSym(STTs[ii + 1], order=3, stateSize=7)
        elif cgtType == 'sCGT':
            HOCGT_nonsym = getSCGTs_nonSym(STTs[ii + 1], order=3, S=Smat)
        elif cgtType == 'qCGT':
            # Make sure the Xk we are evaluating is the correct one for the time domain
            if timeDomain == 't0tk': # Evaluate at Xk
                X_eval = Xs[ii+1]
            elif timeDomain == 'tktkp1':  # Evaluate at Xk+1
                X_eval = Xs[ii+1]
            elif timeDomain == 'tktf': # Evaluate at Xf
                X_eval = Xs[-1]
            elif timeDomain == 'regime': # Evaluate at X_regime end
                X_eval = Xs[ii+1]
            elif timeDomain == 't0tf': # Evaluate at Xf
                X_eval = Xs[-1]
            eta1, eta2, eta3, eta4 = get_Eta_Tensors(3, X_eval, q_func, params)
            etas = [eta1, eta2, eta3, eta4]
            HOCGT_nonsym = getQCGTs_nonSym(STTs[ii + 1], order=3, Xf=X_eval, q_func=q_func, params=params, etas=etas)
        else:
            raise ValueError(f"Unsupported cgtType: {cgtType}")
        HOCGTs_nonsym_list.append(HOCGT_nonsym)

    for ii, HOCGTs_list in enumerate(HOCGTs_nonsym_list):
        for jj, HOCGT in enumerate(HOCGTs_list):
            # Save each HOCGT as a .mat file
            saveTag = f'{cgtType}_{timeDomain}_{ii}_O{jj + 1}' if savePrefix is None else f'{savePrefix}_{cgtType}_{timeDomain}_{ii}_O{jj + 1}'
            savemat(os.path.join(dataPath, f'{saveTag}_{runTag}.mat'), {f'HOCGT{jj}': HOCGT})

def main():
    getHOCGTsFlag = True
    getSCGTsFlag = True
    getQCGTsFlag = True

    t0tkFlag = True
    tktkp1Flag = True
    tktfFlag = True
    regimeFlag = True
    t0tFFlag = True

    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'

    # runTag = 'JP_dstts_nominal'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts()
    # runTag = 'JP_dstts_nominal_drag_mod'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_drag_mod()
    runTag = 'JP_dstts_nominal_fast'
    ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_larger_perts_fast()
    figPath, dataPath = createPaths(basePath, runTag)

    # Load in all STTs
    STTs_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)['STTs']
    Xs_nd_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)['Xs_nd']
    STTs_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)['STTs']
    Xs_nd_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)['Xs_nd']
    STTs_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)['STTs']
    Xs_nd_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)['Xs_nd']
    if regimeFlag:
        STTs_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)['STTs']
        Xs_nd_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)['Xs_nd']
    STTs_t0tf = np.load(f"{dataPath}/STTs_t0tf.npz", allow_pickle=True)['STTs']
    Xs_nd_t0tf = np.load(f"{dataPath}/STTs_t0tf.npz", allow_pickle=True)['Xs_nd']

    # HOCGTs
    if getHOCGTsFlag:
        ## t0tk
        if t0tkFlag:
            get_and_save_HOCGT(STTs_t0tk, 'HOCGT', 't0tk', runTag, dataPath, params)
        ## tktkp1
        if tktkp1Flag:
            get_and_save_HOCGT(STTs_tktkp1, 'HOCGT', 'tktkp1', runTag, dataPath, params)
        ## tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'HOCGT', 'tktf', runTag, dataPath, params)
        ## Regime
        if regimeFlag:
            get_and_save_HOCGT(STTs_regime, 'HOCGT', 'regime', runTag, dataPath, params)
        ## t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'HOCGT', 't0tf', runTag, dataPath, params)

    # sCGTs
    if getSCGTsFlag:
        ## S for r, v, and gamma
        S_rvg = np.array([[1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0]])
        ## t0tk
        if t0tkFlag:
            get_and_save_HOCGT(STTs_t0tk, 'sCGT', 't0tk', runTag, dataPath, params, S_rvg)
        ## tktkp1
        if tktkp1Flag:
            get_and_save_HOCGT(STTs_tktkp1, 'sCGT', 'tktkp1', runTag, dataPath, params, S_rvg)
        ## tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'sCGT', 'tktf', runTag, dataPath, params, S_rvg)
        ## Regime
        if regimeFlag:
            get_and_save_HOCGT(STTs_regime, 'sCGT', 'regime', runTag, dataPath, params, S_rvg)
        ## t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'sCGT', 't0tf', runTag, dataPath, params, S_rvg)

        ## S for r only
        S_r = np.array([[1, 0, 0, 0, 0, 0, 0]])
        ## t0tk
        if t0tkFlag:
            get_and_save_HOCGT(STTs_t0tk, 'sCGT', 't0tk', runTag, dataPath, params, S_r, savePrefix='rOnly')
        ## tktkp1
        if tktkp1Flag:
            get_and_save_HOCGT(STTs_tktkp1, 'sCGT', 'tktkp1', runTag, dataPath, params, S_r, savePrefix='rOnly')
        ## tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'sCGT', 'tktf', runTag, dataPath, params, S_r, savePrefix='rOnly')
        ## Regime
        if regimeFlag:
            get_and_save_HOCGT(STTs_regime, 'sCGT', 'regime', runTag, dataPath, params, S_r, savePrefix='rOnly')
        ## t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'sCGT', 't0tf', runTag, dataPath, params, S_r, savePrefix='rOnly')

    # qCGTs
    if getQCGTsFlag:
        ## Energy
        func = energy_func
        ### t0tk
        if t0tkFlag:
            get_and_save_HOCGT(STTs_t0tk, 'qCGT', 't0tk', runTag, dataPath, params, q_func=func, Xs=Xs_nd_t0tk, savePrefix='energy')
        ### tktkp1
        if tktkp1Flag:
            get_and_save_HOCGT(STTs_tktkp1, 'qCGT', 'tktkp1', runTag, dataPath, params, q_func=func, Xs=Xs_nd_tktkp1, savePrefix='energy')
        ### tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'qCGT', 'tktf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_tktf, savePrefix='energy')
        ### Regime
        if regimeFlag:
            get_and_save_HOCGT(STTs_regime, 'qCGT', 'regime', runTag, dataPath, params, q_func=func, Xs=Xs_nd_regime, savePrefix='energy')
        ### t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'qCGT', 't0tf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_t0tf, savePrefix='energy')

        ## Apoapsis
        func = apoRad_func
        #### tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'qCGT', 'tktf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_tktf, savePrefix='apoRad')
        #### t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'qCGT', 't0tf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_t0tf, savePrefix='apoRad')

        ## Position
        func = pos_func
        ### t0tk
        if t0tkFlag:
            get_and_save_HOCGT(STTs_t0tk, 'qCGT', 't0tk', runTag, dataPath, params, q_func=func, Xs=Xs_nd_t0tk,
                           savePrefix='pos')
        ### tktkp1
        if tktkp1Flag:
            get_and_save_HOCGT(STTs_tktkp1, 'qCGT', 'tktkp1', runTag, dataPath, params, q_func=func, Xs=Xs_nd_tktkp1,
                           savePrefix='pos')
        ### tktf
        if tktfFlag:
            get_and_save_HOCGT(STTs_tktf, 'qCGT', 'tktf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_tktf,
                           savePrefix='pos')
        ### Regime
        if regimeFlag:
            get_and_save_HOCGT(STTs_regime, 'qCGT', 'regime', runTag, dataPath, params, q_func=func, Xs=Xs_nd_regime,
                           savePrefix='pos')
        ### t0tf
        if t0tFFlag:
            get_and_save_HOCGT(STTs_t0tf, 'qCGT', 't0tf', runTag, dataPath, params, q_func=func, Xs=Xs_nd_t0tf,
                           savePrefix='pos')

if __name__ == "__main__":
    main()