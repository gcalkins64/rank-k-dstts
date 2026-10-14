# Author: Grace Calkins
# Date: 08/15/2025
# Purpose: Compute and evaluate STTs for aerocapture trajectory

import numpy as np
from JP_stts_aerocapture.inputs.inputs import uranus_inputs_JP_aero_stts, uranus_inputs_JP_aero_stts_drag_mod, uranus_inputs_JP_aero_stts_larger_perts_fast
from JP_stts_aerocapture.plotters import *
from JP_stts_aerocapture.util.data_manager import createPaths
from JP_stts_aerocapture.util.eoms_aerocapture import eoms3DOFLnDens
from JP_stts_aerocapture.util.stt import getSTTs_general, getNominalTrajLnDens, getPerturbedTrajSTTsNonDimTkTkp1, getPerturbedTrajSTTsNonDimT0Tk, getPerturbedTrajSTTsNonDimT0TkTkTf, reshape_STT
from JP_stts_aerocapture.util.nonDim import *

def main():
    getSTTsFlag = True
    t0tkFlag = True
    tktkp1Flag = True
    tktfFlag = True
    regimeFlag = True
    t0tFFlag = True
    order = 3

    # Get STTs for different time domains
    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'

    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts()
    # runTag = 'JP_dstts_nominal'
    # ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_drag_mod()
    # runTag = 'JP_dstts_nominal_drag_mod'
    runTag = 'JP_dstts_nominal_fast'
    ts, x0, params, init_cov, proc_cov = uranus_inputs_JP_aero_stts_larger_perts_fast()
    figPath, dataPath = createPaths(basePath, runTag)

    dT = ts[1] - ts[0]
    Tf = ts[-1]
    t0_tk_list = [[0, t] for t in np.arange(dT, Tf + dT, dT)]
    tk_tkp1_list = [[t, t + dT] for t in np.arange(0, Tf, dT)]
    tk_tf_list = [[t, Tf] for t in np.arange(0, Tf + dT, dT)][:-1]

    regime_data = np.load(f"{dataPath}/regime_times.npz", allow_pickle=True)
    phase2StartTime = regime_data['phase2StartTime']
    phase3StartTime = regime_data['phase3StartTime']
    t_regime_list = [[0, phase2StartTime], [phase2StartTime, phase3StartTime], [phase3StartTime, Tf]]

    t_t0_tf_list = [[0, Tf]]

    params_pass = params.copy()
    params_pass['eomTag'] = 'eoms3DOFLnDens'
    Xs_d = getNominalTrajLnDens(ts, x0, params_pass)
    Xs_nd = np.array([convertToNonDimDens(Xs_d[i], params) for i in range(Xs_d.shape[0])])

    plotAllStates(ts, [Xs_d.T], ['Nominal'], figPath, params, runTag)
    plotAltVel(Xs_d.T, figPath, params, runTag)
    plt.show()

    if getSTTsFlag:
        if t0tkFlag:
            Xs_nd_t0tk, Xs_dim_t0tk, STTs_t0tk = getSTTs_general(t0_tk_list, x0, order, params, params['eomTag'], mode='t0tk')
            np.savez(f"{dataPath}/STTs_t0tk.npz", STTs=STTs_t0tk, Xs_nd=Xs_nd_t0tk, Xs_dim=Xs_dim_t0tk, allow_pickle=True)
        if tktkp1Flag:
            Xs_nd_tktkp1, Xs_dim_tktkp1, STTs_tktkp1 = getSTTs_general(tk_tkp1_list, x0, order, params, params['eomTag'], mode='tktkp1', X_stt_ref_nd=Xs_nd)
            np.savez(f"{dataPath}/STTs_tktkp1.npz", STTs=STTs_tktkp1, Xs_nd=Xs_nd_tktkp1, Xs_dim=Xs_dim_tktkp1, allow_pickle=True)
        if tktfFlag:
            Xs_nd_tktf, Xs_dim_tktf, STTs_tktf = getSTTs_general(tk_tf_list, x0, order, params, params['eomTag'], mode='tktf', X_stt_ref_nd=Xs_nd)
            np.savez(f"{dataPath}/STTs_tktf.npz", STTs=STTs_tktf, Xs_nd=Xs_nd_tktf, Xs_dim=Xs_dim_tktf, allow_pickle=True)
        if regimeFlag:
            Xs_nd_regime, Xs_dim_regime, STTs_regime = getSTTs_general(t_regime_list, x0, order, params, params['eomTag'], mode='tktkp1')
            np.savez(f"{dataPath}/STTs_regime.npz", STTs=STTs_regime, Xs_nd=Xs_nd_regime, Xs_dim=Xs_dim_regime, allow_pickle=True)
        if t0tFFlag:
            Xs_nd_t0tf, Xs_dim_t0tf, STTs_t0tf = getSTTs_general(t_t0_tf_list, x0, order, params, params['eomTag'], mode='t0tk', X_stt_ref_nd=Xs_nd)
            np.savez(f"{dataPath}/STTs_t0tf.npz", STTs=STTs_t0tf, Xs_nd=Xs_nd_t0tf, Xs_dim=Xs_dim_t0tf, allow_pickle=True)

    data_t0tk = np.load(f"{dataPath}/STTs_t0tk.npz", allow_pickle=True)
    STTs_t0tk = data_t0tk['STTs']
    Xs_nd_t0tk = data_t0tk['Xs_nd']
    Xs_dim_t0tk = data_t0tk['Xs_dim']

    data_tktkp1 = np.load(f"{dataPath}/STTs_tktkp1.npz", allow_pickle=True)
    STTs_tktkp1 = data_tktkp1['STTs']
    Xs_nd_tktkp1 = data_tktkp1['Xs_nd']
    Xs_dim_tktkp1 = data_tktkp1['Xs_dim']

    data_tktf = np.load(f"{dataPath}/STTs_tktf.npz", allow_pickle=True)
    STTs_tktf  = data_tktf['STTs']
    Xs_nd_tktf  = data_tktf['Xs_nd']
    Xs_dim_tktf  = data_tktf['Xs_dim']

    data_regime = np.load(f"{dataPath}/STTs_regime.npz", allow_pickle=True)
    STTs_regime = data_regime['STTs']
    Xs_nd_regime = data_regime['Xs_nd']
    Xs_dim_regime = data_regime['Xs_dim']

    data_t0tf = np.load(f"{dataPath}/STTs_t0tf.npz", allow_pickle=True)
    STTs_t0tf  = data_t0tf['STTs']
    Xs_nd_t0tf  = data_t0tf['Xs_nd']
    Xs_dim_t0tf  = data_t0tf['Xs_dim']


    # Plot the STT expansions
    # plotSTTs(x0, ts, STTs_t0tk, figPath, f'{runTag}_t0tk', titleTag=' t0 tk')
    # plotSTTs(x0, ts, STTs_tktkp1, figPath, f'{runTag}_tktkp1', titleTag=' tk tkp1')
    # plotSTTs(x0, ts, STTs_tktf[1:], figPath, f'{runTag}_tktf', titleTag=' tk tf')
    # plotSTTs(x0, [0, phase2StartTime, phase3StartTime], STTs_regime[1:], figPath, f'{runTag}_regime', titleTag=' regime')

    # Compare all Xs from all the different methods
    plotCompareDxDens([Xs_dim_t0tk, Xs_dim_tktkp1, Xs_dim_tktf, Xs_d], ts, ['t0tk', 'tktkp1', 'tktf', 'Int'], None, figPath, runTag + 'X_refs_compare')
    # plt.show()

    # Check a perturbation propagation between all STTs and compare to numerical integration
    var = 1e-14  # perturbation size to keep within region of linearity
    np.random.seed(42)
    dx0_nd = np.random.multivariate_normal(np.zeros(len(x0)), np.eye(len(x0)) * var)
    print(np.linalg.norm(dx0_nd))
    dx0_dim = convertFromNonDimDens(dx0_nd, params)
    formatted = [f"{x:.2e}" for x in dx0_dim]
    print("dx0_dim:", formatted)

    # Get integrated solutions
    Xs_pert_d = getNominalTrajLnDens(ts, x0 + dx0_dim, params)
    dx_int_d = Xs_pert_d - Xs_d

    # Get STT solutions for t0tk and tktkp1 and t0tf
    Xs_pert_stt_d_tktkp11, dxs_stt_d_tktkp11 = getPerturbedTrajSTTsNonDimTkTkp1(dx0_dim, 1, STTs_tktkp1, Xs_nd_tktkp1,
                                                                              params)
    Xs_pert_stt_d_t0tk1, dxs_stt_d_t0tk1 = getPerturbedTrajSTTsNonDimT0Tk(dx0_dim, 1, STTs_t0tk, Xs_nd_t0tk, params)

    Xs_pert_stt_d_tktkp12, dxs_stt_d_tktkp12 = getPerturbedTrajSTTsNonDimTkTkp1(dx0_dim, 2, STTs_tktkp1, Xs_nd_tktkp1,
                                                                              params)
    Xs_pert_stt_d_t0tk2, dxs_stt_d_t0tk2 = getPerturbedTrajSTTsNonDimT0Tk(dx0_dim, 2, STTs_t0tk, Xs_nd_t0tk, params)

    Xs_pert_stt_d_tktkp1, dxs_stt_d_tktkp1 = getPerturbedTrajSTTsNonDimTkTkp1(dx0_dim, order, STTs_tktkp1, Xs_nd_tktkp1, params)
    Xs_pert_stt_d_t0tk, dxs_stt_d_t0tk = getPerturbedTrajSTTsNonDimT0Tk(dx0_dim, order, STTs_t0tk, Xs_nd_t0tk, params)
    Xs_pert_stt_d_t0tf, dxs_stt_d_t0tf = getPerturbedTrajSTTsNonDimT0Tk(dx0_dim, order, STTs_t0tf, Xs_nd_t0tf, params)

    # Get multi-segment STT solutions using the STTs from t0 to tk and from tk to tf
    Xs_pert_stt_d_t0tf, dxs_stt_d_t0tf = getPerturbedTrajSTTsNonDimT0TkTkTf(dx0_dim, order, STTs_t0tk, STTs_tktf, Xs_d, params)

    # Plot the tk results
    plotCompareDxDens([dxs_stt_d_tktkp1, dxs_stt_d_t0tk, dx_int_d], ts, ['tktkp1', 't0tk', 'Int'], None, figPath, runTag + 'STT_tkp1_t0tk_compare')
    # plt.show()

    plotStateDiff(ts, [dxs_stt_d_tktkp1.T, dxs_stt_d_t0tk.T, dx_int_d.T], ['tktkp1', 't0tk', 'Int'], [[0,2], [1,2]], figPath, params, runTag, title='STTs - Integrated')
    # plt.show()

    plotStateDiff(ts, [dxs_stt_d_tktkp1.T, dxs_stt_d_tktkp12.T, dxs_stt_d_tktkp11.T, dx_int_d.T], ['O3', 'O2', 'O1', 'Int'], [[0, -1], [1, -1], [2, -1]],
                  figPath, params, runTag + 'STT_order_compare_log', title='STTs Orders - Integrated', yscale='log', absFlag=True)
    plotStateDiff(ts, [dxs_stt_d_tktkp1.T, dxs_stt_d_tktkp12.T, dxs_stt_d_tktkp11.T, dx_int_d.T],
                  ['O3', 'O2', 'O1', 'Int'], [[0, -1], [1, -1], [2, -1]],
                  figPath, params, runTag + 'STT_order_compare_lin', title='STTs Orders - Integrated', yscale='linear',
                  absFlag=True)
    fig = plt.gcf()  # get current figure
    axes = fig.get_axes()  # list of all Axes in the figure
    for ax in axes:
        ax.set_ylim([0, 1e-8])
    axes[0].set_ylim([0,1e-6])
    plt.savefig(f"{figPath}/{runTag}_STT_order_compare_lin_zoom.png", dpi=300)
    # plt.show()

    stateNames, stateSymbols, _ = stateProps()
    # Plot the tf results
    fig, axs = plt.subplots(2, 4, figsize=(10, 7), sharex='col')
    for i in range(7):
        axs[i // 4, i % 4].plot(ts[2:], dxs_stt_d_t0tf[1:, i], color='C0', label='t0tk + tktf')
        axs[i // 4, i % 4].axhline(y=dx_int_d[1:, i][-1], color='C1', linestyle='--', label='Integrated')
        axs[i // 4, i % 4].axhline(y=dxs_stt_d_tktkp1[1:, i][-1], color='C2', linestyle='-.', label='tktkp1')
        axs[i // 4, i % 4].axhline(y=dxs_stt_d_t0tk[1:, i][-1], color='C3', linestyle=':', label='t0tk')
        axs[i // 4, i % 4].axhline(y=dxs_stt_d_t0tf[1:, i][-1], color='C4', linestyle='--', label='t0tf')
        axs[i // 4, i % 4].set_title(f'{stateSymbols[i]}')
    axs[1, 3].set_axis_off()
    plt.suptitle('Perturbation at the final time')
    plt.tight_layout()
    axs[1, 2].legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(f"{figPath}/{runTag}_STT_t0tk_tktf_compare.png", dpi=300)
    # plt.show()

    # Plot the tf results
    fig, axs = plt.subplots(2, 4, figsize=(10, 7), sharex='col')
    for i in range(7):
        axs[i // 4, i % 4].plot(ts[2:], abs(dxs_stt_d_t0tf[1:, i] - dx_int_d[1:, i][-1]) / dx_int_d[1:, i][-1], color='C0', label='t0tk + tktf')
        axs[i // 4, i % 4].axhline(y=abs(dxs_stt_d_tktkp1[1:, i][-1] - dx_int_d[1:, i][-1]) / dx_int_d[1:, i][-1], color='C1', linestyle='-.', label='tktkp1')
        axs[i // 4, i % 4].axhline(y=abs(dxs_stt_d_t0tk[1:, i][-1] - dx_int_d[1:, i][-1])/ dx_int_d[1:, i][-1], color='C2', linestyle=':', label='t0tk')
        axs[i // 4, i % 4].axhline(y=abs(dxs_stt_d_t0tf[1:, i][-1] - dx_int_d[1:, i][-1])/ dx_int_d[1:, i][-1], color='C3', linestyle='--', label='t0tf')
        axs[i // 4, i % 4].set_title(f'{stateSymbols[i]}')
    axs[1, 3].set_axis_off()
    plt.suptitle('Percent difference from integrated perturbation at final time')
    plt.tight_layout()
    axs[1, 2].legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(f"{figPath}/{runTag}_STT_t0tk_tktf_difference.png", dpi=300)

    # Check that STM t0tk and STT tktf = STM t0tf for all tk
    STMs_t0tf = []
    for ii in range(len(ts)-2):
        STM_t0tk = reshape_STT(STTs_t0tk[ii+1], order, len(x0))[0]
        STM_tktf = reshape_STT(STTs_tktf[ii+2], order, len(x0))[0]
        STM_t0tf = STM_tktf @ STM_t0tk
        STMs_t0tf.append(STM_t0tf)

    STM_t0tf_true = reshape_STT(STTs_t0tf[1,:], order, len(x0))[0]
    STM_t0tf_diff = np.array([STMs_t0tf[ii] - STM_t0tf_true for ii in range(len(STMs_t0tf))])
    # make a plot with a subplot for each element of the difference over time
    fig, axs = plt.subplots(7, 7, figsize=(10, 7), sharex='col')
    for i in range(7):
        for j in range(7):
            axs[i, j].plot(ts[2:], STM_t0tf_diff[:,i,j])
            axs[i, j].axhline(y=0, color='r', linestyle='--')
    plt.suptitle('STM t0tk * STM tktf - STM t0tf')
    plt.tight_layout()
    plt.savefig(f"{figPath}/{runTag}_STM_t0tk_tktf_compare.png", dpi=300)

    # Check that STM tktkp1 and STM t0tk are the same
    STMs_t0tk_constructed = []
    for ii in range(len(ts) - 1):
        STM_t0tk = np.eye(len(x0))
        for jj in range(ii + 1):
            STM_tktkp1 = reshape_STT(STTs_tktkp1[jj+1], order, len(x0))[0]
            STM_t0tk = STM_tktkp1 @ STM_t0tk
        STMs_t0tk_constructed.append(STM_t0tk)

    STM_t0tk_diff = []
    for ii in range(len(STMs_t0tk_constructed)-1):
        STM_t0tk_true = reshape_STT(STTs_t0tk[ii+1], order, len(x0))[0]
        STM_t0tk_diff.append(STMs_t0tk_constructed[ii] - STM_t0tk_true)

    STM_t0tk_diff = np.array(STM_t0tk_diff)
    # make a plot with a subplot for each element of the difference over time
    fig, axs = plt.subplots(7, 7, figsize=(10, 7), sharex='col')
    for i in range(7):
        for j in range(7):
            axs[i, j].plot(ts[2:], STM_t0tk_diff[:, i, j])
            axs[i, j].axhline(y=0, color='r', linestyle='--')
    plt.suptitle('STM tktkp1 * ... - STM t0tk')
    plt.tight_layout()
    plt.savefig(f"{figPath}/{runTag}_STM_tktkp1_t0tk_compare.png", dpi=300)

    # Check that STM tktkp1 and STM tktf are the same
    STMs_tktf_constructed = []
    for ii in range(len(ts) - 1):
        STM_tktf = np.eye(len(x0))
        for jj in range(len(ts)-(ii+1)):
            STM_tktkp1 = reshape_STT(STTs_tktkp1[ii+1 + jj], order, len(x0))[0]
            STM_tktf = STM_tktkp1 @ STM_tktf
        STMs_tktf_constructed.append(STM_tktf)

    STM_tktf_diff = []
    for ii in range(len(STMs_tktf_constructed) - 1):
        STM_tktf_true = reshape_STT(STTs_tktf[ii + 1], order, len(x0))[0]
        STM_tktf_diff.append(STMs_tktf_constructed[ii] - STM_tktf_true)

    STM_tktf_diff = np.array(STM_tktf_diff)
    # make a plot with a subplot for each element of the difference over time
    fig, axs = plt.subplots(7, 7, figsize=(10, 7), sharex='col')
    for i in range(7):
        for j in range(7):
            axs[i, j].plot(ts[2:], STM_tktf_diff[:, i, j])
            axs[i, j].axhline(y=0, color='r', linestyle='--')
    plt.suptitle('STM tktkp1 * ... - STM tktf')
    plt.tight_layout()
    plt.savefig(f"{figPath}/{runTag}_STM_tktkp1_tktf_compare.png", dpi=300)

if __name__ == "__main__":
    main()
    plt.show()
