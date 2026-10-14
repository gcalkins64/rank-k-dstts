# Created on 8/20/2026
# Purpose:
#   1. Load second-order (order=2) reference STTs for the Uranus KLE-augmented state.
#   2. Construct four flavors of 2nd-order DSTTs from those STTs:
#        a. "regular" DSTT   -- 2nd-order HOCGT (STM-based) max-eigenvector direction,
#                                direction determined using only the first 6 physical states.
#        b. "ho"      DSTT   -- 3rd-order HOCGT max-eigenvector direction,
#                                direction determined using only the first 6 physical states.
#        c. "regular" DSTT   -- 2nd-order HOCGT max-eigenvector direction,
#                                direction determined using the full augmented state
#                                (6 physical states + KLE coefficients).
#        d. "ho"      DSTT   -- 3rd-order HOCGT max-eigenvector direction,
#                                direction determined using the full augmented state.
#      In all four cases the DSTT itself is built (and later propagated) over the FULL
#      augmented state; the "6-state" variants only restrict which sub-block of the STT
#      is used to pick the compression direction (the resulting direction vector is
#      zero-padded over the KLE-coefficient dimensions before being fed to STTstoDSTTs).
#   3. Run GMM_UQ with prescribed intermediate splits at t=90s and t=180s for each of the
#      four DSTT variants, and compare against Linear, UT, and (full order-2) STT
#      propagation. Each DSTT variant can either use precomputed DSTT tensors, or be
#      integrated along the way (tensors allocated/filled on the fly) using only the
#      computed R direction -- see `integrateAlongTheWayFlag` below.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time
import concurrent.futures
import datetime

from multiprocessing import Pool
from functools import partial
from dev_dstt_gmm.plotters import *
from dev_dstt_gmm.inputs.inputs import uranus_inputs_KLE
from dev_dstt_gmm.util.data_manager import createPaths
from dev_dstt_gmm.util.kle import getEigenpairsKLE, getKLEfun
from dev_dstt_gmm.util.stt import (
    getSTTs_general, getNominalTrajNonDimKLE, reshape_STT, reshape_DSTT,
    STTstoDSTTs, compute_dx_STT, compute_dx_XDSTT,
)
from dev_dstt_gmm.util.hocgts import getHOCGTs_nonSym, getSCGTs_nonSym, getQCGTs_nonSym, get_Eta_Tensors
from dev_dstt_gmm.util.tensor_toolbox import symmetrize, compute_eigenpairs_multiguess
from dev_dstt_gmm.util.nonDim import *
from dev_dstt_gmm.util.params_planet import *
from dev_dstt_gmm.util.gmm_uq import *
from dev_dstt_gmm.util.data_manager import load_case_tag


# --------------------------------------------------------------------------------------
# Helpers: restricting an STT to a sub-block of states, and building DSTT directions
# --------------------------------------------------------------------------------------

def reduce_STT_to_substates(STT_flat, order, stateSize_full, keep_idx):
    """
    Restrict a flattened order-`order` STT (built for `stateSize_full` states) to the
    sub-block of state components given by `keep_idx`, e.g. dropping the appended KLE
    coefficient dimensions to isolate the physical-state-only coupling.

    Returns a flattened STT over len(keep_idx) states (same tensor order).
    """
    tensors = reshape_STT(STT_flat, order, stateSize_full)
    reduced = []
    for o, T in enumerate(tensors, start=1):
        idx = np.ix_(*([keep_idx] * (o + 1)))
        reduced.append(T[idx])
    return np.concatenate([T.flatten() for T in reduced])


def build_directions(
    STTs_ref,
    order,
    stateSize_full,
    keep_idx=None,
    desc='',
    n_eigenvectors=1,
):
    """
    For each propagated time step in STTs_ref (row 0 is the identity / t0 entry and is
    skipped), compute:

        - the "regular" directions: the first `n_eigenvectors` eigenvectors of the
          2nd-order (STM-only) HOCGT.
        - the "ho" direction: the dominant eigenvector of the 3rd-order HOCGT, built
          using the full order-2 STT (STM + 2nd-order term).

    If `keep_idx` is given, the eigen-problem is solved on the sub-block of states
    given by keep_idx only. The resulting direction vectors are then zero-padded
    back out to the full `stateSize_full` dimension.

    Parameters
    ----------
    STTs_ref : ndarray
        STT array containing the propagated STTs.
    order : int
        STT order.
    stateSize_full : int
        Full state dimension.
    keep_idx : array-like, optional
        Indices of states to retain for the eigen-problem.
    desc : str, optional
        Description for the progress bar.
    n_eigenvectors : int, optional
        Number of regular eigenvectors to retain. Default is 1.

    Returns
    -------
    regular_Rs : ndarray
        Shape (n_steps, n_eigenvectors, stateSize_full).
        Contains the first `n_eigenvectors` regular directions.

    ho_Rs : ndarray
        Shape (n_steps, 1, stateSize_full).
        Contains the dominant HO direction.
    """

    if n_eigenvectors < 1:
        raise ValueError("n_eigenvectors must be >= 1")

    n_steps = STTs_ref.shape[0] - 1

    regular_Rs = np.zeros(
        (n_steps, n_eigenvectors, stateSize_full)
    )
    ho_Rs = np.zeros(
        (n_steps, 1, stateSize_full)
    )

    for k in trange(n_steps, desc=f'Building DSTT directions {desc}'):

        STT_k_full = STTs_ref[k + 1, :]

        if keep_idx is not None:
            STT_k_sub = reduce_STT_to_substates(
                STT_k_full,
                order,
                stateSize_full,
                keep_idx
            )
            sub_dim = len(keep_idx)
        else:
            STT_k_sub = STT_k_full
            sub_dim = stateSize_full

        if n_eigenvectors > sub_dim:
            raise ValueError(
                f"n_eigenvectors={n_eigenvectors} exceeds the "
                f"available state dimension ({sub_dim})."
            )

        # Build HOCGTs
        HOCGT_full = getHOCGTs_nonSym(
            STT_k_sub,
            order,
            sub_dim
        )

        HOCGT_full_sym = [
            symmetrize(H) for H in HOCGT_full
        ]

        lams, evecs, _, _, _ = compute_eigenpairs_multiguess(
            [HOCGT_full_sym],
            sub_dim,
            order,
            verbFlag=False
        )

        # ---------------------------------------------------------
        # Regular directions: first d eigenvectors of 2nd-order HOCGT
        # ---------------------------------------------------------

        # evecs[0] corresponds to the regular / 2nd-order HOCGT.
        #
        # Expected shape:
        #     evecs[0] -> (sub_dim, sub_dim)
        #
        # Columns are eigenvectors.
        e_reg, V_reg = np.linalg.eig(HOCGT_full_sym[0])
        # sort e_reg and get sorted indices
        sorted_indices = np.argsort(e_reg)[::-1]  # descending order
        sorted_V_reg = V_reg[:, sorted_indices]

        for j in range(n_eigenvectors):

            v_reg = sorted_V_reg[:, j]

            if keep_idx is not None:
                v_reg_full = np.zeros(stateSize_full)
                v_reg_full[keep_idx] = v_reg
            else:
                v_reg_full = v_reg

            regular_Rs[k, j, :] = v_reg_full

        # ---------------------------------------------------------
        # HO direction: dominant eigenvector of 3rd-order HOCGT
        # ---------------------------------------------------------

        V_ho = np.squeeze(evecs[1])
        v_ho = V_ho

        if keep_idx is not None:
            v_ho_full = np.zeros(stateSize_full)
            v_ho_full[keep_idx] = v_ho
        else:
            v_ho_full = v_ho

        ho_Rs[k, 0, :] = v_ho_full

    return regular_Rs, ho_Rs


def plot_R_over_time(t_prop, R_dict, state_labels, figPath=None, saveTag='R_over_time'):
    """
    Plot the DSTT compression direction R(t) over time for multiple methods, one
    subplot per state component, with a single shared legend distinguishing methods.

    Parameters
    ----------
    t_prop : array, shape (n_steps,)
        Time vector corresponding to the propagated (post-t0) direction samples.
    R_dict : dict[str, ndarray]
        Mapping of method name -> R array, each of shape (n_steps, 1, stateSize_full)
        (as returned by build_directions).
    state_labels : list of str, len stateSize_full
        Label for each state component (used as subplot titles).
    figPath : str or None
        If given, the figure is saved to this directory.
    saveTag : str
        Filename tag used when saving.
    """
    method_names = list(R_dict.keys())
    stateSize_full = next(iter(R_dict.values())).shape[-1]
    linestyles = ['-', '--', '-.', ':'] * (len(method_names) // 4 + 1)

    n_cols = 4
    n_rows = int(np.ceil(stateSize_full / n_cols))

    colors = plt.get_cmap('tab10')
    method_colors = {name: colors(i) for i, name in enumerate(method_names)}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for comp in range(stateSize_full):
        row, col = comp // n_cols, comp % n_cols
        ax = axes[row, col]

        for name in method_names:
            R = np.squeeze(R_dict[name], axis=1)  # (n_steps, stateSize_full)
            ax.plot(t_prop, abs(R[:, comp]), color=method_colors[name], label=name, linestyle=linestyles[method_names.index(name)])

        label = state_labels[comp] if comp < len(state_labels) else f'p{comp - len(state_labels)}'
        ax.set_title(label)
        ax.grid(True)
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)')

    # Hide any unused subplots
    for comp in range(stateSize_full, n_rows * n_cols):
        row, col = comp // n_cols, comp % n_cols
        axes[row, col].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5))

    fig.suptitle('DSTT Direction Abs(R(t)) Component-wise Comparison', fontsize=14)

    if figPath is not None:
        fig.savefig(os.path.join(figPath, f'{saveTag}.png'), dpi=300, bbox_inches='tight')

    return fig, axes

def run_single_case(case, prop_configs, ts, params, x0_augmented, P0_dim_augmented, dataPath, runTag, split_method, num_splits):
    split_config = SplitConfig(method=split_method)
    split_crit = SplitCriterionConfig(criterion='manual')
    init_split_cfg = InitSplitConfig(enabled=False)
    intermed_split_cfg = IntermedSplitConfig(enabled=True, multi=1, time=[100,200])
    proc_noise = ProcNoiseConfig(enabled=False)
    multi_ref_cfg = MultiRefConfig(enabled=False)

    prop_config = list(prop_configs.values())[case]
    name = list(prop_configs.keys())[case]
    print(f'\n--- Running GMM_UQ: {name} ---')
    gmm_uq = GMM_UQ(
        ts=ts,
        params=params,
        init_mean=x0_augmented,
        init_cov=P0_dim_augmented,
        num_splits=num_splits,
        split_config=split_config,
        prop_config=prop_config,
        proc_noise=proc_noise,
        split_crit=split_crit,
        init_split_cfg=init_split_cfg,
        intermed_split_cfg=intermed_split_cfg,
        multi_ref_cfg=multi_ref_cfg,
    )
    gmm_uq.run()
    gmm_uq.printPerformanceMetrics()
    gmm_uq.saveSelf(dataPath, f'{runTag}_{name}')

def main():
    # ------------------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------------------
    buildDirectionsAndDSTTsFlag = False   # (re)compute the 4 DSTT variants from the STTs
    runGMMUQFlag = True                  # run GMM_UQ for every propagation method
    useRay = False  # DO NOT TURN ON WHEN PARALLELIZATION FOR MIXANDS IS ON
    n_proc = 10
    plotComparisonFlag = True            # generate comparison plots after running

    # If True: propagate each DSTT/hoDSTT variant by integrating the DSTT along the way
    # each step (tensors allocated internally as a NaN buffer and filled in during
    # propagation), using only the R direction computed below.
    # If False: use the DSTT tensors precomputed up front via STTstoDSTTs (tensors=...).
    integrateAlongTheWayFlag = False
    propParFlag = True

    stt_order = 2    # order of the *loaded* reference STTs (must match how they were generated)
    dstt_order = 2   # order of the DSTTs to construct from those STTs

    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'
    runTag = 'DSTT_GMMUQ_ESAGNC_comp_prop_methods_inc_dist_single_ref'
    # runTag = 'TEST_DSTT_GMMUQ_comparison_Uranus_larger_dist_alongtheway_debug_DSTTs'
    DSTTloadTag = 'DSTT_GMMUQ_ESAGNC_2d_DSTT'
    STTloadTag = 'prop_w_KLE_Uranus_larger_dist'   # run tag under which the order-2 STTs were generated/saved
    MC_Tag = 'prop_w_KLE_Uranus_larger_dist'

    figPath, dataPath = createPaths(basePath, runTag)
    STTloadPath = os.path.join(basePath, 'data', STTloadTag)
    DSTTloadPath = os.path.join(basePath, 'data', DSTTloadTag)

    # ------------------------------------------------------------------------------
    # Base inputs / KLE setup (mirrors TEST_prop_w_KLE_Uranus.py)
    # ------------------------------------------------------------------------------
    ts, m0, params, init_cov, _ = uranus_inputs_KLE()

    kle_data = np.load(os.path.join(basePath, 'data', 'TEST_get_KLE_Uranus', 'KLE_eigenpairs.npz'))
    evals = kle_data['evals']
    evecs = kle_data['evecs']
    rhoSampMean = kle_data['rhoSampMean']
    hRho = kle_data['h']
    d = 10

    p0 = np.zeros(d)
    x0_augmented = np.concatenate((m0, p0))
    stateSize_full = len(x0_augmented)
    keep_idx_6 = np.arange(6)

    P0_dim_augmented = np.block([
        [init_cov, np.zeros((len(m0), d))],
        [np.zeros((d, len(m0))), np.diag(evals[:d])]
    ])

    params['d_KLE'] = d
    params['KLE_eigvals'] = evals[:d]
    params['KLE_eigfuncs_interp'] = evecs[:, :d]
    params['eomTag'] = 'IntAtmoNonDim_KLE'
    params['hs_interp'] = hRho
    params['rho_interp'] = rhoSampMean

    # ------------------------------------------------------------------------------
    # 1. Load the second-order reference STTs
    # ------------------------------------------------------------------------------
    data = np.load(os.path.join(STTloadPath, 'STT_tktkp1_reference.npz'), allow_pickle=True)
    STTs_ref = data['STTs']       # order-2 STT series, shape (len(ts), n_flat)
    Xs_d_ref = data['Xs_dim']     # dimensional nominal trajectory
    Xs_nd_ref = data['Xs_nd']     # non-dimensional nominal trajectory

    dT = ts[1] - ts[0]
    Tf = ts[-1]
    t_prop = ts[1:]

    # ------------------------------------------------------------------------------
    # 2. Construct the four DSTT directions (and, if not integrating along the way,
    #    the precomputed DSTT tensors themselves)
    # ------------------------------------------------------------------------------
    # dstt_file = os.path.join(dataPath, 'DSTT_variants.npz')
    dstt_file = os.path.join(DSTTloadPath, 'DSTT_variants.npz')  # TODO

    if buildDirectionsAndDSTTsFlag:
        regular_Rs_6, ho_Rs_6 = build_directions(
            STTs_ref, stt_order, stateSize_full, keep_idx=keep_idx_6, desc='(6-state)', n_eigenvectors=2
        )
        regular_Rs_full, ho_Rs_full = build_directions(
            STTs_ref, stt_order, stateSize_full, keep_idx=None, desc='(full augmented state)', n_eigenvectors=2
        )

    #     # Plot the 4 direction sets over time for comparison
    #     wsNames = [f'w{i}' for i in range(d)]
    #     _, stateSymbols, _ = stateProps()
    #     state_labels_full = stateSymbols[:-1] + wsNames  # drop the ln(rho) symbol used for atmo-only states
    #     plot_R_over_time(
    #         t_prop,
    #         {
    #             'DSTT_reg_6state': regular_Rs_6,
    #             'hoDSTT_6state': ho_Rs_6,
    #             'DSTT_reg_fullstate': regular_Rs_full,
    #             'hoDSTT_fullstate': ho_Rs_full,
    #         },
    #         state_labels_full,
    #         figPath=figPath,
    #     )
    #     plt.close('all')

        if not integrateAlongTheWayFlag:
            # Only bother precomputing the DSTT tensors up front if we're actually going
            # to use them -- if integrating along the way, they're allocated/filled
            # internally by GMM_UQ instead.
            print('Constructing DSTT (2nd-order HOCGT direction, 6-state) ...')
            DSTT_reg_6 = STTstoDSTTsLargerLatent(t_prop, STTs_ref[1:, :], regular_Rs_6, dstt_order, stateSize_full)

            print('Constructing hoDSTT (3rd-order HOCGT direction, 6-state) ...')
            DSTT_ho_6 = STTstoDSTTsLargerLatent(t_prop, STTs_ref[1:, :], ho_Rs_6, dstt_order, stateSize_full)

            print('Constructing DSTT (2nd-order HOCGT direction, full augmented state) ...')
            DSTT_reg_full = STTstoDSTTsLargerLatent(t_prop, STTs_ref[1:, :], regular_Rs_full, dstt_order, stateSize_full)

            print('Constructing hoDSTT (3rd-order HOCGT direction, full augmented state) ...')
            DSTT_ho_full = STTstoDSTTsLargerLatent(t_prop, STTs_ref[1:, :], ho_Rs_full, dstt_order, stateSize_full)
        else:
            DSTT_reg_6 = DSTT_ho_6 = DSTT_reg_full = DSTT_ho_full = None

        np.savez(
            dstt_file,
            regular_Rs_6=regular_Rs_6, ho_Rs_6=ho_Rs_6,
            regular_Rs_full=regular_Rs_full, ho_Rs_full=ho_Rs_full,
            DSTT_reg_6=DSTT_reg_6, DSTT_ho_6=DSTT_ho_6,
            DSTT_reg_full=DSTT_reg_full, DSTT_ho_full=DSTT_ho_full,
            allow_pickle=True,
        )
    else:
        data = np.load(dstt_file, allow_pickle=True)
        regular_Rs_6 = data['regular_Rs_6']
        ho_Rs_6 = data['ho_Rs_6']
        regular_Rs_full = data['regular_Rs_full']
        ho_Rs_full = data['ho_Rs_full']
        DSTT_reg_6 = data['DSTT_reg_6']
        DSTT_ho_6 = data['DSTT_ho_6']
        DSTT_reg_full = data['DSTT_reg_full']
        DSTT_ho_full = data['DSTT_ho_full']

    # ------------------------------------------------------------------------------
    # 3. GMM_UQ setup: prescribed intermediate splits at t=90s and t=180s
    # ------------------------------------------------------------------------------
    num_splits = 3
    split_method = 'covariance'

    # DSTT / hoDSTT variants: either integrate along the way (R direction only) or use
    # the precomputed tensors built above.
    dstt_direction_sets = {
        'DSTT_reg_6state': (DSTT_reg_6, regular_Rs_6),
        # 'hoDSTT_6state': (DSTT_ho_6, ho_Rs_6),
        # 'DSTT_reg_fullstate': (DSTT_reg_full, regular_Rs_full),
        # 'hoDSTT_fullstate': (DSTT_ho_full, ho_Rs_full),
    }

    prop_configs = {}

    # Non-DSTT propagation methods TODO
    # Make sure propParFlag and useRay are not both true or the evil will come out
    if propParFlag and useRay:
        raise ValueError(f"propParFlag is {propParFlag} and useRay is {useRay}. Both cannot be true at the same time.")


    # DSTTs
    for name, (DSTT_tensors, R) in dstt_direction_sets.items():
        prop_configs[name] = PropConfig(
            method='xDSTT', integrateAlongTheWay=integrateAlongTheWayFlag,
            tensors=DSTT_tensors, Xs_nom=Xs_nd_ref, order=dstt_order,
            R=R, propParFlag=propParFlag, max_workers=n_proc, latentDim=2
        )

    prop_configs['Linear'] = PropConfig(
            method='STT', integrateAlongTheWay=integrateAlongTheWayFlag,
            tensors=STTs_ref, Xs_nom=Xs_nd_ref, order=1, propParFlag=propParFlag, max_workers=n_proc)
    prop_configs['STT'] = PropConfig(
            method='STT', integrateAlongTheWay=integrateAlongTheWayFlag,
            tensors=STTs_ref, Xs_nom=Xs_nd_ref, order=stt_order, propParFlag=propParFlag, max_workers=n_proc
        )
    prop_configs['UT'] = PropConfig( method='UT', propParFlag=propParFlag, max_workers=n_proc)

    gmm_uq_results = {}

    if runGMMUQFlag:
        if useRay:
            tic = time.perf_counter()
            complete = 0
            results = list()
            sum_time = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
                futures = [
                    executor.submit(run_single_case, case, prop_configs, ts, params, x0_augmented, P0_dim_augmented, dataPath, runTag, split_method, num_splits) for case in
                    range(len(prop_configs))]
                for future in concurrent.futures.as_completed(futures):
                    run_time = future.result()
                    complete += 1
                    avg_time = (sum_time + run_time) / (complete + 1)
                    time_left = (len(prop_configs) - complete) * avg_time / n_proc
                    sum_time += run_time
                    toc = time.perf_counter()
                    now = datetime.datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    print(
                        f'{dt_string} {complete} of {len(prop_configs)}, loop time = {run_time / 60:0.3f} min, estimated time left = {time_left / 60 / 60:0.4f} hrs',
                        flush=True)

            toc = time.perf_counter()
            print(f"Time to compute GMM mixtures: {(toc - tic) / 60:0.4f} mins")
        else:
            for case in range(len(prop_configs)):
                run_single_case(case, prop_configs, ts, params, x0_augmented, P0_dim_augmented, dataPath, runTag, split_method, num_splits)

    # Load in data
    for name in prop_configs:
        gmm_uq_results[name] = np.load(
            os.path.join(
                dataPath,
                f'gmm_uq_{prop_configs[name].method}_{split_method}_manual_{num_splits}_'
                f'DENone_ISFalse_{runTag}_{name}.npy'
            ),
            allow_pickle=True,
        ).item()

    # ------------------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------------------
    N_MC = 100
    snapshot_time_inds = [33, len(ts)-2]
    print('Loading KLE Data')
    dataPathKLE = os.path.join(basePath, 'data', MC_Tag)
    load_case_with_path = partial(load_case_tag, dataPath=dataPathKLE, tag='kle')
    tic = time.perf_counter()
    with Pool(processes=10) as pool:
        KLE_data = pool.map(load_case_with_path, range(N_MC))
    toc = time.perf_counter()
    print(f'KLE Data Loaded in = {toc - tic:0.2f} seconds')
    MC_Data = np.array([KLE_data[i]['X'] for i in range(N_MC)])  # Nruns x stateSize x times

    if plotComparisonFlag:
        # Order/labels for the comparison plot -- only methods actually present are used
        method_order = [
            'STT', 'Linear', 'UT',
            'DSTT_reg_6state', 'hoDSTT_6state',
            'DSTT_reg_fullstate', 'hoDSTT_fullstate',
        ]
        method_labels = [
            'STT', 'Lin', 'UT',
            'DSTT (reg, 6-state)', 'hoDSTT (6-state)',
            'DSTT (reg, full)', 'hoDSTT (full)',
        ]
        gmm_uq_list = [gmm_uq_results[name] for name in method_order if name in gmm_uq_results]
        labels_present = [lbl for name, lbl in zip(method_order, method_labels) if name in gmm_uq_results]

        for t_ind in snapshot_time_inds:
            fig, axs = plotTkDistribution_subplots(
                gmm_uq_list, MC_Data, 3, 5, labels_present,
                list(range(stateSize_full)), ts[t_ind], ts,
                MCOn=True, CIOn=True, covOn=False, legendOn=True,
            )
            plt.suptitle(f't = {ts[t_ind]} s')
            plt.savefig(
                os.path.join(figPath, f'{runTag}_distribution_V_psi_comparison_all_methods_t{t_ind}.png'),
                dpi=400, transparent=False,
            )
            fig, axs = plotTkDistribution_subplots(
                gmm_uq_list, MC_Data, 3, 5, labels_present,
                list(range(stateSize_full)), ts[t_ind], ts,
                MCOn=True, CIOn=True, legendOn=True, plotSelectedCovariance=True,
                selectedMixandInd=12, #annotateMixandNum=True
            )
            plt.suptitle(f't = {ts[t_ind]} s')
            plt.savefig(
                os.path.join(figPath, f'{runTag}_distribution_V_psi_comparison_all_methods_sel_cov_t{t_ind}.png'),
                dpi=400, transparent=False,
            )
            fig, axs = plotTkDistribution_subplots(
                gmm_uq_list, MC_Data, 0, 3, labels_present,
                list(range(stateSize_full)), ts[t_ind], ts,
                MCOn=True, CIOn=True, covOn=False, legendOn=True,
            )
            plt.suptitle(f't = {ts[t_ind]} s')
            plt.savefig(
                os.path.join(figPath, f'{runTag}_distribution_r_V_comparison_all_methods_t{t_ind}.png'),
                dpi=400, transparent=False,
            )

            plt.close('all')

        # --- Animated state evolution over time, one animation per method ---
        t_inds = np.arange(0, len(ts) - 1, 1)
        method_names_present = [name for name in method_order if name in gmm_uq_results]
        for name, gmm_uq in zip(method_names_present, gmm_uq_list):
            animateStateEvolution(
                MC_Data, gmm_uq, t_inds, ts, [0,3], figPath,
                f'{runTag}_{name}', name,
                plotMixandsIndividually=False, useContourFlag=True, MCCovarianceFlag=False,
            )
            animateStateEvolution(
                MC_Data, gmm_uq, t_inds, ts, [0, 3], figPath,
                f'{runTag}_{name}_indiv', name,
                plotMixandsIndividually=True, useContourFlag=False, MCCovarianceFlag=False,
            )
            plt.close('all')



if __name__ == "__main__":
    main()
    plt.show()