# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for running GMM propagation

import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange
import itertools
import seaborn as sns
from scipy.stats import multivariate_normal, norm
import matplotlib as mpl
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
from dev_dstt_gmm.inputs import *
from dev_dstt_gmm.plotters import seabornSettings
from dev_dstt_gmm.util.nonDim import *
from dev_dstt_gmm.util.differential_entropy import *
from dev_dstt_gmm.util.gmm_splitting import *
from dev_dstt_gmm.util.stt import *
from dev_dstt_gmm.util.params_planet import *
from dev_dstt_gmm.util.eoms_aerocapture import *
import warnings
from dataclasses import dataclass
from typing import Optional, Any


from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np

@dataclass
class PropConfig:
    """Propagation-method config. tensors/R/order/Xs_nom are only needed
    when method != 'UT'. tensors specifically is only needed when
    integrateAlongTheWay is False (otherwise it's allocated internally)."""
    method: str  # 'UT', 'STT', 'DSTT', 'xDSTT'
    integrateAlongTheWay: bool = False
    tensors: Optional[np.ndarray] = None   # STTs / DSTTs / xDSTTs (nondimensional)
    R: Optional[np.ndarray] = None         # Rs / xRs — unused for 'STT'
    order: Optional[int] = None            # stt_order / dstt_order / xdstt_order
    Xs_nom: Optional[np.ndarray] = None    # nominal trajectory (nondimensional)
    propParFlag: bool = True  # whether to parallelize individual mixand propagation
    max_workers: Optional[int] = 4  # maximum number of parallel workers (if propParFlag is True)
    latentDim: Optional[int] = 1  # latent dimension for DSTT/xDSTT (default 1)

    def __post_init__(self):
        if self.method == 'UT':
            return  # nothing else needed
        if self.method not in ('STT', 'DSTT', 'xDSTT', 'Linear'):
            raise ValueError(f"Unknown prop method: {self.method!r}")
        if self.method in ('DSTT', 'xDSTT') and self.R is None:
            raise ValueError(f"R required when method == {self.method!r}")
        if self.integrateAlongTheWay:
            if self.order is None:
                raise ValueError(
                    f"order required when method == {self.method!r} "
                    f"and integrateAlongTheWay is True"
                )
        else:
            if self.tensors is None:
                raise ValueError(
                    f"tensors required when method == {self.method!r} "
                    f"and integrateAlongTheWay is False"
                )
            if self.Xs_nom is None:
                raise ValueError(
                    f"Xs_nom required when method == {self.method!r} "
                    f"and integrateAlongTheWay is True"
                )

@dataclass
class ProcNoiseConfig:
    enabled: bool
    proc_cov: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.enabled and self.proc_cov is None:
            raise ValueError("proc_cov is required when procNoiseFlag is enabled")

@dataclass
class SplitCriterionConfig:
    criterion: str
    DE_threshold: Optional[float] = None
    SOLC_threshold: Optional[float] = None
    max_mixands: Optional[int] = None
    trackDE: bool = False  # whether to track differential entropy
    trackPyest: bool = False  # whether to track the Pyest splitting metrics

    def __post_init__(self):
        if self.criterion == 'DE' and self.DE_threshold is None:
            raise ValueError("DE_threshold is required when split_criterion == 'DE'")
        if self.criterion == 'SOLC_R' and self.SOLC_threshold is None:
            raise ValueError("SOLC_threshold is required when split_criterion == 'SOLC_R'")


@dataclass
class SplitConfig:
    """Splitting config. splitDir is only needed when split_method involves
    anything other than plain covariance-based splitting."""
    method: Union[str, List[str]]  # can be a single method or a list
    splitDir: Optional[Any] = None

    def __post_init__(self):
        methods = self.method if isinstance(self.method, list) else [self.method]
        needs_dir = any(m not in ['covariance', 'arbitrary_R1', 'arbitrary_R2'] for m in methods)
        if needs_dir and self.splitDir is None:
            raise ValueError(
                "splitDir is required when split_method is not (only) 'covariance'"
            )
@dataclass
class InitSplitConfig:
    enabled: bool
    multi: Optional[Any] = None

    def __post_init__(self):
        if self.enabled and self.multi is None:
            raise ValueError("init_split_multi is required when init_split is enabled")


@dataclass
class IntermedSplitConfig:
    enabled: bool
    multi: Optional[Any] = None
    time: Optional[Any] = None  # can be a list

    def __post_init__(self):
        if self.enabled and self.multi is None:
            raise ValueError("intermed_split_multi is required when intermed_split is enabled")


@dataclass
class MultiRefConfig:
    enabled: bool
    inds: Optional[Any] = None

    def __post_init__(self):
        if self.enabled and self.inds is None:
            raise ValueError("multiRefInds is required when useMultiRef is enabled")

def loadData(dataPath, basePath, dstt_order, stt_order, dataRunTag, dt, MC_tag, MC_num):
    # Load DSTT data
    DSTTs_nd = np.load(os.path.join(dataPath, f'DSTTs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    Rs_nd = np.load(os.path.join(dataPath, f'Rs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    print("DSTTs loaded successfully!")

    # Load sDSTT data
    sDSTTs_nd = np.load(os.path.join(dataPath, f'sDSTTs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    sRs_nd = np.load(os.path.join(dataPath, f's_Rs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    print("sDSTTs loaded successfully!")

    # Load qDSTT data
    qDSTTs_nd = np.load(os.path.join(dataPath, f'qDSTTs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    qRs_nd = np.load(os.path.join(dataPath, f'q_Rs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    print("qDSTTs loaded successfully!")

    # Load hoDSTT data
    hoDSTTs_nd = np.load(os.path.join(dataPath, f'hoDSTTs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    hoRs_nd = np.load(os.path.join(dataPath, f'ho_Rs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    print("hoDSTTs loaded successfully!")

    # Load pDSTT data
    pDSTTs_nd = np.load(os.path.join(dataPath, f'pDSTTs_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    pRs_nd = np.load(os.path.join(dataPath, f'Rs_p_{dstt_order}_{dataRunTag}_T{dt}.npy'))
    print("pDSTTs loaded successfully!")

    # Load STT data
    Xs_nd = np.load(os.path.join(dataPath, f'Xs_{stt_order}_{dataRunTag}_T{dt}.npy'))
    STTs_nd = np.load(os.path.join(dataPath, f'STTs_{stt_order}_{dataRunTag}_T{dt}.npy'))
    print("STTs loaded successfully!")

    # Load MC Data
    MCpath = os.path.join(basePath, 'data', MC_tag)
    MCData = []
    for i in range(MC_num):
        data = loadmat(os.path.join(MCpath, f'case{i}.mat'))
        Xs = data['X']
        MCData.append(Xs)
    MCData = np.array(MCData)  # Nruns x stateSize x times


    return DSTTs_nd, Rs_nd, Xs_nd, STTs_nd, MCData, sDSTTs_nd, sRs_nd, qDSTTs_nd, qRs_nd, pDSTTs_nd, pRs_nd, hoDSTTs_nd, hoRs_nd


# --- top-level, importable, NOT inside the class ---
def _propagate_worker(payload):
    """
    payload is a plain dict of picklable data (numpy arrays, floats, strings,
    and references to plain module-level functions like convertToNonDim).
    Returns (traj_id, mean_kp1, P_kp1).
    """
    prop_method = payload['prop_method']
    params      = payload['params']       # per-call copy, safe to mutate (gdualFlag etc.)
    to_nd       = payload['to_nd']
    from_nd     = payload['from_nd']
    to_nd_cov   = payload['to_nd_cov']
    from_nd_cov = payload['from_nd_cov']
    mean_k, P_k = payload['mean_k'], payload['P_k']
    stateSize   = payload['stateSize']
    TSExps = []

    if prop_method == "STT":
        order = payload['order']
        if payload['integrateAlongTheWay']:
            Xs_nd_int_kp1, Xs_d_int_kp1, STTs_k = getSTTs_general(
                [[0, payload['dt']]], mean_k, order, params, params['eomTag'], mode='tktkp1')
            X_nomk_nd   = to_nd(mean_k, params)
            X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
            STTs_k_nd   = STTs_k[-1, :]
        else:
            X_nomk_nd   = payload['X_nomk_nd']
            X_nomkp1_nd = payload['X_nomkp1_nd']
            STTs_k_nd   = payload['STTs_k_nd']

        TSExps = STTs_k_nd
        dm_k = -from_nd(X_nomk_nd, params) + mean_k
        dm_k_nd = to_nd(dm_k, params)
        P_k_nd = to_nd_cov(P_k, params)
        expected_list_nd = compute_expected(dm_k_nd, P_k_nd, order, stateSize)
        phi_list_nd = reshape_STT(STTs_k_nd, order, stateSize)
        dm_kp1_nd = compute_expected_dm(phi_list_nd, expected_list_nd, order, stateSize)
        P_kp1_nd  = compute_covariance(phi_list_nd, expected_list_nd, dm_kp1_nd, order, stateSize)
        mean_kp1 = from_nd(X_nomkp1_nd + dm_kp1_nd, params)
        P_kp1    = from_nd_cov(P_kp1_nd, params)
    elif prop_method == "DSTT":
        latentDim   = payload['latentDim']
        order = payload['order']
        R_k_nd   = payload['R_k_nd']
        if payload['integrateAlongTheWay']:
            Xs_nd_int_kp1, Xs_d_int_kp1, DSTTs_k = getDSTTs_general(
                [[0, payload['dt']]], mean_k, [R_k_nd], order, params, params['eomTag'], latentDim, mode='tktkp1')
            X_nomk_nd   = to_nd(mean_k, params)
            X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
            DSTT_k_nd   = DSTTs_k[-1, :]
        else:
            X_nomk_nd   = payload['X_nomk_nd']
            X_nomkp1_nd = payload['X_nomkp1_nd']
            DSTT_k_nd   = payload['DSTTs_k_nd']

        TSExps = DSTT_k_nd
        dm_k = mean_k - from_nd(X_nomk_nd, params)
        dm_k_nd = to_nd(dm_k, params)
        P_k_nd = to_nd_cov(P_k, params)
        expected_list_nd = compute_expected(dm_k_nd, P_k_nd, order, stateSize)
        DSTT_list = reshape_DSTT(DSTT_k_nd, order, stateSize, latentDim=latentDim)
        # sigma_xi_k_nd = np.sqrt(np.einsum('i,ij,j->', R_k_nd, P_k_nd, R_k_nd))
        # (DSTT_list, order, stateSize, R, expected_list)
        dm_kp1_nd = compute_expected_dm_DSTT(DSTT_list, order, stateSize, R_k_nd, expected_list_nd)
        # (DSTT_list, expected_list, dm_kp1, order, R)
        P_kp1_nd = compute_covariance_DSTT(DSTT_list, expected_list_nd, dm_kp1_nd, order, R_k_nd)
        P_kp1 = from_nd_cov(P_kp1_nd, params)
        mean_kp1 = from_nd(X_nomkp1_nd + dm_kp1_nd, params)
    elif prop_method == "xDSTT":
        order = payload['order']
        R_k_nd = payload['R_k_nd']
        latentDim   = payload['latentDim']
        if payload['integrateAlongTheWay']:
            raise NotImplementedError
            # print("Integrating xDSTTs along the way...")
            # Xs_nd_int_kp1, Xs_d_int_kp1, STTs_k = getSTTs_general(
            #     [[0, payload['dt']]], mean_k, order, params, params['eomTag'], mode='tktkp1')
            #
            # print("...Done integrating STTs")
            # DSTTs_k =  STTstoDSTTs([0], STTs_k[-1, :][np.newaxis], R_k_nd[np.newaxis], order, stateSize)
            # # Xs_nd_int_kp1, Xs_d_int_kp1, DSTTs_k = getxDSTTs_general(  TODO
            # #     [[0, payload['dt']]], mean_k, [R_k_nd], order, params, params['eomTag'], latentDim, mode='tktkp1')
            # X_nomk_nd   = to_nd(mean_k, params)
            # X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
            # xDSTT_k_nd   = DSTTs_k[-1, :]
        else:
            X_nomk_nd   = payload['X_nomk_nd']
            X_nomkp1_nd = payload['X_nomkp1_nd']
            xDSTT_k_nd   = payload['xDSTTs_k_nd']

        TSExps = xDSTT_k_nd
        dm_k = mean_k - from_nd(X_nomk_nd, params)
        dm_k_nd = to_nd(dm_k, params)
        P_k_nd = to_nd_cov(P_k, params)
        expected_dict_nd = compute_expected_mixed(dm_k_nd, P_k_nd, order, R_k_nd, stateSize)
        xDSTT_list = reshape_DSTT(xDSTT_k_nd, order, stateSize, latentDim=latentDim)
        if xDSTT_list[1].shape[1] != latentDim:
            raise ValueError(f"xDSTT latent dimension mismatch: expected {latentDim}, got {xDSTT_list[1].shape[1]}")
        dm_kp1_nd = compute_expected_dm_DSTT_larger_ld(xDSTT_list, order, stateSize, expected_dict_nd)
        P_kp1_nd = compute_covariance_DSTT_larger_ld(xDSTT_list, expected_dict_nd, dm_kp1_nd, order)
        P_kp1 = from_nd_cov(P_kp1_nd, params)
        mean_kp1 = from_nd(X_nomkp1_nd + dm_kp1_nd, params)
    elif prop_method == "UT":
        lambda_ = payload['lambda_']
        t = payload['t']
        dt = payload['dt']
        P_k_nd = to_nd_cov(P_k, params)
        Sk_nd = np.linalg.cholesky(P_k_nd)
        n = stateSize
        # Generate 2n + 1 sigma points
        xi_ks_nd = np.zeros((n, 2 * n + 1))
        mean_k_nd = to_nd(mean_k, params)
        xi_ks_nd[:, 0] = mean_k_nd
        for ii in range(n):
            xi_ks_nd[:, ii + 1] = mean_k_nd + np.sqrt(n + lambda_) * Sk_nd[:, ii]
            xi_ks_nd[:, ii + n + 1] = mean_k_nd - np.sqrt(n + lambda_) * Sk_nd[:, ii]
        # Propagate each sigma point through the dynamics function
        xi_kp1s_nd = np.zeros((n, 2 * n + 1))
        for ii in range(2 * n + 1):
            tspan_nd = [t / np.sqrt(
                gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get(
                    'g_surf')), (t + dt) / np.sqrt(
                gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get(
                    'g_surf'))]
            eom_func = params['eomFunc']
            sol = solve_ivp(lambda t, X: eom_func(t, X, params), t_span=tspan_nd, y0=xi_ks_nd[:, ii],
                            method='RK45', rtol=1e-13, atol=1e-18)
            xi_kp1s_nd[:, ii] = sol.y[:, -1]
        # Recombine resultant points to get predicted mean and covariance
        P_kp1_nd = np.zeros((n, n))
        Wm, Wc = payload['getSigmaPointWeights']()
        mean_kp1_nd = np.zeros(n)
        for ii in range(2 * n + 1):
            mean_kp1_nd = mean_kp1_nd + Wm[ii] * xi_kp1s_nd[:, ii]
        for ii in range(2 * n + 1):
            # print(from_nd(xi_kp1s_nd[:, ii] - mean_kp1_nd, params))
            P_kp1_nd += Wc[ii] * np.outer(xi_kp1s_nd[:, ii] - mean_kp1_nd, xi_kp1s_nd[:, ii] - mean_kp1_nd)
        P_kp1_nd = 0.5 * (P_kp1_nd + P_kp1_nd.T)
        mean_kp1 = from_nd(mean_kp1_nd, params)
        P_kp1 = from_nd_cov(P_kp1_nd, params)
    elif prop_method == "Linear":
        raise ValueError("Use STT with order 1")
    else:
        raise ValueError(f'Invalid mean / cov propagation method {prop_method}')

    if payload['procNoiseFlag']:
        P_kp1 = P_kp1 + payload['proc_cov']
    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)

    return payload['traj_id'], mean_kp1, P_kp1, TSExps

class GMM_UQ:
    def __init__(self, ts, params, init_mean, init_cov, num_splits,
                 split_config: SplitConfig, prop_config: PropConfig,
                 proc_noise: ProcNoiseConfig, split_crit: SplitCriterionConfig,
                 init_split_cfg: InitSplitConfig, intermed_split_cfg: IntermedSplitConfig,
                 multi_ref_cfg: MultiRefConfig):
        # Given parameters
        self.ts = ts
        self.params = params
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.num_splits = num_splits
        self.stateSize = len(init_mean)

        # Split Methods
        self.split_method = split_config.method
        self.splitDir = split_config.splitDir

        #  Propagation Methods
        self.prop_method = prop_config.method
        self.integrateAlongTheWay = prop_config.integrateAlongTheWay
        self.propParFlag = prop_config.propParFlag
        self.max_workers = prop_config.max_workers

        # Process noise
        self.procNoiseFlag = proc_noise.enabled
        self.proc_cov = proc_noise.proc_cov

        # Split criterion
        self.split_criterion = split_crit.criterion
        self.DE_threshold = split_crit.DE_threshold
        self.max_mixands = split_crit.max_mixands
        self.SOLC_threshold = split_crit.SOLC_threshold
        self.trackDE = split_crit.trackDE
        self.trackPyest = split_crit.trackPyest
        # You can't track pyest unless the prop method is DSTT or xDSTT
        if self.trackPyest and self.prop_method not in ['DSTT', 'xDSTT']:
            raise ValueError("trackPyest can only be True if prop_method is 'DSTT' or 'xDSTT'")

        # Initial / intermediate splitting
        self.init_split = init_split_cfg.enabled
        self.init_split_multi = init_split_cfg.multi
        self.intermed_split = intermed_split_cfg.enabled
        self.intermed_split_multi = intermed_split_cfg.multi
        self.intermed_split_time = intermed_split_cfg.time
        self.init_split_counter = 0

        # Multi-reference
        self.useMultiRef = multi_ref_cfg.enabled
        self.multiRefInds = multi_ref_cfg.inds

        self.split_directions = {'ts': [], 'dirs': []}

        # --- tensor setup: only relevant for non-UT methods ---
        self.STTs = self.DSTTs = self.xDSTTs = None
        self.Rs = self.xRs = None
        self.stt_order = self.dstt_order = self.xdstt_order = None
        self.Xs_nom = prop_config.Xs_nom

        if self.prop_method == 'STT':
            self.STTs = prop_config.tensors
            self.stt_order = prop_config.order
        elif self.prop_method == 'DSTT':
            self.DSTTs = prop_config.tensors
            self.Rs = prop_config.R
            self.dstt_order = prop_config.order
            self.latentDim = prop_config.latentDim
        elif self.prop_method == 'xDSTT':
            self.xDSTTs = prop_config.tensors
            self.xRs = prop_config.R
            self.xdstt_order = prop_config.order
            self.latentDim = prop_config.latentDim
        # 'UT' needs nothing further

        if self.integrateAlongTheWay and self.prop_method != 'UT':
            self._init_tensor_buffer()

        # Derived parameters
        self.t = ts[0]
        self.dt = ts[1] - ts[0]
        self.split_count = 0
        self.total_time = 0

        # Derived parameters
        self.t = ts[0]
        self.dt = ts[1] - ts[0]
        self.stateSize = len(init_mean)
        self.split_count = 0
        self.total_time = 0

        # UT Specific Parameters
        self.kappa = 0
        self.beta = 2
        self.alpha = 0.1
        self.lambda_ = self.alpha ** 2 * (len(init_mean) + self.kappa) - len(init_mean)

        # Trajs will save everything dimensionally, but all propagation will be done nondimensionally
        self.trajectories = []
        if 'dens' in params['eomTag'].lower():
            H0 = get_nonlin_entropy(self.to_nd_cov(self.init_cov, self.params))
        else:
            H0 = get_nonlin_entropy(convertToNonDimCov(self.init_cov, self.params))
        traj = self.createNewTraj(0, self.t, init_mean, init_cov, H0, H0, 0, 1., H0, 0)
        self.trajectories.append(traj)
        self.traj_id_counter = 1  # start next trajectory with id 1
        self.num_active_trajs = []

        # Select appropriate dimensionalization functions
        if 'dens' in self.params['eomTag'].lower():
            self.to_nd = convertToNonDimDens
            self.from_nd = convertFromNonDimDens
            self.to_nd_cov = convertToNonDimDensCov
            self.from_nd_cov = convertFromNonDimDensCov
        else:
            self.to_nd = convertToNonDim
            self.from_nd = convertFromNonDim
            self.to_nd_cov = convertToNonDimCov
            self.from_nd_cov = convertFromNonDimCov


    def _init_tensor_buffer(self):
        """Allocate the NaN-filled tensor history buffer for the active prop_method."""
        n = self.stateSize

        if self.prop_method == 'STT':
            dim = int(np.sum([n ** (i + 2) for i in range(self.stt_order)]))
            self.STTs = np.full((len(self.ts), dim), np.nan)

        elif self.prop_method == 'DSTT':
            l = 1  # TODO only support single latent dimension for now
            dim = int(n ** 2 + n * l * (self.dstt_order-1))
            self.DSTTs = np.full((len(self.ts), dim), np.nan)

        elif self.prop_method == 'xDSTT':
            l = 1  # TODO only support single latent dimension for now
            dim = int(n ** 2 + n * l * (self.xdstt_order - 1))
            self.xDSTTs = np.full((len(self.ts), dim), np.nan)

    def createNewTraj(self, id, time, mean, cov, lin_entropy, nonlin_entropy, DE, weight, H0, ref_traj_ind):
        traj = {'id': id,
                 'times': [time], # times of just this trajectory
                 'means': [mean],
                 'covs': [cov],
                 'isStale': False, # Change this to true once split
                 'lin_entropy': [lin_entropy], # linear entropy propagated with the ODE
                 'nonlin_entropy': [nonlin_entropy], # nonlinear entropy
                 'DE': [DE], # nonlin - lin entropy / H0
                 'SOLC_R_scores': [0], # Pyest SOLC_R scores
                 'weight': weight, # trajectory weight
                 'H0': H0, # Initial entropy to normalize by
                 'ref_traj_ind': ref_traj_ind, # Index of reference trajectories to use, 0 if using single ref traj, otherwise corresponds to index for the STTs / ref
                }

        if self.prop_method == 'STT':
            traj['STTs'] = []
        elif self.prop_method == 'DSTT':
            traj['DSTTs'] = []
        elif self.prop_method == 'xDSTT':
            traj['xDSTTs'] = []

        return traj

    def integrateLinearEntropy(self, i):
        if self.trackDE:
            tstar = np.sqrt(gravParams[self.params['planetName']].get('R_eq') /
                            gravParams[self.params['planetName']].get('g_surf'))
            ts_nd = [self.t / tstar, self.ts[i + 1] / tstar]
            for traj in self.trajectories:
                if not traj['isStale']:
                    mean_k_nd = self.to_nd(traj['means'][-1], self.params)
                    H_lin_k = traj['lin_entropy'][-1]
                    local_params = copy.deepcopy(self.params)  # isolate gdualFlag mutation
                    sol = solve_ivp(lambda t, H: lin_entropy_ode_general(t, H, [mean_k_nd], local_params),
                                    y0=np.array([H_lin_k]), t_span=ts_nd, method='RK45', rtol=1e-7, atol=1e-8)
                    traj['lin_entropy'].append(sol.y[-1][-1])
        else:
            for traj in self.trajectories:
                if not traj['isStale']:
                    traj['lin_entropy'].append(np.nan)
            # print('...Finished integrating linear entropy (not tracking DE)...')
        return

    def trackSplitCrit(self, t_ind):
        for traj in self.trajectories:
            if not traj['isStale']:
                if self.trackDE:
                    cov = traj['covs'][-1]
                    P_nd = self.to_nd_cov(cov, self.params)
                    E_lin_k = traj['lin_entropy'][-1]
                    DE_k, H_nonlin_k = get_instantaneous_DE(P_nd, E_lin_k, traj['H0'])
                    traj['DE'].append(DE_k)
                    traj['nonlin_entropy'].append(H_nonlin_k)

                    print(f"Tracking DE for traj {traj['id']} at time {self.t}: {DE_k}")
                else:
                    traj['DE'].append(np.nan)
                    traj['nonlin_entropy'].append(np.nan)

                # Track other criterion
                if self.trackPyest:
                    cov = traj['covs'][-1]
                    P_nd = self.to_nd_cov(cov, self.params)
                    R_i = np.squeeze(self.Rs[t_ind, 0, :]) if self.prop_method == 'DSTT' else np.squeeze(self.xRs[t_ind, 0, :])
                    if not self.integrateAlongTheWay:
                        thisDSTT = self.DSTTs[t_ind + 1, :] if self.prop_method =='DSTT' else self.xDSTTs[t_ind + 1, :]
                    else:
                        thisDSTT = traj['DSTTs'][-1] if self.prop_method == 'DSTT' else traj['xDSTTs'][-1]
                        thisDSTT = traj['DSTTs'][-1] if self.prop_method == 'DSTT' else traj['xDSTTs'][-1]
                    order = self.dstt_order if self.prop_method == 'DSTT' else self.xdstt_order
                    DSTT_list = reshape_DSTT(thisDSTT, order, self.stateSize)
                    score = solc_R_score(DSTT_list, R_i, P_nd, whiten=True)
                    print(f"Tracking SOLC_R score for traj {traj['id']} at time {self.t}: {score}")
                    traj['SOLC_R_scores'].append(score)
            # print('...Finished tracking split criterion metrics...')
        return

    def checkSplitCriteria(self, time_ind):
        ids_to_split = []
        solc_scores = {}
        for traj in self.trajectories:
            # If t = 0 and an initial split is desired, split all trajectories
            if self.init_split and self.t == 0 and traj['isStale'] is False:
                ids_to_split.append(traj['id'])
                continue
            # Get current mean and covariance for this trajectory if it is an active trajectory
            if not traj['isStale']:
                mean = traj['means'][-1]
                cov = traj['covs'][-1]
                # Differential entropy method
                if self.split_criterion == 'DE':
                    H_lin_k = traj['lin_entropy'][-1]
                    # Check if we need to split using differential entropy
                    P_nd = self.to_nd_cov(cov, self.params)
                    DE_k, _ = get_instantaneous_DE(P_nd, H_lin_k, traj['H0'])
                    if DE_k >= self.DE_threshold:
                        ids_to_split.append(traj['id'])
                        print(f"Adding traj {traj['id']} to split at time {self.t} with DE {DE_k}")
                elif self.split_criterion == "manual":
                    # H_lin_k = traj['lin_entropy'][-1]
                    # P_nd = self.to_nd_cov(cov, self.params)
                    # DE_k, _ = get_instantaneous_DE(P_nd, H_lin_k, traj['H0'])
                    timeCheck = []
                    if isinstance(self.intermed_split_time, list):
                        for split_time in self.intermed_split_time:
                            timeCheck.append(np.abs(self.t - split_time) <= 1e-14)
                    else:
                        timeCheck.append(np.abs(self.t - self.intermed_split_time) <= 1e-14)

                    if any(timeCheck):  #TODO hardcoded to test
                        if self.t > 0:
                            if traj['id'] % 1 == 0:
                                ids_to_split.append(traj['id'])
                                print(f"Adding traj {traj['id']} to split at time {self.t}")  # with DE {DE_k}")
                        else:
                            ids_to_split.append(traj['id'])
                            print(f"Adding traj {traj['id']} to split at time {self.t}")
                elif self.split_criterion == 'SOLC_R':
                    R_i = np.squeeze(self.Rs[time_ind, 0, :]) if self.prop_method == 'DSTT' else np.squeeze(self.xRs[time_ind, 0, :])
                    if not self.integrateAlongTheWay:
                        thisDSTT = self.DSTTs[time_ind + 1, :] if self.prop_method == 'DSTT' else self.xDSTTs[time_ind + 1, :]
                    else:
                        thisDSTT = traj['DSTTs'][-1] if self.prop_method == 'DSTT' else traj['xDSTTs'][-1]
                    order = self.dstt_order if self.prop_method == 'DSTT' else self.xdstt_order
                    DSTT_list = reshape_DSTT(thisDSTT,order, self.stateSize)
                    P_nd = self.to_nd_cov(cov, self.params)
                    print(f"R shape {R_i.shape}, P shape {P_nd.shape}, DSTT shape {[d.shape for d in DSTT_list]}")
                    score = solc_R_score(DSTT_list, R_i, P_nd, whiten=True)
                    solc_scores[traj['id']] = score
                    if score >= self.SOLC_threshold:
                        ids_to_split.append(traj['id'])
                elif self.split_criterion == 'none':
                    pass
                else:
                    raise ValueError('Invalid split criterion')

        if self.max_mixands is not None:
            ids_to_split = self.enforceMaxMixands(ids_to_split, solc_scores if self.split_criterion == 'SOLC_R' else {}, self.num_splits)

        return ids_to_split

    def enforceMaxMixands(self, ids_to_split, metric_by_id, num_new_per_split):
        """Truncate split candidates to respect self.max_mixands.
        metric_by_id: {traj_id: score}, higher score = split first when budget-limited.
        num_new_per_split: mixands created per split (== self.num_splits)."""
        active_count = sum(1 for t in self.trajectories if not t['isStale'])
        budget = self.max_mixands - active_count
        if budget <= 0:
            return []
        net_new_per_split = num_new_per_split - 1  # 1 stale traj replaced by num_splits new
        max_splits_allowed = max(0, budget // net_new_per_split)
        if max_splits_allowed >= len(ids_to_split):
            return ids_to_split
        ids_sorted = sorted(ids_to_split, key=lambda i: metric_by_id.get(i, 0.0), reverse=True)
        return ids_sorted[:max_splits_allowed]

    def splitTrajectories(self, ids_to_split, time_ind, splitNum=0):
        for traj in self.trajectories:
            # If the current trajectory is on the list, split it using the desired method into the desired number of mixands
            if traj['id'] in ids_to_split:
                # Get current mean and covariance for this trajectory
                mean = traj['means'][-1]
                cov = traj['covs'][-1]
                parent_weight = traj['weight']
                # Split the trajectory
                current_split_method = self.split_method if not isinstance(self.split_method, list) else self.split_method[splitNum]

                if current_split_method == 'covariance':
                    # Split along covariance direction, nondimensionally
                    weights, means_nd, covs_nd, split_dir = splitMixandsCovariance(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params))
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'dstt':
                    # Split along DSTT direction, nondimensionally
                    weights, means_nd, covs_nd, split_dir = splitMixandsDSTT(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), self.Rs[time_ind])
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'dsttCGT':
                    # TODO THIS DOES NOT WORK
                    # Split along DSTT direction, nondimensionally, using eigenvalues and vectors of the CGT until the end of the trajectory
                    weights, means_nd, covs_nd = splitMixandsDSTTCGT(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), self.Rs[time_ind], self.DSTTs, time_ind, self.params, self.ts)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'manualDSTT':
                    # Split along hardcoded direction, with DSTT method
                    # X is [r, theta, phi, V, gamma, psi, rho]
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd, split_dir = splitMixandsDSTT(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'manualCov':
                    # Split along hardcoded direction, with DSTT method
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd, split_dir = splitMixandsCovarianceDir(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'arbitrary':
                    # Split along hardcoded direction with Vittaldev's method
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'arbitrary_R':
                    # Split along hardcoded direction with Vittaldev's method
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]

                elif current_split_method == 'arbitrary_R1':
                    # Split along hardcoded direction with Vittaldev's method for a 2 latent dimension, choose the first direction
                    split_dir = np.squeeze(self.xRs[time_ind,0,:] if self.prop_method == 'xDSTT' else self.Rs[time_ind,0,:])
                    print('Splitting mixands along DSTT R direction...')
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'arbitrary_R2':
                    # Split along hardcoded direction with Vittaldev's method for a 2 latent dimension, choose the second direction
                    split_dir = np.squeeze(self.xRs[time_ind,1,:] if self.prop_method == 'xDSTT' else self.Rs[time_ind,1,:])
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'arbitrary_R_STT':
                    # Split along hardcoded direction with Vittaldev's method for a 2 latent dimension, when we have to pass in R for the STTs
                    split_dir = np.squeeze(self.splitDir[0]) if time_ind <= 15 else np.squeeze(self.splitDir[1]) # TODO hardcode in for now
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, self.to_nd_cov(cov, self.params), self.to_nd(mean, self.params), split_dir)
                    means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                elif current_split_method == 'cov_arb':
                    # Split along hardcoded direction with Vittaldev's method
                    if time_ind < 100: # TODO hardcode in for now
                        weights, means_nd, covs_nd, split_dir = splitMixandsCovariance(self.num_splits,
                                                                                       self.to_nd_cov(cov,
                                                                                                              self.params),
                                                                                       self.to_nd(mean,
                                                                                                           self.params))
                        means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                        covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                    else:
                        split_dir = self.splitDir
                        weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits,
                                                                                   self.to_nd_cov(cov,
                                                                                                          self.params),
                                                                                   self.to_nd(mean,
                                                                                                       self.params),
                                                                                   split_dir)
                        means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                        covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]

                elif current_split_method == 'arb_cov':
                    # Split along hardcoded direction with Vittaldev's method
                    if time_ind > 100: # TODO hardcode in for now
                        weights, means_nd, covs_nd, split_dir = splitMixandsCovariance(self.num_splits,
                                                                                       self.to_nd_cov(cov,
                                                                                                              self.params),
                                                                                       self.to_nd(mean,
                                                                                                           self.params))
                        means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                        covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                    else:
                        split_dir = self.splitDir
                        weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits,
                                                                                   self.to_nd_cov(cov,
                                                                                                          self.params),
                                                                                   self.to_nd(mean,
                                                                                                       self.params),
                                                                                   split_dir)
                        means = [self.from_nd(mean_nd, self.params) for mean_nd in means_nd]
                        covs = [self.from_nd_cov(cov_nd, self.params) for cov_nd in covs_nd]
                else:
                    raise ValueError('Invalid split method')
                # Create new trajectories for each mixand
                for ii, mean_i in enumerate(means):
                    H0 = get_nonlin_entropy(self.to_nd_cov(covs[ii], self.params))

                    # Check if the new covariance is PSD
                    if np.any(np.abs(np.linalg.eigvals(covs[ii] - 1e-16)) <= 0):
                        # Get exactly which diagonal elements are negative
                        diags = np.diag(covs[ii])
                        neg_diag_inds = np.where(diags <= 0)
                        print(f"Negative covariance at time {self.t + self.dt} for traj {self.traj_id_counter} in split {ii}. Diagonal elements: {diags}, negative indices: {neg_diag_inds}")
                        print(np.linalg.eigvals(covs[ii]))
                        blah = 42

                    # If this is the first time, update which reference each trajectory corresponds to
                    # All child splits use reference from parent
                    if self.useMultiRef:
                        if time_ind == 0:
                            traj_new = self.createNewTraj(self.traj_id_counter, self.t, mean_i, covs[ii], H0, H0, 0, parent_weight*weights[ii], H0, self.multiRefInds[ii])
                        else:  # Later split - inherit reference trajectory from parent
                            traj_new = self.createNewTraj(self.traj_id_counter, self.t, mean_i, covs[ii], H0, H0, 0, parent_weight*weights[ii], H0, traj['ref_traj_ind'])
                    else:
                        traj_new = self.createNewTraj(self.traj_id_counter, self.t, mean_i, covs[ii], H0, H0, 0,
                                                          parent_weight * weights[ii], H0, 0)
                    self.trajectories.append(traj_new)
                    # Print Update
                    update = f"New traj {self.traj_id_counter} created"
                    if self.trackDE:
                        update.join(f" with DE {traj_new['DE'][-1]} and weight {traj_new['weight']}")
                    if self.useMultiRef:
                        update.join(f" using reference {traj_new['ref_traj_ind']}")
                    self.traj_id_counter += 1


                # Set the current trajectory as stale
                traj['isStale'] = True
                self.split_count += 1
                self.split_directions['ts'].append(self.t)
                self.split_directions['dirs'].append(split_dir)

                # self.plotSplitMixands(cov, mean, 0, 3, covs, means, weights, self.t)
                # plt.savefig(f'/Users/gracecalkins/Local Documents/local_code/sttgmm/figures/AAS_paper/split_{self.t}_{self.split_method}.eps', format='eps')
                # plt.savefig(
                #     f'/Users/gracecalkins/Local Documents/local_code/sttgmm/figures/AAS_paper/split_{self.t}_{self.split_method}.png',
                #     dpi=200)
                # plt.close()

        # Check that the sum of all non-stale weights is 1
        total_weight = sum([traj['weight'] for traj in self.trajectories if traj['isStale'] is False])
        if (1-total_weight) > 1e-8:
            print('Weights do not sum to 1')
            blah = 42

        # Update number of active trajectories
        active_traj = 0
        for traj in self.trajectories:
            if not traj['isStale'] and self.t in traj['times']:
                active_traj += 1
        self.num_active_trajs.append(active_traj)

        return

    def getSigmaPointWeights(self):
        n = len(self.init_mean)
        Wm = np.zeros(2*n + 1)
        Wc = np.zeros(2*n + 1)
        for i in range(2*n + 1):
            if i == 0:
                Wm[i] = self.lambda_ / (n + self.lambda_)
                Wc[i] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
            else:
                Wm[i] = 1 / (2*(n + self.lambda_))
                Wc[i] = 1 / (2*(n + self.lambda_))

        return Wm, Wc

    # def _propagate_one(self, traj, i):
    #     """Pure-ish worker: computes mean_kp1, P_kp1 for a single trajectory.
    #     Reads self.* (read-only) and traj (read-only), returns results —
    #     does NOT append to traj['means']/['covs']/['times']."""
    #     local_params = copy.deepcopy(self.params) if self.integrateAlongTheWay else self.params
    #
    #     if not self.integrateAlongTheWay and not self.prop_method == "UT":
    #         if not self.useMultiRef:
    #             X_nomk_nd = self.Xs_nom[i, :]
    #             X_nomkp1_nd = self.Xs_nom[i + 1, :]
    #         else:
    #             X_nomk_nd = self.Xs_nom[traj['ref_traj_ind']][i, :]
    #             X_nomkp1_nd = self.Xs_nom[traj['ref_traj_ind']][i + 1, :]
    #
    #     mean_k = traj['means'][-1]
    #     P_k = traj['covs'][-1]
    #
    #     if self.prop_method == "DSTT":
    #         if self.useMultiRef:
    #             DSTT_k_nd = traj['ref_traj_ind'][i + 1, :]
    #             R_k_nd = traj['ref_traj_ind'][i + 1, :]
    #         else:
    #             R_k_nd = self.Rs[i, :]
    #             if not self.integrateAlongTheWay:
    #                 DSTT_k_nd = self.DSTTs[i, :]
    #             else:
    #                 Xs_nd_int_kp1, Xs_d_int_kp1, DSTTs_k = getDSTTs_general([[0, self.dt]], mean_k, [R_k_nd],
    #                                                                         self.dstt_order, local_params,
    #                                                                         local_params['eomTag'], 'tktkp1')
    #                 X_nomk_nd = self.to_nd(mean_k, local_params)
    #                 X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
    #                 DSTT_k_nd = DSTTs_k[-1, :]
    #                 traj['DSTTs'].append(DSTT_k_nd)  # store the DSTTs for this trajectory if integrating along the way
    #
    #         dm_k = mean_k - self.from_nd(X_nomk_nd, local_params)
    #         dm_k_nd = self.to_nd(dm_k, local_params)
    #         P_k_nd = self.to_nd_cov(P_k, local_params)
    #         expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.dstt_order, self.stateSize)
    #         DSTT_list = reshape_DSTT(DSTT_k_nd, self.dstt_order, self.stateSize)
    #         # sigma_xi_k_nd = np.sqrt(np.einsum('i,ij,j->', R_k_nd, P_k_nd, R_k_nd))
    #         # (DSTT_list, order, stateSize, R, expected_list)
    #         dm_kp1_nd = compute_expected_dm_DSTT(DSTT_list, self.dstt_order, self.stateSize, R_k_nd, expected_list_nd)
    #         # (DSTT_list, expected_list, dm_kp1, order, R)
    #         P_kp1_nd = compute_covariance_DSTT(DSTT_list, expected_list_nd, dm_kp1_nd, self.dstt_order, R_k_nd)
    #         P_kp1 = self.from_nd_cov(P_kp1_nd, local_params)
    #         mean_kp1 = self.from_nd(X_nomkp1_nd + dm_kp1_nd, local_params)
    #         if self.procNoiseFlag:
    #             P_kp1 += self.proc_cov
    #         P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
    #         test = np.diag(P_kp1)
    #         return mean_kp1, P_kp1, traj
    #
    #     elif self.prop_method == "xDSTT":
    #         if self.useMultiRef:
    #             xDSTT_k_nd = traj['ref_traj_ind'][i + 1, :]
    #             Rs_k_nd = traj['ref_traj_ind'][i + 1, :, :]
    #         else:
    #             Rs_k_nd = self.xRs[i, :, :]
    #             if not self.integrateAlongTheWay:
    #                 xDSTT_k_nd = self.xDSTTs[i, :]
    #             else:
    #                 Xs_nd_int_kp1, Xs_d_int_kp1, xDSTTs_k = getDSTTs_general([[0, self.dt]], mean_k, [Rs_k_nd],
    #                                                                          self.xdstt_order, local_params,
    #                                                                          local_params['eomTag'], 'tktkp1')
    #                 X_nomk_nd = self.to_nd(mean_k, local_params)
    #                 X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
    #                 xDSTT_k_nd = xDSTTs_k[-1, :]
    #                 traj['xDSTTs'].append(xDSTT_k_nd)  # store the DSTTs for this trajectory if integrating along the way
    #
    #         dm_k = mean_k - self.from_nd(X_nomk_nd, local_params)
    #         dm_k_nd = self.to_nd(dm_k, local_params)
    #         P_k_nd = self.to_nd_cov(P_k, local_params)
    #         expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.xdstt_order, self.stateSize)
    #         xDSTT_list = reshape_DSTT(xDSTT_k_nd, self.xdstt_order, self.stateSize)
    #         dm_kp1_nd = compute_expected_dm_xDSTT(xDSTT_list, self.xdstt_order, self.stateSize, Rs_k_nd,
    #                                               expected_list_nd)
    #         P_kp1_nd = compute_covariance_xDSTT(xDSTT_list, expected_list_nd, dm_kp1_nd, self.xdstt_order, Rs_k_nd)
    #         P_kp1 = self.from_nd_cov(P_kp1_nd, local_params)
    #         mean_kp1 = self.from_nd(X_nomkp1_nd + dm_kp1_nd, local_params)
    #         if self.procNoiseFlag:
    #             P_kp1 += self.proc_cov
    #         P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
    #         test = np.diag(P_kp1)
    #         return mean_kp1, P_kp1, traj
    #
    #     elif self.prop_method == "STT":
    #         if self.useMultiRef:
    #             STTs_k_nd = traj['ref_traj_ind'][i + 1, :]
    #         else:
    #             if not self.integrateAlongTheWay:
    #                 STTs_k_nd = self.STTs[i + 1, :]
    #             else:
    #                 Xs_nd_int_kp1, Xs_d_int_kp1, STTs_k = getSTTs_general([[0, self.dt]], mean_k, self.stt_order,
    #                                                                       local_params, local_params['eomTag'],
    #                                                                       mode='tktkp1')
    #                 X_nomk_nd = self.to_nd(mean_k, local_params)
    #                 X_nomkp1_nd = Xs_nd_int_kp1[-1, :]
    #                 STTs_k_nd = STTs_k[-1, :]
    #
    #         dm_k = -self.from_nd(X_nomk_nd, local_params) + mean_k
    #         dm_k_nd = self.to_nd(dm_k, local_params)
    #         P_k_nd = self.to_nd_cov(P_k, local_params)
    #         expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.stt_order, self.stateSize)
    #         phi_list_nd = reshape_STT(STTs_k_nd, self.stt_order, self.stateSize)
    #         dm_kp1_nd = compute_expected_dm(phi_list_nd, expected_list_nd, self.stt_order, self.stateSize)
    #         P_kp1_nd = compute_covariance(phi_list_nd, expected_list_nd, dm_kp1_nd, self.stt_order, self.stateSize)
    #         mean_kp1 = self.from_nd(X_nomkp1_nd + dm_kp1_nd, local_params)
    #         P_kp1 = self.from_nd_cov(P_kp1_nd, local_params)
    #         if self.procNoiseFlag:
    #             P_kp1 += self.proc_cov
    #         P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
    #         return mean_kp1, P_kp1, traj
    #
    #     elif self.prop_method == "UT":
    #         P_k_nd = self.to_nd_cov(P_k, local_params)
    #         Sk_nd = np.linalg.cholesky(P_k_nd)
    #         n = self.stateSize
    #         # Generate 2n + 1 sigma points
    #         xi_ks_nd = np.zeros((n, 2 * n + 1))
    #         mean_k_nd = self.to_nd(mean_k, local_params)
    #         xi_ks_nd[:, 0] = mean_k_nd
    #         for ii in range(n):
    #             xi_ks_nd[:, ii + 1] = mean_k_nd + np.sqrt(n + self.lambda_) * Sk_nd[:, ii]
    #             xi_ks_nd[:, ii + n + 1] = mean_k_nd - np.sqrt(n + self.lambda_) * Sk_nd[:, ii]
    #
    #         # Propagate each sigma point through the dynamics function
    #         xi_kp1s_nd = np.zeros((n, 2 * n + 1))
    #         for ii in range(2 * n + 1):
    #             tspan_nd = [self.t / np.sqrt(
    #                 gravParams[local_params['planetName']].get('R_eq') / gravParams[local_params['planetName']].get(
    #                     'g_surf')), self.ts[i + 1] / np.sqrt(
    #                 gravParams[local_params['planetName']].get('R_eq') / gravParams[local_params['planetName']].get(
    #                     'g_surf'))]
    #
    #             eom_func = local_params['eomFunc']
    #             sol = solve_ivp(lambda t, X: eom_func(t, X, local_params), t_span=tspan_nd, y0=xi_ks_nd[:, ii],
    #                             method='RK45', rtol=1e-13, atol=1e-18)
    #             xi_kp1s_nd[:, ii] = sol.y[:, -1]
    #         # Recombine resultant points to get predicted mean and covariance
    #         P_kp1_nd = np.zeros((n, n))
    #         Wm, Wc = self.getSigmaPointWeights()
    #         mean_kp1_nd = np.zeros(n)
    #         for ii in range(2 * n + 1):
    #             mean_kp1_nd = mean_kp1_nd + Wm[ii] * xi_kp1s_nd[:, ii]
    #         for ii in range(2 * n + 1):
    #             # print(self.from_nd(xi_kp1s_nd[:, ii] - mean_kp1_nd, local_params))
    #             P_kp1_nd += Wc[ii] * np.outer(xi_kp1s_nd[:, ii] - mean_kp1_nd, xi_kp1s_nd[:, ii] - mean_kp1_nd)
    #         P_kp1_nd = 0.5 * (P_kp1_nd + P_kp1_nd.T)
    #         mean_kp1 = self.from_nd(mean_kp1_nd, local_params)
    #         P_kp1 = self.from_nd_cov(P_kp1_nd, local_params)
    #         if self.procNoiseFlag:
    #             P_kp1 += self.proc_cov  # MAKE SURE TO ONLY ADD PROCESS NOISE ONCE DIMENSIONALIZED
    #         P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
    #         return mean_kp1, P_kp1, traj
    #
    #     elif self.prop_method == "Linear":
    #         if self.useMultiRef:
    #             STTs_k_nd = traj['ref_traj_ind'][i + 1, :]
    #         else:
    #             if not self.integrateAlongTheWay:
    #                 STTs_k_nd = self.STTs[i + 1, :]
    #             else:
    #                 Xs_nd_int_kp1, Xs_d_int_kp1, STTs_k = getSTTs_general([[0, self.dt]], mean_k, 1, local_params,
    #                                                                       local_params['eomTag'], mode='tktkp1')
    #                 X_nomk_nd = self.to_nd(mean_k, local_params)
    #                 X_nomkp1_nd = Xs_nd_int_kp1[-1, :]  # TODO
    #                 STTs_k_nd = STTs_k[-1, :]
    #
    #         dm_k = self.from_nd(X_nomk_nd, local_params) - mean_k
    #         dm_k_nd = self.to_nd(dm_k, local_params)
    #         P_k_nd = self.to_nd_cov(P_k, local_params)
    #         expected_list_nd = compute_expected(dm_k_nd, P_k_nd, 1, self.stateSize)
    #         phi_list_nd = reshape_STT(STTs_k_nd[0:self.stateSize ** 2], 1, self.stateSize)
    #         dm_kp1_nd = compute_expected_dm(phi_list_nd, expected_list_nd, 1, self.stateSize)
    #         P_kp1_nd = compute_covariance(phi_list_nd, expected_list_nd, dm_kp1_nd, 1, self.stateSize)
    #         mean_kp1 = self.from_nd(X_nomkp1_nd + dm_kp1_nd, local_params)
    #         P_kp1 = self.from_nd_cov(P_kp1_nd, local_params)
    #         if self.procNoiseFlag:
    #             P_kp1 += self.proc_cov
    #         P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
    #
    #         return mean_kp1, P_kp1, traj
    #     else:
    #         raise ValueError('Invalid mean / cov propagation method')
    #     # etc. for xDSTT / UT / Linear

    def _build_payload(self, traj, i):
        payload = {
            'traj_id': traj['id'],
            'prop_method': self.prop_method,
            'mean_k': traj['means'][-1],
            'P_k': traj['covs'][-1],
            'stateSize': self.stateSize,
            'dt': self.dt,
            'integrateAlongTheWay': self.integrateAlongTheWay,
            'useMultiRef': self.useMultiRef,
            'procNoiseFlag': self.procNoiseFlag,
            'proc_cov': self.proc_cov,
            # per-call copy so gdualFlag mutation stays isolated to this process
            'params': copy.deepcopy(self.params),
            # plain module functions -- picklable by reference, not closures
            'to_nd': self.to_nd,
            'from_nd': self.from_nd,
            'to_nd_cov': self.to_nd_cov,
            'from_nd_cov': self.from_nd_cov,
        }
        if self.prop_method == "STT":
            payload['order'] = self.stt_order
            if not self.integrateAlongTheWay:
                ref = traj['ref_traj_ind'] if self.useMultiRef else None
                payload['STTs_k_nd'] = (self.STTs[ref][i + 1, :] if self.useMultiRef
                                        else self.STTs[i + 1, :])
                payload['X_nomk_nd'] = self.Xs_nom[i, :] if not self.useMultiRef else self.Xs_nom[ref][i, :]
                payload['X_nomkp1_nd'] = self.Xs_nom[i + 1, :] if not self.useMultiRef else self.Xs_nom[ref][i + 1, :]
        if self.prop_method == 'DSTT':
            payload['order'] = self.dstt_order
            payload['R_k_nd'] = traj['ref_traj_ind'][i + 1, :] if self.useMultiRef else self.Rs[i, :]
            payload['latentDim'] = self.latentDim
            if not self.integrateAlongTheWay:
                ref = traj['ref_traj_ind'] if self.useMultiRef else None
                payload['DSTTs_k_nd'] = (self.DSTTs[ref][i + 1, :] if self.useMultiRef
                                        else self.DSTTs[i + 1, :])
                payload['X_nomk_nd'] = self.Xs_nom[i, :] if not self.useMultiRef else self.Xs_nom[ref][i, :]
                payload['X_nomkp1_nd'] = self.Xs_nom[i + 1, :] if not self.useMultiRef else self.Xs_nom[ref][i + 1, :]
        if self.prop_method == 'xDSTT':
            payload['order'] = self.xdstt_order
            payload['R_k_nd'] = traj['ref_traj_ind'][i + 1, :] if self.useMultiRef else self.xRs[i, :]
            payload['latentDim'] = self.latentDim
            if not self.integrateAlongTheWay:
                ref = traj['ref_traj_ind'] if self.useMultiRef else None
                payload['xDSTTs_k_nd'] = (self.xDSTTs[ref][i + 1, :] if self.useMultiRef
                                        else self.xDSTTs[i + 1, :])
                payload['X_nomk_nd'] = self.Xs_nom[i, :] if not self.useMultiRef else self.Xs_nom[ref][i, :]
                payload['X_nomkp1_nd'] = self.Xs_nom[i + 1, :] if  not self.useMultiRef else self.Xs_nom[ref][i + 1, :]
        if self.prop_method == "UT":
            payload['lambda_'] = self.lambda_
            payload['getSigmaPointWeights'] = self.getSigmaPointWeights
            payload['t'] = self.t
        return payload

    def predictionStep(self, i, executor=None):
        active_trajs = [t for t in self.trajectories if not t['isStale']]
        payloads = [self._build_payload(traj, i) for traj in active_trajs]

        if executor is not None and len(active_trajs) > 1:
            results = list(executor.map(_propagate_worker, payloads))
        else:
            results = [_propagate_worker(p) for p in payloads]

        # print('...Finished propagation loop')

        results_by_id = {tid: (m, P, TSExp) for tid, m, P, TSExp in results}
        for traj in active_trajs:
            mean_kp1, P_kp1, TSExp = results_by_id[traj['id']]
            if self.prop_method == "STT":
                traj['STTs'].append(TSExp)
            elif self.prop_method == "DSTT":
                traj['DSTTs'].append(TSExp)
            elif self.prop_method == "xDSTT":
                traj['xDSTTs'].append(TSExp)
            if np.any(np.abs(np.linalg.eigvals(P_kp1) + 1e-16) <= 0):
                # Get exactly which diagonal elements are negative
                diags = np.diag(P_kp1)
                neg_diag_inds = np.where(diags <= 0)
                print(f"Traj {traj['id']}: Negative covariance at time {self.t + self.dt}. Diagonal elements: {diags}, negative indices: {neg_diag_inds}")
            traj['means'].append(mean_kp1)
            traj['covs'].append(P_kp1)
            traj['times'].append(self.ts[i + 1])

    def _run_loop(self, executor):
        if self.prop_method == 'STT':
            printMethod = f'{self.prop_method}, order {self.stt_order}'
        elif self.prop_method == 'xDSTT':
            printMethod = f'{self.prop_method}, order {self.xdstt_order}'
        elif self.prop_method == 'DSTT':
            printMethod = f'{self.prop_method}, order {self.dstt_order}'
        else:
            printMethod = f'{self.prop_method}'
        intFlag = ", int along-the-way" if self.integrateAlongTheWay else ""
        print(
            f'Starting GMM UQ with {printMethod}{intFlag} splitting via {self.split_method} into {self.num_splits} mixands...')
        ts_len = len(self.ts)
        step_times = []
        per_mixand_times = []  # normalized cost: seconds per mixand per step
        for i in range(ts_len - 2):
            tic = time.perf_counter()
            # Check if we need to split
            if i == 0:
                if self.init_split:
                    for split in range(self.init_split_multi):
                        ids_to_split = self.checkSplitCriteria(i)
                        # Perform splitting if necessary before propagation
                        self.splitTrajectories(ids_to_split, i, splitNum=split)
            elif self.intermed_split:
                for split in range(self.intermed_split_multi):  # TODO does this work as god intended
                    ids_to_split = self.checkSplitCriteria(i)
                    # Perform splitting if necessary before propagation
                    self.splitTrajectories(ids_to_split, i, splitNum=split)

            # Number of mixands THIS step will actually propagate (post-split, pre-prediction)
            num_active_this_step = sum(1 for traj in self.trajectories if not traj['isStale'])

            # Prediction step
            self.predictionStep(i, executor)
            # Integrate linear entropy
            self.integrateLinearEntropy(i)
            self.trackSplitCrit(i)
            self.t = self.t + self.dt
            toc = time.perf_counter()
            step_time = toc - tic
            step_times.append(step_time)
            if num_active_this_step > 0:
                per_mixand_times.append(step_time / num_active_this_step)
            now = datetime.now()
            expected_future_mixand_counts = self._estimate_future_mixand_counts(i, ts_len, num_active_this_step)

            avg_per_mixand_time = np.mean(per_mixand_times) if per_mixand_times else 0.0
            est_remaining_sec = avg_per_mixand_time * sum(expected_future_mixand_counts)

            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(f"{dt_string} | Sim Time {self.t} / {self.ts[-1]} | Step Time {toc - tic:0.4f} s | Est Remaining Time {(np.mean(step_times) * (ts_len - i - 1)) / 60:0.4f} min")
        total_time = sum(step_times)
        self.total_time = total_time
        print(
            f'Finished GMM UQ with {printMethod} splitting via {self.split_method} into {self.num_splits} mixands with DE treshold {self.DE_threshold} in {total_time / 60:0.4f} min')

    def _estimate_future_mixand_counts(self, i, ts_len, current_count):
        """
        Returns a list of expected active-mixand counts for each remaining step
        (i+1 .. ts_len-2), used to scale the per-mixand time estimate.

        If split_criterion == 'manual' and intermed_split_time is known in advance,
        project splits exactly. Otherwise, assume current_count holds steady
        """
        remaining_steps = ts_len - 2 - i
        if remaining_steps <= 0:
            return []

        counts = [current_count] * remaining_steps

        if self.intermed_split and self.split_criterion == 'manual' and self.intermed_split_time is not None:
            split_times = (self.intermed_split_time if isinstance(self.intermed_split_time, list)
                           else [self.intermed_split_time])
            multiplier = self.num_splits  # each split turns 1 mixand into num_splits
            running_count = current_count
            for step_offset in range(remaining_steps):
                t_step = self.ts[i + 1 + step_offset]
                # crude: if a scheduled split time falls at/near this step, bump the count
                for split_time in split_times:
                    if np.abs(t_step - split_time) <= 1e-14:
                        running_count = running_count * multiplier * self.intermed_split_multi \
                            if self.intermed_split_multi else running_count * multiplier
                counts[step_offset] = running_count

        return counts

    def run(self):
            if self.propParFlag:
                with ProcessPoolExecutor(max_workers=self.max_workers or os.cpu_count()) as executor:
                    self._run_loop(executor)
            else:
                self._run_loop(executor=None)

    def saveSelf(self, dataPath, runTag):
        DE_str = str(self.DE_threshold).replace('.', '_')
        np.save(os.path.join(dataPath, f'gmm_uq_{self.prop_method}_{self.split_method}_{self.split_criterion}_{self.num_splits}_DE{DE_str}_IS{self.init_split}_{runTag}.npy'), self)

    def printPerformanceMetrics(self):
        print(f'GMM UQ with {self.prop_method} splitting via {self.split_method} into {self.num_splits} mixands')
        print(f"Number of Splits: {self.num_splits}")
        num_active_trajs = 0
        for traj in self.trajectories:
            if not traj['isStale']:
                num_active_trajs += 1
        print(f"Final Active Number of Trajectories: {num_active_trajs}")

    def plotMixandsOverTime(self, covOn=False):
        custom_palette = sns.color_palette("flare", as_cmap=True)
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]', r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]', r'Heading Angle, $\psi$ [$^\circ$]']
        if 'dens' in self.params['eomTag'].lower():
            titles.append(r'Density, $\rho$ [kg/m$^3$]')
            fig, axs = plt.subplots(2, 4, figsize=(15, 10))
        else:  # KLE
            wsNames =  [f"w{i}" for i in range(self.params['d_KLE'])]
            titles = titles + wsNames
            nRows = int(np.ceil(len(titles)/4))
            fig, axs = plt.subplots(nRows, 4, figsize=(15, 10))

        axs = np.atleast_1d(axs).ravel()

        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixands')):
            # color = f"C{next(colors)}"
            # color = custom_palette[ii]
            this_weight = traj['weight']
            color = custom_palette(this_weight)
            this_ts = traj['times']
            for ii in range(self.stateSize):
                this_mean = np.array(traj['means'])[:,ii]
                this_stds = np.sqrt(np.array(traj['covs'])[:,ii,ii])
                if traj['id'] == 0:
                    this_label = f'3$\sigma$'
                    this_mean_label = 'Mean'
                else:
                    this_label = None
                    this_mean_label = None
                if covOn:
                    arg1 = this_mean + 3 * this_stds
                    arg2 = this_mean - 3 * this_stds
                    if ii in [1, 2, 4, 5]:
                        arg1 = np.degrees(arg1)
                        arg2 = np.degrees(arg2)
                    axs[ii].fill_between(this_ts, arg1, arg2, alpha=0.3, label=this_label, color=color)

                arg = this_mean
                if ii in [1, 2, 4, 5]:
                    arg = np.degrees(arg)
                axs[ii].plot(np.array(this_ts), arg, label=this_mean_label, color=color, linewidth=this_weight, alpha=1)
                axs[ii].set_title(titles[ii])


        for ax in axs[len(titles):]:
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        handles, labels = axs[-1].get_legend_handles_labels()

        axs[-1].legend(handles, labels, loc='center')
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs.tolist(), label='Weight')
        fig.suptitle(f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def plotDifferentialEntropy(self):
        custom_palette = sns.color_palette("flare", len(self.trajectories))
        fig, axs = plt.subplots(1,1,figsize=(7, 5))

        for ii, traj in enumerate(self.trajectories):
            axs.scatter(traj['times'], traj['DE'], color=custom_palette[ii], label=f'Mixand {ii}', marker='o', s=100*traj['weight'])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Differential Entropy')
        axs.set_yscale('log')
        if self.split_criterion == 'DE':
            plt.axhline(y=self.DE_threshold, color='crimson', linestyle='--', label='Threshold')
        # axs.legend()
        fig.suptitle(f'DE: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
        plt.tight_layout()

    def plotPyestMetrics(self):
        custom_palette = sns.color_palette("flare", len(self.trajectories))
        fig, axs = plt.subplots(1,1,figsize=(7, 5))

        for ii, traj in enumerate(self.trajectories):
            axs.scatter(traj['times'], traj['SOLC_R_scores'], color=custom_palette[ii], label=f'Mixand {ii}', marker='o', s=100*traj['weight'])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('SOLC along R')
        axs.set_yscale('log')
        if self.split_criterion == 'SOLC_R':
            plt.axhline(y=self.SOLC_threshold, color='crimson', linestyle='--', label='Threshold')
        # axs.legend()
        fig.suptitle(f'SOLC_R: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
        plt.tight_layout()

    def plotLinearEntropy(self):
        custom_palette = sns.color_palette("flare", len(self.trajectories))
        fig, axs = plt.subplots(1,1,figsize=(7, 5))

        for ii, traj in enumerate(self.trajectories):
            axs.scatter(traj['times'], traj['lin_entropy'], color=custom_palette[ii], label=f'Mixand {ii}', marker='o', s=100*traj['weight'])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Linearized Entropy')
        # axs.legend()
        fig.suptitle(f'Linear Entropy: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
        plt.tight_layout()

    def plotLinearNonlinearEntropy(self):
        custom_palette = sns.color_palette("flare", len(self.trajectories))
        custom_palette1 = sns.color_palette("crest", len(self.trajectories))
        fig, axs = plt.subplots(1,1,figsize=(7, 5))

        for ii, traj in enumerate(self.trajectories):
            axs.scatter(traj['times'], traj['lin_entropy'], color=custom_palette[ii], label=f'Mixand {ii}', marker='o', s=100*traj['weight'])
            axs.scatter(traj['times'], traj['nonlin_entropy'], color=custom_palette1[ii], label=None, marker='d', s=50*traj['weight'])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Differential Entropy')
        plt.legend(['Linear', 'Nonlinear'])
        # axs.legend()
        fig.suptitle(f'Differential Entropy: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
        plt.tight_layout()

    def plotSplitMixands(self, P0, m0, ind1, ind2, Ps_new, ms_new, weights, t):
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']
        P = P0[[ind1, ind1, ind2, ind2],[ind1,ind2,ind1,ind2]].reshape((2,2))
        mean = m0[[ind1, ind2]]
        if t == 0:
            offset0 = 0.0001
            offset1 = 0.01
        else:
            offset0 = 0.005
            offset1 = 0.01
        x = np.linspace(m0[ind1] - offset0*abs(m0[ind1]), m0[ind1] + offset0*abs(m0[ind1]), 100)
        y = np.linspace(m0[ind2] - offset1*abs(m0[ind2]), m0[ind2] + offset1*abs(m0[ind2]), 100)
        [X,Y] = np.meshgrid(x,y)
        pos = np.dstack((X, Y))
        input_pdf = multivariate_normal.pdf(pos, mean, P, allow_singular=True)
        output_pdfs = []
        for ii in range(len(ms_new)):
            m_new = ms_new[ii][[ind1, ind2]]
            P_new = Ps_new[ii][[ind1, ind1, ind2, ind2],[ind1,ind2,ind1,ind2]].reshape((2,2))
            output_pdfs.append(weights[ii] * multivariate_normal.pdf(pos, m_new, P_new, allow_singular=True))
        output_pdf = np.sum(output_pdfs, axis=0)

        gridspec = {'width_ratios': [1, 1, 0.1]}
        fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw=gridspec)

        c1 = axs[0].contourf(X, Y, input_pdf, cmap='flare_r')
        # plt.colorbar(c1, ax=axs[0])
        axs[0].set_title('Input PDF')
        axs[0].set_xlabel(titles[ind1])
        axs[0].set_ylabel(titles[ind2])
        axs[0].set_yticks([])
        axs[0].set_xticks([])

        c2 = axs[1].contourf(X, Y, output_pdf, cmap='flare_r')
        # plt.colorbar(c2, ax=axs[1])
        axs[1].set_title('Output PDF')
        axs[1].set_xlabel(titles[ind1])
        axs[1].set_ylabel(titles[ind2])
        axs[1].set_yticks([])
        axs[1].set_xticks([])

        cax = axs[2]
        plt.colorbar(c2, cax=cax)

        # plt.suptitle(f'Split PDF along State {ind1} and State {ind2} at time {t}')
        plt.tight_layout()

    def plotNumMixandsOverTime(self):
        fig, axs = plt.subplots(1, 1, figsize=(7, 5))

        numMixands = np.zeros(len(self.ts)-1)
        for t in range(len(self.ts)):
            for traj in self.trajectories:
                if len(traj['times']) > 1:
                    if traj['times'][0] <= self.ts[t] <= traj["times"][-2]:
                        numMixands[t] += 1


        axs.plot(self.ts[:-1], numMixands)
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Number of Mixands')
        # axs.legend()
        fig.suptitle(
            f'Number of Mixands: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
        plt.tight_layout()

    def compareMixandsToMonte(self, MCData, covOn=False):
        custom_palette = sns.color_palette("flare", as_cmap=True)
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]', r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]', r'Heading Angle, $\psi$ [$^\circ$]']
        if 'dens' in self.params['eomTag'].lower():
            titles.append(r'Density, $\rho$ [kg/m$^3$]')
        nRows = int(np.ceil(len(titles) / 4))
        fig, axs = plt.subplots(nRows, 4, figsize=(15, 10))
        axs = np.atleast_1d(axs).ravel()

        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixands')):
            # color = f"C{next(colors)}"
            # color = custom_palette[ii]
            this_weight = traj['weight']
            color = custom_palette(this_weight)
            this_ts = traj['times']
            for ii in range(self.stateSize - self.params['d_KLE']):
                this_mean = np.array(traj['means'])[:, ii]
                this_stds = np.sqrt(np.array(traj['covs'])[:, ii, ii])
                if traj['id'] == 0:
                    this_label = f'3$\sigma$'
                    this_mean_label = 'Mixand Mean'
                else:
                    this_label = None
                    this_mean_label = None
                if covOn:
                    arg1 = this_mean + 3 * this_stds
                    arg2 = this_mean - 3 * this_stds
                    if ii in [1, 2, 4, 5]:
                        arg1 = np.degrees(arg1)
                        arg2 = np.degrees(arg2)
                    axs[ii].fill_between(this_ts, arg1, arg2, alpha=0.3,
                                                      label=this_label, color=color)

                arg = this_mean
                if ii in [1, 2, 4, 5]:
                    arg = np.degrees(arg)
                axs[ii].plot(np.array(this_ts), arg,
                                          label=this_mean_label, color=color, linewidth=this_weight, alpha=1)
                axs[ii].set_title(titles[ii])


        # Plot Monte Carlo mean and Covariance
        this_ts = self.ts
        for ii in range(self.stateSize - self.params['d_KLE']):
            this_mean = np.mean(MCData[:, ii, :], axis=0)
            this_stds = np.sqrt(np.var(MCData[:, ii, :], axis=0))
            if covOn:
                arg1 = this_mean + 3 * this_stds
                arg2 = this_mean - 3 * this_stds
                if ii in [1, 2, 4, 5]:
                    arg1 = np.degrees(arg1)
                    arg2 = np.degrees(arg2)
                axs[ii].fill_between(this_ts, arg1, arg2, alpha=0.1,
                                                  label=f'Monte Carlo 3$\sigma$', color='mediumspringgreen')

            arg = this_mean
            if ii in [1, 2, 4, 5]:
                arg = np.degrees(arg)
            axs[ii].plot(np.array(this_ts), arg, label='Monte Carlo Mean', color='mediumspringgreen', alpha=0.8, linewidth=0.5)
            axs[ii].set_title(titles[ii])

        for ax in axs[len(titles):]:
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        handles, labels = axs[0].get_legend_handles_labels()

        axs[-1].legend(handles, labels,
                   loc='center')
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs.tolist(), label='Weight')
        fig.suptitle(
            f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def plotTerminalDistribution(self, MCData, state1, state2, covOn=True):
        # Get all weights for non-stale terminal trajectories
        weights = []
        for jj, traj in enumerate((self.trajectories)):
            if traj['isStale'] == False:
                weights.append(traj['weight'])
        # normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

        custom_palette = sns.color_palette("flare", as_cmap=True)
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]', r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]', r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']

        plt.xlabel(titles[state1])
        plt.ylabel(titles[state2])

        zord = 0

        # Plot Monte Carlo data, mean and Covariance
        MC_final_states = MCData[:, [state1, state2], -1]
        MC_mean = np.mean(MC_final_states, axis=0)


        for ii in trange(len(MC_final_states), desc='Monte Carlo'):
            if ii == 0:
                this_label = "Monte Carlo Data"
            else:
                this_label = None
            if state1 in [1, 2, 4, 5]:
                MC_final_states[ii, 0] = np.degrees(MC_final_states[ii, 0])
            if state2 in [1, 2, 4, 5]:
                MC_final_states[ii, 1] = np.degrees(MC_final_states[ii, 1])
            plt.scatter(MC_final_states[ii, 0], MC_final_states[ii, 1], color='slategrey', s=10, zorder=zord, label=this_label,
                        alpha=0.5)
            zord += 1

        if state1 in [1, 2, 4, 5]:
            MC_mean[0] = np.degrees(MC_mean[0])
        if state2 in [1, 2, 4, 5]:
            MC_mean[1] = np.degrees(MC_mean[1])
        plt.scatter(MC_mean[0], MC_mean[1], color='darkorchid', s=50, label='Monte Carlo Mean', zorder=zord + 100,
                    alpha=1)


        label_count = zord
        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixandsCov')):
            this_weight = traj['weight']
            if len(weights) > 1:
                color = custom_palette((this_weight - min(weights)) / (max(weights) - min(weights)))
            else:
                color = custom_palette(custom_palette.N)

            if traj['isStale'] == False:
                if zord == label_count:
                    this_label = f'3$\sigma$'
                else:
                    this_label = None
                mean = np.array(traj['means'])[-1, [state1, state2]]
                # Covariance
                if covOn:
                    cov = np.array(traj['covs'])[-1, [state1, state1, state2, state2], [state1, state2, state1, state2]].reshape((2,2))
                    if state1 in [1, 2, 4, 5]:
                        cov[0,0] = cov[0,0]*(180/np.pi)**2
                        cov[0,1] = cov[0,1]*180/np.pi
                        cov[1,0] = cov[1,0]*180/np.pi
                        mean[0] = np.degrees(mean[0])
                    if state2 in [1, 2, 4, 5]:
                        mean[1] = np.degrees(mean[1])
                        cov[1,1] = cov[1,1]*(180/np.pi)**2
                        cov[0,1] = cov[0,1]*180/np.pi
                        cov[1,0] = cov[1,0]*180/np.pi
                    ellipse = self.getCovEllipse(mean, cov, 3)
                    plt.plot(ellipse[0, :], ellipse[1, :], label=this_label, color=color, alpha=0.3, linewidth=1, zorder=zord)
                    zord += 1

        label_count = zord
        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixandsMean')):
            this_weight = traj['weight']
            if len(weights) > 1:
                color = custom_palette((this_weight - min(weights)) / (max(weights) - min(weights)))
            else:
                color = custom_palette(custom_palette.N)

            if traj['isStale'] == False:
                if zord == label_count:
                    this_mean_label = 'Mixand Mean'
                else:
                    this_mean_label = None
                # Mean
                mean = np.array(traj['means'])[-1, [state1, state2]]
                if state1 in [1, 2, 4, 5]:
                    mean[0] = np.degrees(mean[0])
                if state2 in [1, 2, 4, 5]:
                    mean[1] = np.degrees(mean[1])
                plt.scatter(mean[0], mean[1], label=this_mean_label, color=color, s=500*this_weight, zorder=zord)
                zord += 1

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.legend()
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs, label='Weight')
        fig.suptitle(
            f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def getCovEllipse(self, mean, cov, sigma):
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Calculate the angle of the ellipse
        angle = np.arctan2(*eigenvectors[:, 0][::-1])
        # print(np.degrees(angle))

        # Width and height of the ellipse are 2 * sqrt(eigenvalues)
        width, height = 2 * sigma * np.sqrt(eigenvalues)

        # Generate points on a unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)])

        # Scale and rotate the unit circle to get the ellipse
        ellipse = np.dot(np.diag([width / 2, height / 2]), unit_circle)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = np.dot(rotation_matrix, ellipse)

        # Translate the ellipse to the mean
        ellipse[0, :] += mean[0]
        ellipse[1, :] += mean[1]

        # # Eigen decomposition
        # eigvals, eigvecs = np.linalg.eig(cov)
        #
        # # Calculate the angle of rotation
        # angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        #
        # # Define the ellipse in unit space
        # theta = np.linspace(0, 2 * np.pi, 100)
        # ellipse = np.array([np.cos(theta), np.sin(theta)])
        #
        # # Transform the ellipse to match the covariance matrix
        # ellipse = np.dot(np.diag(np.sqrt(eigvals)), ellipse)
        # ellipse = np.dot(eigvecs, ellipse)
        #
        # # Shift the ellipse to the mean
        # ellipse = sigma*ellipse + mean[:, np.newaxis]
        return ellipse

    def plotSplittingDirection(self, R):
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']

        plt.xlabel('Time [s]')
        plt.ylabel('Split Direction')
        ts_split = np.array(self.split_directions['ts'])
        dirs = np.array(self.split_directions['dirs'])

        for ii in range(self.stateSize):
            # this_label = 'DSTT Directions' if ii == 0 else None
            this_label = None
            plt.hlines(y=abs(R[ii]), xmin=ts_split[0], xmax=ts_split[-1], colors=f'C{ii}', linestyles='--', label=this_label)

        for ii in range(self.stateSize):
            plt.plot(ts_split, abs(dirs[:, ii]), 'o-', label=titles[ii], c=f'C{ii}',)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.legend(title='Solid = Covariance\nDashed = DSTT')
        plt.yscale('symlog')
        axs.set_yticks([0, 1e-1, 1])
        fig.suptitle(
            f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def compareMonteGMMInitialStateHistogram(self, MCData):
        custom_palette = sns.color_palette("flare", as_cmap=True)
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'ln(Density), $\ln(\rho)$ [kg/m$^3$]']

        # Get initial states for all non-stale trajectories at t = 0
        initial_states = []
        initial_covs = []
        initial_weights = []
        for traj in self.trajectories:
            if traj['isStale'] == False and 0 in traj['times']:
                initial_states.append(traj['means'][0])
                initial_covs.append(traj['covs'][0])
                initial_weights.append(traj['weight'])

        # Get initial states for all Monte Carlo data
        MC_initial_states = MCData[:, :, 0]

        for ii in range(self.stateSize):
            # Plot Monte Carlo data
            sns.histplot(MC_initial_states[:, ii] - self.init_mean[ii], ax=axs[ii // 4, ii % 4], color='C0', kde=True, label='Monte Carlo', alpha=0.5, stat="density")
            # Plot Gaussian pdf with mean of initials states and variance from initial covariance matrix for each trajectory
            xs = []
            ys = []
            for jj in range(len(initial_states)):
                mean = initial_states[jj][ii] - self.init_mean[ii]
                std = np.sqrt(initial_covs[jj][ii, ii])
                x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
                y = norm.pdf(x, mean, std)
                axs[ii // 4, ii % 4].plot(x, initial_weights[jj]*y, label=f'Traj {jj}', color=custom_palette(initial_weights[jj]), alpha=0.5)
                xs.append(x)
                ys.append(y)
            # Plot weighted sum of all pdfs
            # Get min and max of all xs
            min_x = min([min(x) for x in xs])
            max_x = max([max(x) for x in xs])
            x = np.linspace(min_x, max_x, 100)
            y = np.zeros_like(x)
            for jj in range(len(initial_states)):
                y += initial_weights[jj] * np.interp(x, xs[jj], ys[jj])
            axs[ii // 4, ii % 4].plot(x, y, label='Weighted Sum', color=custom_palette(1000), linewidth=2)

            axs[ii // 4, ii % 4].set_title(titles[ii])
            axs[ii // 4, ii % 4].set_xlabel('')
            axs[ii // 4, ii % 4].set_ylabel('')

        axs[1, 3].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        axs[1, 2].legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs[1, 2], label='Weight')
        fig.suptitle(
            f'Initial Monte Carlo / GMM State Distributions')


    def lookAtBadMonteInitialStates(self, MCData, Vf, rf):
        custom_palette = sns.color_palette("flare", as_cmap=True)
        # Get indices of all Monte Carlo data points where the final velocity is less than Vf and final position is greater than rf
        bad_inds = []
        for ii in range(len(MCData)):
            V = MCData[ii, -1, 3]
            r = MCData[ii, -1, 0]
            if V < Vf or r > rf:
                bad_inds.append(ii)

        # Plot the initial state perturbation histograms for these points from Monte Carlo data
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                    r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                    r'Heading Angle, $\psi$ [$^\circ$]', r'ln(Density), $\ln(\rho)$ [kg/m$^3$]']

        for ii in range(self.stateSize):
            sns.histplot(MCData[bad_inds, 0, ii] - self.init_mean[ii], ax=axs[ii // 4, ii % 4], color='C0', label='Monte Carlo', alpha=0.5, stat="density")

         # Also plot a histogram of the non-stale trajectory's initial means to see if there is overlap
        # Get initial states for all non-stale trajectories at t = 0
        initial_states = []
        initial_covs = []
        initial_weights = []
        for traj in self.trajectories:
            if traj['isStale'] == False and 0 in traj['times']:
                initial_states.append(traj['means'][0])
                initial_covs.append(traj['covs'][0])
                initial_weights.append(traj['weight'])

        for ii in range(self.stateSize):
            # Plot Gaussian pdf with mean of initials states and variance from initial covariance matrix for each trajectory
            xs = []
            ys = []
            for jj in range(len(initial_states)):
                mean = initial_states[jj][ii] - self.init_mean[ii]
                std = np.sqrt(initial_covs[jj][ii, ii])
                x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
                y = norm.pdf(x, mean, std)
                label = f'Traj {jj}' if jj == 0 else None
                axs[ii // 4, ii % 4].plot(x, initial_weights[jj]*y, label=label, color=custom_palette(initial_weights[jj]), alpha=0.5)
                xs.append(x)
                ys.append(y)
            # Plot weighted sum of all pdfs
            # Get min and max of all xs
            min_x = min([min(x) for x in xs])
            max_x = max([max(x) for x in xs])
            x = np.linspace(min_x, max_x, 100)
            y = np.zeros_like(x)
            for jj in range(len(initial_states)):
                y += initial_weights[jj] * np.interp(x, xs[jj], ys[jj])
            axs[ii // 4, ii % 4].set_title(titles[ii])
            axs[ii // 4, ii % 4].set_xlabel('')
            axs[ii // 4, ii % 4].set_ylabel('')
            axs[ii // 4, ii % 4].plot(x, y, label='Weighted Sum', color=custom_palette(1000), linewidth=2)

        axs[1, 3].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        axs[1, 2].legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs[1, 2], label='Weight')
        fig.suptitle(f'Initial Monte Carlo State Distributions for Bad Final States')

        return bad_inds



def main():
    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    ############################################################################
    # Inputs
    basePath = '/Users/gracecalkins/Local Documents/local_code/sttgmm'
    seabornSettings()
    stt_order = 4  # STT order
    dstt_order = 4
    atmoTag = 'single_exp'
    sttTag = 'get_Xdstts_ln_dens'
    MC_tag = 'uranus_ln_dens_MC'
    MC_num = 1000
    thisRunTag = 'test_gmm_prop'
    runGMM = True
    pltFlag = True


    ############################################################################
    ts, x0, params, tag, _, init_cov, proc_cov = uranus_inputs_ln_dens(basePath, atmoTag)
    dt = ts[1] - ts[0]
    dataLoadPath = os.path.join(basePath, 'data', sttTag)
    dataPath = os.path.join(basePath, 'data', thisRunTag)
    figPath = os.path.join(basePath, 'figures', thisRunTag)
    if not os.path.exists(figPath):
        os.makedirs(figPath)
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    DSTTs_nd, Rs_nd, Xs_nd, STTs_nd, MCData, sDSTTs_nd, sRs_nd, qDSTTs_nd, qRs_nd, pDSTTs_nd, pRs_nd, hoDSTTs_nd, hoRs_nd = loadData(dataLoadPath, basePath, dstt_order, stt_order, sttTag, dt, MC_tag, MC_num)

    ## Dynamic pressure DSTTs
    # xDSTTs_nd = pDSTTs_nd
    # xRs_nd = pRs_nd
    ## Selective DSTTs
    xDSTTs_nd = sDSTTs_nd
    xRs_nd = sRs_nd
    ## QOI DSTTs
    # xDSTTs_nd = qDSTTs_nd
    # xRs_nd = qRs_nd
    ## HO DSTTs
    # xDSTTs_nd = hoDSTTs_nd
    # xRs_nd = hoRs_nd
    xdstt_order = 3


    # Construct GMM UQ object
    num_splits = 3
    split_method = 'arbitrary' # 'dstt' 'covariance' 'manualCov' 'dsttCGT' 'manualDSTT' 'arbitrary'
    prop_method = 'xDSTT' # 'STT', 'UT', 'DSTT' 'Linear' 'xDSTT'
    split_criterion = 'none' # 'DE' 'none' 'manual'
    init_split = True
    init_split_multi = 1  # Number of times to perform initial split (1 for one split)
    intermed_split = False # Perform intermediate splits
    DE_threshold = 1
    procNoiseFlag = False
    # init_cov = 10*init_cov
    splitDir = np.array([-0.0561,    0.0007,   -0.0039,    0.1241,    0.9844,   -0.0300,   -0.1073])

    # x0 = x0 + 0.1*x0
    if runGMM:
        gmm_uq = GMM_UQ(ts, params, x0, init_cov, proc_cov, num_splits, split_method, prop_method, split_criterion, DSTTs_nd, Rs_nd, xDSTTs_nd, xRs_nd, STTs_nd, Xs_nd, dstt_order, stt_order, xdstt_order, DE_threshold, init_split, init_split_multi, intermed_split, procNoiseFlag, splitDir)
        gmm_uq.run()
        gmm_uq.printPerformanceMetrics()
        gmm_uq.saveSelf(dataPath, thisRunTag)
    else:
        DE_str = str(DE_threshold).replace('.', '_')
        gmm_uq = np.load(os.path.join(dataPath, f'gmm_uq_{prop_method}_{split_method}_{split_criterion}_{num_splits}_DE{DE_str}_IS{init_split}_{thisRunTag}.npy'), allow_pickle=True).item()

    DE_str = str(DE_threshold).replace('.', '_')
    save_suffix = f'{prop_method}_{split_method}_{split_criterion}_{num_splits}_DE{DE_str}_IS{init_split}_{thisRunTag}'


    if pltFlag:
        # gmm_uq.compareMonteGMMInitialStateHistogram(MCData)
        # plt.savefig(os.path.join(figPath, f'gmm_uq_init_state_{save_suffix}.png'))
        # plt.show()

        if len(gmm_uq.split_directions['ts']) > 0:
            gmm_uq.plotSplittingDirection(splitDir)
            plt.savefig(os.path.join(figPath, f'gmm_uq_split_dir_{save_suffix}.png'))
        # plt.show()
        state1, state2 = 0,3
        gmm_uq.plotTerminalDistribution(MCData, state1, state2, covOn=True)
        plt.savefig(os.path.join(figPath,f'gmm_uq_terminal_{state1}_{state2}_{save_suffix}.png'))
        gmm_uq.compareMixandsToMonte(MCData, covOn=True)
        plt.savefig(os.path.join(figPath, f'gmm_uq_MC_comp_{save_suffix}.png'))
        # plt.show()
        gmm_uq.plotMixandsOverTime(covOn=True)
        plt.savefig(os.path.join(figPath, f'gmm_uq_{save_suffix}.png'))
        # plt.show()
        # gmm_uq.plotDifferentialEntropy()
        # plt.savefig(os.path.join(figPath, f'gmm_uq_DE_{save_suffix}.png'))
        # plt.show()
        # gmm_uq.plotLinearNonlinearEntropy()
        # plt.savefig(os.path.join(figPath, f'gmm_uq_LNE_{save_suffix}.png'))
        # plt.show()
        # gmm_uq.plotNumMixandsOverTime()
        # plt.savefig(os.path.join(figPath, f'gmm_uq_num_mixands_{save_suffix}.png'))
        plt.show(block=False)

        # gmm_uq.plotLinearEntropy()
        # plt.savefig(os.path.join(figPath, f'gmm_uq_LE_{save_suffix}.png'))

    # Save gmm_uq mean and covariance for trajectory 0 to a npy file
    # np.save(os.path.join(dataPath, f'means_gmms_nd_{thisRunTag}_o_{dstt_order}.npy'), gmm_uq.trajectories[0].get('means'))
    # np.save(os.path.join(dataPath, f'covs_gmms_nd_{thisRunTag}_o_{dstt_order}.npy'), gmm_uq.trajectories[0].get('covs'))


if __name__ == '__main__':
    # profile code
    import cProfile
    cProfile.run('main()',  'restats')
    import pstats
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats(-1).print_stats(10)
    # main()
    plt.show()


