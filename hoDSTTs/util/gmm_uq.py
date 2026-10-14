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

from ..inputs.inputs import *
from JP_stts_aerocapture.plotters import seabornSettings
from .nonDim import *
from .differential_entropy import *
from .gmm_splitting import *
from .stt import *
from .params_planet import *
from .eoms_aerocapture import *
import warnings

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


class GMM_UQ:
    def __init__(self, ts, params, init_mean, init_cov, proc_cov, num_splits=3, split_method='covariance', prop_method=None, split_criterion='none', DSTTs=None, Rs=None, xDSTTs=None, xRs=None, STTs=None, Xs_nom=None, dstt_order=3, stt_order=3, xdstt_order=3, DE_threshold=-1, init_split=False, init_split_multi=3, intermed_split=False, procNoiseFlag=False, splitDir=None, useMultiRef=False, specifySplit=False, splitMeans=[], splitCovs=[], splitWeights=[], intSTTsFlag=False):
        # Given parameters
        self.ts = ts
        self.params = params
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.proc_cov = proc_cov
        self.num_splits = num_splits
        self.split_method = split_method
        self.prop_method = prop_method
        self.split_criterion = split_criterion
        self.DSTTs = DSTTs  # NONDIMENSIONAL
        self.xDSTTs = xDSTTs
        self.STTs = STTs # NONDIMENSIONAL
        self.Rs = Rs # NONDIMENSIONAL
        self.xRs = xRs
        self.Xs_nom = Xs_nom # NONDIMENSIONAL
        self.dstt_order = dstt_order
        self.stt_order = stt_order
        self.xdstt_order = xdstt_order
        self.DE_threshold = DE_threshold
        self.split_directions = {'ts': [],
                                 'dirs': []}
        self.init_split = init_split
        self.init_split_multi = init_split_multi
        self.intermed_split = intermed_split
        self.init_split_counter = 0
        self.procNoiseFlag = procNoiseFlag
        self.splitDir = splitDir
        self.useMultiRef = useMultiRef
        # Specify initial split means and covariances
        self.specifySplit = specifySplit
        self.splitMeans = splitMeans
        self.splitCovs = splitCovs
        self.splitWeights = splitWeights
        self.intSTTsFlag = intSTTsFlag

        # Derived parameters
        self.t = ts[0]
        self.dt = ts[1] - ts[0]
        self.stateSize = len(init_mean)
        self.split_count = 0

        # UT Specific Parameters
        self.kappa = 0
        self.beta = 2
        self.alpha = 0.1
        self.lambda_ = self.alpha ** 2 * (len(init_mean) + self.kappa) - len(init_mean)

        # Trajs will save everything dimensionally, but all propagation will be done nondimensionally
        self.trajectories = []
        H0 = get_nonlin_entropy(convertToNonDimDensCov(self.init_cov, self.params))
        traj = self.createNewTraj(0, self.t, init_mean, init_cov, H0, H0, 0, 1., H0)
        self.trajectories.append(traj)
        self.traj_id_counter = 1  # start next trajectory with id 1
        self.num_active_trajs = []
        self.weighted_mean = []
        # self.weighted_mean.append(init_mean)
        self.weighted_cov = []
        self.dm_outer_product = [[] for _ in range(self.num_splits ** self.init_split_multi)]
        # self.weighted_cov.append(init_cov)

    def createNewTraj(self, id, time, mean, cov, lin_entropy, nonlin_entropy, DE, weight, H0):
        traj = {'id': id,
                 'times': [time], # times of just this trajectory
                 'means': [mean],
                 'covs': [cov],
                 'isStale': False, # Change this to true once split
                 'lin_entropy': [lin_entropy], # linear entropy propagated with the ODE
                 'nonlin_entropy': [nonlin_entropy], # nonlinear entropy
                 'DE': [DE], # nonlin - lin entropy / H0
                 'weight': weight, # trajectory weight
                 'H0': H0, # Initial entropy to normalize by
                }
        return traj

    def integrateLinearEntropy(self, i):
        nonDimFlag = True
        if nonDimFlag:
            ts_nd = [self.t / np.sqrt(gravParams[self.params['planetName']].get('R_eq') / gravParams[self.params['planetName']].get('g_surf')),self.ts[i + 1] / np.sqrt(gravParams[self.params['planetName']].get('R_eq') / gravParams[self.params['planetName']].get('g_surf'))]
            for traj in self.trajectories:
                if not traj['isStale']:
                    mean_k_nd = convertToNonDimDens(traj['means'][-1], self.params)
                    H_lin_k = traj['lin_entropy'][-1]
                    sol = solve_ivp(lambda t, X: lin_entropy_ode(t, X, [mean_k_nd], self.params, 0), y0=np.array([H_lin_k]),
                                    t_span=[ts_nd[0], ts_nd[1]], method='RK45', rtol=1e-7, atol=1e-8)
                    E_lin_kp1 = sol.y[-1][-1]
                    traj['lin_entropy'].append(E_lin_kp1)
        else:
            ts = [self.t, self.ts[i + 1]]
            for traj in self.trajectories:
                if not traj['isStale']:
                    mean_k = traj['means'][-1]
                    H_lin_k = traj['lin_entropy'][-1]
                    sol = solve_ivp(lambda t, X: lin_entropy_ode_dim(t, X, [mean_k], self.params, 0),
                                    y0=np.array([H_lin_k]),
                                    t_span=[ts[0], ts[1]], method='RK45', rtol=1e-7, atol=1e-8)
                    E_lin_kp1 = sol.y[-1][-1]
                    traj['lin_entropy'].append(E_lin_kp1)
        return

    def trackDifferentialEntropy(self):
        for traj in self.trajectories:
            if not traj['isStale']:
                cov = traj['covs'][-1]
                P_nd = convertToNonDimDensCov(cov, self.params)
                E_lin_k = traj['lin_entropy'][-1]
                DE_k, H_nonlin_k = get_instantaneous_DE(P_nd, E_lin_k, traj['H0'])
                traj['DE'].append(DE_k)
                traj['nonlin_entropy'].append(H_nonlin_k)
        return

    def checkSplitCriteria(self):
        ids_to_split = []
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
                    P_nd = convertToNonDimDensCov(cov, self.params)
                    DE_k, _ = get_instantaneous_DE(P_nd, H_lin_k, traj['H0'])
                    if DE_k >= self.DE_threshold:
                        ids_to_split.append(traj['id'])
                        print(f"Adding traj {traj['id']} to split at time {self.t} with DE {DE_k}")
                elif self.split_criterion == "manual":
                    H_lin_k = traj['lin_entropy'][-1]
                    P_nd = convertToNonDimDensCov(cov, self.params)
                    DE_k, _ = get_instantaneous_DE(P_nd, H_lin_k, traj['H0'])
                    if self.t % 650 == 0:  #TODO hardcoded to test
                        if self.t > 0:
                            if traj['id'] % 1 == 0:
                                ids_to_split.append(traj['id'])
                                print(f"Adding traj {traj['id']} to split at time {self.t} with DE {DE_k}")
                        else:
                            ids_to_split.append(traj['id'])
                            print(f"Adding traj {traj['id']} to split at time {self.t}")
                elif self.split_criterion == 'none':
                    pass
                else:
                    raise ValueError('Invalid split criterion')


        return ids_to_split

    def splitTrajectories(self, ids_to_split, time_ind):
        for traj in self.trajectories:
            # If the current trajectory is on the list, split it using the desired method into the desired number of mixands
            if traj['id'] in ids_to_split:
                # Get current mean and covariance for this trajectory
                mean = traj['means'][-1]
                cov = traj['covs'][-1]
                parent_weight = traj['weight']
                # Split the trajectory
                if self.split_method == 'covariance':
                    # Split along covariance direction, nondimensionally
                    weights, means_nd, covs_nd, split_dir = splitMixandsCovariance(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params))
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                elif self.split_method == 'dstt':
                    # Split along DSTT direction, nondimensionally
                    weights, means_nd, covs_nd, split_dir = splitMixandsDSTT(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params), self.Rs[time_ind])
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                elif self.split_method == 'dsttCGT':
                    # TODO THIS DOES NOT WORK
                    # Split along DSTT direction, nondimensionally, using eigenvalues and vectors of the CGT until the end of the trajectory
                    weights, means_nd, covs_nd = splitMixandsDSTTCGT(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params), self.Rs[time_ind], self.DSTTs, time_ind, self.params, self.ts)
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                elif self.split_method == 'manualDSTT':
                    # Split along hardcoded direction, with DSTT method
                    # X is [r, theta, phi, V, gamma, psi, rho]
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd, split_dir = splitMixandsDSTT(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params), split_dir)
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                elif self.split_method == 'manualCov':
                    # Split along hardcoded direction, with DSTT method
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd, split_dir = splitMixandsCovarianceDir(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params), split_dir)
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                elif self.split_method == 'arbitrary':
                    # Split along hardcoded direction with Vittaldev's method
                    split_dir = self.splitDir
                    weights, means_nd, covs_nd = splitMixandsArbitaryDirection(self.num_splits, convertToNonDimDensCov(cov, self.params), convertToNonDimDens(mean, self.params), split_dir)
                    means = [convertFromNonDimDens(mean_nd, self.params) for mean_nd in means_nd]
                    covs = [convertFromNonDimDensCov(cov_nd, self.params) for cov_nd in covs_nd]
                else:
                    raise ValueError('Invalid split method')
                # Create new trajectories for each mixand
                for ii, mean_i in enumerate(means):
                    H0 = get_nonlin_entropy(convertToNonDimDensCov(covs[ii], self.params))
                    print(f"New traj {self.traj_id_counter} created with H0 {H0}")
                    # Check if the new covariance is PSD
                    if np.any(np.abs(np.linalg.eigvals(covs[ii]) - 1e-16) <= 0):
                        print(f"Negative covariance at time {self.t + self.dt}")
                        print(np.linalg.eigvals(covs[ii]))
                        blah = 42
                    traj_new = self.createNewTraj(self.traj_id_counter, self.t, mean_i, covs[ii], H0, H0, 0, parent_weight*weights[ii], H0)
                    self.trajectories.append(traj_new)
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

    def splitTrajectoriesManual(self):
        # Set all current trajectories as stale
        for traj in self.trajectories:
            traj['isStale'] = True

        for ii, mean in enumerate(self.splitMeans):
            print(f"New traj {self.traj_id_counter} created MANUALLY")
            # Check if the specified covariance is PSD
            if np.any(np.abs(np.linalg.eigvals(self.splitCovs[ii]) - 1e-16) <= 0):
                print(f"Negative covariance at time {self.t + self.dt}")
                print(np.linalg.eigvals(self.splitCovs[ii]))
                blah = 42
            traj_new = self.createNewTraj(self.traj_id_counter, self.t, mean, self.splitCovs[ii], -1, -1, 0, self.splitWeights[ii], -1)
            self.trajectories.append(traj_new)
            self.traj_id_counter += 1

            self.split_count += 1
            self.split_directions['ts'].append(self.t)
            self.split_directions['dirs'].append(np.zeros(self.stateSize))

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

    def predictionStep(self, i):
        # Get nominal state at time k
        if not self.useMultiRef:  # single reference trajectory
            X_nomk_nd = self.Xs_nom[i, :]
            X_nomkp1_nd = self.Xs_nom[i+1, :]
        else:  # multiple reference trajectories
            X_nomk_nd, X_nomkp1_nd = [], []
            for nom in self.Xs_nom:
                X_nomk_nd.append(nom[i, :])
                X_nomkp1_nd.append(nom[i+1, :])
        # For each trajectory, propagate mean and covariance using desired method from time k to kp1
        # print(f'Time {self.t} s: Running {self.num_active_trajs[-1]} Mixands')
        # for traj in tqdm(self.trajectories, desc='Prediction Step', position=1, leave=False):
        traj_ind = 0
        for traj in self.trajectories:
            mean_k = traj['means'][-1]
            P_k = traj['covs'][-1]
            # # Check positive semi definiteness of covariance
            # if np.any(np.linalg.eigvals(P_k) <= 0):
            #     print(f"Trajectory {traj['id']} has non-positive semi-definite covariance at time {self.t}")
            #     continue

            if not traj['isStale']:
                if self.prop_method == "DSTT":
                    if not self.useMultiRef:
                        X_nomk_nd_i = X_nomk_nd
                        X_nomkp1_nd_i = X_nomkp1_nd
                        DSTTs_i = self.DSTTs
                        Rs_i = self.Rs
                    else:
                        X_nomk_nd_i = X_nomk_nd[traj_ind]
                        X_nomkp1_nd_i = X_nomkp1_nd[traj_ind]
                        DSTTs_i = self.DSTTs[traj_ind]
                        Rs_i = self.Rs[traj_ind]
                    dm_k = mean_k - convertFromNonDimDens(X_nomk_nd_i, self.params)
                    dm_k_nd = convertToNonDimDens(dm_k, self.params)
                    P_k_nd = convertToNonDimDensCov(P_k, self.params)
                    expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.dstt_order, self.stateSize)
                    DSTT_k_nd = DSTTs_i[i + 1, :] # TODO check index
                    R_k_nd = Rs_i[i + 1, :]
                    DSTT_list = reshape_DSTT(DSTT_k_nd, self.dstt_order, self.stateSize)
                    # sigma_xi_k_nd = np.sqrt(np.einsum('i,ij,j->', R_k_nd, P_k_nd, R_k_nd))
                    # (DSTT_list, order, stateSize, R, expected_list)
                    dm_kp1_nd = compute_expected_dm_DSTT(DSTT_list, self.dstt_order, self.stateSize, R_k_nd[0,:], expected_list_nd)
                    P_kp1_nd = compute_covariance_DSTT(DSTT_list, expected_list_nd, dm_kp1_nd,self.dstt_order, R_k_nd[0,:])
                    P_kp1 = convertFromNonDimDensCov(P_kp1_nd, self.params)
                    mean_kp1 = convertFromNonDimDens(X_nomkp1_nd_i + dm_kp1_nd, self.params)
                    if self.procNoiseFlag:
                        P_kp1 += self.proc_cov
                    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)

                elif self.prop_method == "xDSTT":
                    # TODO handle multiple reference trajectories
                    dm_k = mean_k - convertFromNonDimDens(X_nomk_nd, self.params)
                    dm_k_nd = convertToNonDimDens(dm_k, self.params)
                    P_k_nd = convertToNonDimDensCov(P_k, self.params)
                    expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.xdstt_order, self.stateSize)
                    xDSTT_k_nd = self.xDSTTs[i, :]  # indexing for DSTTs is time at the end of the interval
                    Rs_k_nd = self.xRs[i, :, :]
                    xDSTT_list = reshape_DSTT(xDSTT_k_nd, self.xdstt_order, self.stateSize)
                    dm_kp1_nd = compute_expected_dm_xDSTT(xDSTT_list, self.xdstt_order, self.stateSize, Rs_k_nd, expected_list_nd)
                    P_kp1_nd = compute_covariance_xDSTT(xDSTT_list, expected_list_nd, dm_kp1_nd, self.xdstt_order, Rs_k_nd)
                    P_kp1 = convertFromNonDimDensCov(P_kp1_nd, self.params)
                    mean_kp1 = convertFromNonDimDens(X_nomkp1_nd + dm_kp1_nd, self.params)
                    if self.procNoiseFlag:
                        P_kp1 += self.proc_cov
                    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)
                    test = np.diag(P_kp1)

                elif self.prop_method == "STT":
                    if not self.useMultiRef:
                        X_nomk_nd_i = X_nomk_nd
                        X_nomkp1_nd_i = X_nomkp1_nd
                        if not self.intSTTsFlag: # Use precomputed STTs
                            STTs_i = self.STTs
                        else:
                            # TODO integrate STTs about this mixand mean
                            blah = 42
                    else:
                        X_nomk_nd_i = X_nomk_nd[traj_ind]
                        X_nomkp1_nd_i = X_nomkp1_nd[traj_ind]
                        if not self.intSTTsFlag:  # Use precompute STTs
                            STTs_i = self.STTs[traj_ind]
                        else:
                            # TODO integrate STTs about effective mean
                            blah = 42
                    dm_k = -convertFromNonDimDens(X_nomk_nd_i, self.params) + mean_k
                    dm_k_nd = convertToNonDimDens(dm_k, self.params)
                    P_k_nd = convertToNonDimDensCov(P_k, self.params)
                    expected_list_nd = compute_expected(dm_k_nd, P_k_nd, self.stt_order, self.stateSize)
                    STTs_k_nd = STTs_i[i + 1, :]
                    phi_list_nd = reshape_STT(STTs_k_nd, self.stt_order, self.stateSize)
                    dm_kp1_nd = compute_expected_dm(phi_list_nd, expected_list_nd, self.stt_order, self.stateSize)
                    P_kp1_nd = compute_covariance(phi_list_nd, expected_list_nd, dm_kp1_nd, self.stt_order, self.stateSize)
                    mean_kp1 = convertFromNonDimDens(X_nomkp1_nd_i + dm_kp1_nd, self.params)
                    P_kp1 = convertFromNonDimDensCov(P_kp1_nd, self.params)
                    if self.procNoiseFlag:
                        P_kp1 += self.proc_cov
                    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)

                elif self.prop_method == "UT":
                    P_k_nd = convertToNonDimDensCov(P_k, self.params)
                    Sk_nd = np.linalg.cholesky(P_k_nd)
                    n = self.stateSize
                    # Generate 2n + 1 sigma points
                    xi_ks_nd = np.zeros((n, 2 * n + 1))
                    mean_k_nd = convertToNonDimDens(mean_k, self.params)
                    xi_ks_nd[:, 0] = mean_k_nd
                    for ii in range(n):
                        # rows not columns from ChatGPT
                        xi_ks_nd[:, ii + 1] = mean_k_nd + np.sqrt(n + self.lambda_) * Sk_nd[:, ii]
                        xi_ks_nd[:, ii + n + 1] = mean_k_nd - np.sqrt(n + self.lambda_) * Sk_nd[:, ii]

                    # Propagate each sigma point through the dynamics function
                    xi_kp1s_nd = np.zeros((n, 2 * n + 1))
                    for ii in range(2 * n + 1):
                        tspan_nd = [self.t / np.sqrt(
                            gravParams[self.params['planetName']].get('R_eq') / gravParams[self.params['planetName']].get('g_surf')),
                                    self.ts[i + 1] / np.sqrt(
                                        gravParams[self.params['planetName']].get('R_eq') / gravParams[self.params['planetName']].get(
                                            'g_surf'))]

                        sol = solve_ivp(lambda t, X: eoms3DOFLnDensNonDim(t, X, self.params), t_span=tspan_nd, y0=xi_ks_nd[:, ii], method='RK45', rtol=1e-13, atol=1e-18)
                        xi_kp1s_nd[:, ii] = sol.y[:, -1]
                    # Recombine resultant points to get predicted mean and covariance
                    P_kp1_nd = np.zeros((n, n))
                    Wm, Wc = self.getSigmaPointWeights()
                    mean_kp1_nd = np.zeros(n)
                    for ii in range(2 * n + 1):
                        mean_kp1_nd = mean_kp1_nd + Wm[ii] * xi_kp1s_nd[:, ii]
                    for ii in range(2 * n + 1):
                        # print(convertFromNonDimDens(xi_kp1s_nd[:, ii] - mean_kp1_nd, self.params))
                        P_kp1_nd += Wc[ii] * np.outer(xi_kp1s_nd[:, ii] - mean_kp1_nd, xi_kp1s_nd[:, ii] - mean_kp1_nd)
                    P_kp1_nd = 0.5 * (P_kp1_nd + P_kp1_nd.T)
                    mean_kp1 = convertFromNonDimDens(mean_kp1_nd, self.params)
                    P_kp1 = convertFromNonDimDensCov(P_kp1_nd, self.params)
                    if self.procNoiseFlag:
                        P_kp1 += self.proc_cov # MAKE SURE TO ONLY ADD PROCESS NOISE ONCE DIMENSIONALIZED
                    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)

                elif self.prop_method == "Linear":
                    # TODO handle multiple reference trajectories
                    dm_k = convertFromNonDimDens(X_nomk_nd, self.params) - mean_k
                    dm_k_nd = convertToNonDimDens(dm_k, self.params)
                    P_k_nd = convertToNonDimDensCov(P_k, self.params)
                    expected_list_nd = compute_expected(dm_k_nd, P_k_nd, 1, self.stateSize)
                    STTs_k_nd = self.STTs[i + 1, :]
                    phi_list_nd = reshape_STT(STTs_k_nd[0:self.stateSize**2], 1, self.stateSize)
                    dm_kp1_nd = compute_expected_dm(phi_list_nd, expected_list_nd, 1, self.stateSize)
                    P_kp1_nd = compute_covariance(phi_list_nd, expected_list_nd, dm_kp1_nd, 1, self.stateSize)
                    mean_kp1 = convertFromNonDimDens(X_nomkp1_nd + dm_kp1_nd, self.params)
                    P_kp1 = convertFromNonDimDensCov(P_kp1_nd, self.params)
                    if self.procNoiseFlag:
                        P_kp1 += self.proc_cov
                    P_kp1 = 0.5 * (P_kp1 + P_kp1.T)

                else:
                    raise ValueError('Invalid mean / cov propagation method')

                # Check if covariance is positive semi-definite
                if np.any(np.linalg.eigvals(P_kp1) <= 0):
                    print(f"Traj {traj['id']}: Negative covariance at time {self.t+self.dt}")
                    print(np.linalg.eigvals(P_kp1))
                    blah = 42

                traj['means'].append(mean_kp1)
                traj['covs'].append(P_kp1)
                traj['times'].append(self.ts[i + 1])
                traj_ind += 1
        return
    def run(self):
        if self.prop_method == 'STT':
            printMethod = f'{self.prop_method}, order {self.stt_order}'
        elif self.prop_method == 'xDSTT':
            printMethod = f'{self.prop_method}, order {self.xdstt_order}'
        elif self.prop_method == 'DSTT':
            printMethod = f'{self.prop_method}, order {self.dstt_order}'
        else:
            printMethod = f'{self.prop_method}'
        print(f'Starting GMM UQ with {printMethod} splitting via {self.split_method} into {self.num_splits} mixands with DE treshold {self.DE_threshold}...')
        ts_len = len(self.ts)
        step_times = []
        for i in range(ts_len - 1):
            tic = time.perf_counter()
            # Check if we need to split
            if i == 0:
                if not self.specifySplit:
                    for split in range(self.init_split_multi):
                        ids_to_split = self.checkSplitCriteria()
                        # Perform splitting if necessary before propagation
                        self.splitTrajectories(ids_to_split, i)
                else:
                    self.splitTrajectoriesManual()

            elif self.intermed_split:
                ids_to_split = self.checkSplitCriteria()
                # Perform splitting if necessary before propagation
                self.splitTrajectories(ids_to_split, i)
            self.updateWeightedMeanCov()
            # Prediction step
            self.predictionStep(i)
            # Integrate linear entropy
            self.integrateLinearEntropy(i)
            self.trackDifferentialEntropy()
            self.t = self.t + self.dt
            toc = time.perf_counter()
            step_times.append(toc - tic)
            if i % 10 == 0:
                print(f"Sim Time {self.t} / {self.ts[-1]} | Step Time {toc - tic:0.4f} s | Est Remaining Time {(np.mean(step_times) * (ts_len - i - 1))/60:0.4f} min")

        self.updateWeightedMeanCov()


    def updateWeightedMeanCov(self):
        # Get the weighted mean and covariance of all active trajectories
        weighted_mean = np.zeros(self.stateSize)
        weighted_cov = np.zeros((self.stateSize, self.stateSize))
        for traj in self.trajectories:
            if not traj['isStale']:
                weighted_mean += traj['weight'] * np.array(traj['means'][-1])
        ind = 0
        for traj in self.trajectories:
            if not traj['isStale']:
                dm = np.array(traj['means'][-1]) - weighted_mean
                weighted_cov += traj['weight'] * (np.array(traj['covs'][-1]) + np.outer(dm, dm))
                self.dm_outer_product[ind].append(np.outer(dm, dm))
                ind += 1
                # weighted_cov += traj['weight'] * (np.array(traj['covs'][-1]))
        self.weighted_mean.append(weighted_mean)
        self.weighted_cov.append(weighted_cov)

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
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']
        colors = itertools.cycle([0,1,2,3,4,5,6,7])

        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixands')):
            # color = f"C{next(colors)}"
            # color = custom_palette[ii]
            this_weight = traj['weight']
            color = custom_palette(this_weight)
            this_ts = traj['times']
            for ii in range(self.stateSize):
                this_mean = np.array(traj['means'])[:,ii]
                this_stds = np.sqrt(np.array(traj['covs'])[:,ii,ii])
                if traj['id'] == 1:
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
                    axs[ii // 4, ii % 4].fill_between(this_ts, arg1, arg2, alpha=0.3,
                                                      label=this_label, color=color)

                arg = this_mean
                if ii in [1, 2, 4, 5]:
                    arg = np.degrees(arg)
                axs[ii // 4, ii % 4].plot(np.array(this_ts), arg,
                                          label=this_mean_label, color=color, linewidth=this_weight, alpha=1)
                axs[ii // 4, ii % 4].set_title(titles[ii])

        axs[1, 3].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        axs[1, 2].legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs[1,2],label='Weight')
        fig.suptitle(f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def plotDifferentialEntropy(self):
        custom_palette = sns.color_palette("flare", len(self.trajectories))
        fig, axs = plt.subplots(1,1,figsize=(7, 5))

        for ii, traj in enumerate(self.trajectories):
            axs.scatter(traj['times'], traj['DE'], color=custom_palette[ii], label=f'Mixand {ii}', marker='o', s=100*traj['weight'])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Differential Entropy')
        if self.split_criterion == 'DE':
            plt.axhline(y=self.DE_threshold, color='crimson', linestyle='--', label='Threshold')
        # axs.legend()
        fig.suptitle(f'DE: {self.prop_method} GMM with {self.split_method} split on {self.split_criterion} criterion')
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
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']
        colors = itertools.cycle([0, 1, 2, 3, 4, 5, 6, 7])

        for jj, traj in enumerate(tqdm(self.trajectories, desc='PlotMixands')):
            # color = f"C{next(colors)}"
            # color = custom_palette[ii]
            this_weight = traj['weight']
            color = custom_palette(this_weight)
            this_ts = traj['times']
            for ii in range(self.stateSize):
                this_mean = np.array(traj['means'])[:, ii]
                this_stds = np.sqrt(np.array(traj['covs'])[:, ii, ii])
                if traj['id'] == 1:
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
                    axs[ii // 4, ii % 4].fill_between(this_ts, arg1, arg2, alpha=0.3,
                                                      label=this_label, color=color)

                arg = this_mean
                if ii in [1, 2, 4, 5]:
                    arg = np.degrees(arg)
                axs[ii // 4, ii % 4].plot(np.array(this_ts), arg,
                                          label=this_mean_label, color=color, linewidth=this_weight, alpha=1)
                axs[ii // 4, ii % 4].set_title(titles[ii])

        axs[1, 3].axis('off')

        # Plot Monte Carlo mean and Covariance
        this_ts = self.ts
        for ii in range(self.stateSize):
            this_mean = np.mean(MCData[:, ii, :], axis=0)
            this_stds = np.sqrt(np.var(MCData[:, ii, :], axis=0))
            if covOn:
                arg1 = this_mean + 3 * this_stds
                arg2 = this_mean - 3 * this_stds
                if ii in [1, 2, 4, 5]:
                    arg1 = np.degrees(arg1)
                    arg2 = np.degrees(arg2)
                axs[ii // 4, ii % 4].fill_between(this_ts, arg1, arg2, alpha=0.1,
                                                  label=f'Monte Carlo 3$\sigma$', color='darkblue')

            arg = this_mean
            if ii in [1, 2, 4, 5]:
                arg = np.degrees(arg)
            axs[ii // 4, ii % 4].plot(np.array(this_ts), arg,
                                      label='Monte Carlo Mean', color='darkblue', alpha=0.8)
            axs[ii // 4, ii % 4].set_title(titles[ii])


        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        axs[1, 2].legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
        normalized = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=normalized, cmap=sns.color_palette("flare", as_cmap=True))
        cmap.set_array([])
        fig.colorbar(cmap, ax=axs[1, 2], label='Weight')
        fig.suptitle(
            f'{self.prop_method} GMM Propagation with {self.split_method} split on {self.split_criterion} criterion')

    def plotTerminalDistribution(self,MCData, state1, state2, covOn=True):
        # Get all weights for non-stale terminal trajectories
        weights = []
        for jj, traj in enumerate((self.trajectories)):
            if traj['isStale'] == False:
                weights.append(traj['weight'])
        # normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

        custom_palette = sns.color_palette("flare", as_cmap=True)
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        titles = [r'Radial Distance, $r$ [m]', r'Longitude, $\theta$ [$^\circ$]', r'Latitude, $\phi$ [$^\circ$]',
                  r'Velocity, $V$ [m/s]', r'Flight Path Angle, $\gamma$ [$^\circ$]',
                  r'Heading Angle, $\psi$ [$^\circ$]', r'Density, $\rho$ [kg/m$^3$]']

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



