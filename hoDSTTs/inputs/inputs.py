# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for inputting initial conditions and parameters

import sys
import numpy as np
import os
from JP_stts_aerocapture.util.nonDim import *
from JP_stts_aerocapture.util.params_planet import gravParams, atmoParams #
from JP_stts_aerocapture.util.atmospheres import *
from JP_stts_aerocapture.util.aerocapture import *
from scipy.stats import special_ortho_group
from copy import deepcopy

def uranus_inputs_no_dens(basePath, atmoTag):
    # Only "atmospheric flight" portion of the trajectory without density state
    tag = "atmo_only"
    dataPathName = 'data'
    # Make a folder for tag if it does not exist in the data directory
    if not os.path.exists(os.path.join(basePath, dataPathName, tag)):
        os.makedirs(os.path.join(basePath,dataPathName, tag))
    dataPath = os.path.join(basePath,dataPathName, tag)

    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3
    V0_I = 24.936e3  # TODO
    # Assume
    # LD, beta, theta0, psi0, phi0, sigma = 0.24, 146, np.radians(-15.22), np.radians(36.51), np.radians(85.78), np.radians(0)
    LD, beta, theta0, phi0, psi0, sigma0 = 0.3, 140., np.radians(190), np.radians(-9.7), np.radians(45), np.radians(0)
    gamma0_I = np.radians(-11.1906906717)
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 17*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFNonDimNoDens",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    # init_cov = convertFromNonDimCov(np.diag(5e-6*np.ones(len(x0))), params)
    # init_cov[4,4] = 1e-5
    # proc_cov = 1e-4 * init_cov
    proc_cov = []
    init_cov = []


    return ts, x0, params, tag, dataPath, init_cov, proc_cov

def uranus_inputs_ln_dens(basePath, atmoTag):
    # Only "atmospheric flight" portion of the trajectory without density state
    tag = "ln_dens"
    dataPathName = 'data'


    atmoParams['Uranus']['rho_surf'] = 5.209878827646674e-06 # FALL AAS - also used for SFM
    atmoParams['Uranus']['H'] = 180719.42817051418  # FALL AAS - also used for SFM

    # Make a folder for tag if it does not exist in the data directory
    if not os.path.exists(os.path.join(basePath, dataPathName, tag)):
        os.makedirs(os.path.join(basePath,dataPathName, tag))
    dataPath = os.path.join(basePath,dataPathName, tag)

    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3
    V0_I = 24.936e3  # TODO
    # Assume
    # LD, beta, theta0, psi0, phi0, sigma = 0.24, 146, np.radians(-15.22), np.radians(36.51), np.radians(85.78), np.radians(0)
    LD, beta, theta0, phi0, psi0, sigma0 = 0.3, 140., np.radians(190), np.radians(-9.7), np.radians(45), np.radians(0)
    gamma0_I = np.radians(-11.1906906717)
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    if atmoTag == 'single_exp':
        rho0 = exponentialAtmosphereUranusSingle(h0)
    if atmoTag == 'polynomial':
        rho0 = polynomialAtmosphereUranus(h0)
    ln_rho_0 = np.log(rho0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 20*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-5*np.ones(len(x0))), params)
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, tag, dataPath, init_cov, proc_cov

def uranus_inputs_small_cov(basePath, atmoTag):
    # Fit to new atmosphere parameters
    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3 # [m] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    V0_I = 24.936e3  # [m/s] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    # Assume
    LD, beta, theta0, phi0, psi0, sigma0 = 0.25, 140., np.radians(190.045), np.radians(-9.764), np.radians(45), np.radians(0)
    # gamma0_I = np.radians(-10.5569012447)  # Middle of FPA corridor
    gamma0_I = np.radians(-11.0136565975)  # Lift-up FPA
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    atmoParams['Uranus']['rho_surf'] = 6.07086000e-03  # atmo fit 2/17/2025 from 200 to 1000 km
    atmoParams['Uranus']['H'] = 5.53847873e+04  # atmo fit 2/17/2025 from 200 to 1000 km

    if atmoTag == 'single_exp':
        rho0 = exponentialAtmosphereUranusSingle(h0)
    if atmoTag == 'polynomial':
        rho0 = polynomialAtmosphereUranus(h0)
    ln_rho_0 = np.log(rho0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 12*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-7*np.ones(len(x0))), params)
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov, []

def uranus_inputs_med_cov(basePath, atmoTag):
    # Fit to new atmosphere parameters
    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3 # [m] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    V0_I = 24.936e3  # [m/s] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    # Assume
    LD, beta, theta0, phi0, psi0, sigma0 = 0.25, 140., np.radians(190.045), np.radians(-9.764), np.radians(45), np.radians(0)
    # gamma0_I = np.radians(-10.5569012447)  # Middle of FPA corridor
    gamma0_I = np.radians(-11.0136565975)  # Lift-up FPA
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    if atmoTag == 'single_exp':
        rho0 = exponentialAtmosphereUranusSingle(h0)
    if atmoTag == 'polynomial':
        rho0 = polynomialAtmosphereUranus(h0)
    ln_rho_0 = np.log(rho0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 12*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    atmoParams['Uranus']['rho_surf'] = 6.07086000e-03,  # atmo fit 2/17/2025 from 200 to 1000 km
    atmoParams['Uranus']['H'] = 5.53847873e+04,  # atmo fit 2/17/2025 from 200 to 1000 km

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-6*np.ones(len(x0))), params)
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov

def uranus_inputs_large_cov(basePath, atmoTag):
    # Fit to new atmosphere parameters
    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3 # [m] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    V0_I = 24.936e3  # [m/s] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    # Assume
    LD, beta, theta0, phi0, psi0, sigma0 = 0.25, 140., np.radians(190.045), np.radians(-9.764), np.radians(45), np.radians(0)
    # gamma0_I = np.radians(-10.5569012447)  # Middle of FPA corridor
    gamma0_I = np.radians(-11.0136565975)  # Lift-up FPA
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    if atmoTag == 'single_exp':
        rho0 = exponentialAtmosphereUranusSingle(h0)
    if atmoTag == 'polynomial':
        rho0 = polynomialAtmosphereUranus(h0)
    ln_rho_0 = np.log(rho0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 12*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    atmoParams['Uranus']['rho_surf'] = 6.07086000e-03,  # atmo fit 2/17/2025 from 200 to 1000 km
    atmoParams['Uranus']['H'] = 5.53847873e+04,  # atmo fit 2/17/2025 from 200 to 1000 km

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-5*np.ones(len(x0))), params)
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov


def neptune_inputs(basePath, atmoTag):
    # Only "atmospheric flight" portion of the trajectory without density state
    tag = "neptune_base"
    dataPathName = 'data'
    # Make a folder for tag if it does not exist in the data directory
    if not os.path.exists(os.path.join(basePath, dataPathName, tag)):
        os.makedirs(os.path.join(basePath,dataPathName, tag))
    dataPath = os.path.join(basePath,dataPathName, tag)

    # https://www.sciencedirect.com/science/article/pii/S0094576522005422
    h0, V0_I = 1000e3, 29e3  # m, m/s
    # V0_I = 0  # TODO
    # Assume
    LD, beta, theta0, psi0, phi0, sigma = 0.3, 140, np.radians(0), np.radians(0), np.radians(45), np.radians(0)
    gamma0_I = np.radians(-11.258896206302618) # midpoint of the entry corridor
    V0_R, gamma0_R = inertial2planetRel('Neptune', V0_I, gamma0_I, h0, psi0, phi0)

    ln_rho_ref = 20

    planetName = 'Neptune'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    # Tf = 36*60  # s
    Tf = 500*60  # s # TODO for checkouts with terminate condition

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        # "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFNonDimNoDens",
        "V0_I": V0_I,
        }

    # TODO
    # init_cov = convertFromNonDimCov(np.diag(1e-5*np.ones(len(x0))), params)
    # init_cov[4,4] = 2e-5
    # proc_cov = 1e-4 * init_cov
    init_cov = np.diag(1e-5*np.ones(len(x0)))
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, tag, dataPath, init_cov, proc_cov


def uranus_inputs_fall_aas(basePath, **kwargs):
    h0, V0 = 1600e3, 24.936e3  # m, m/s
    # Assume
    LD, beta, theta0, psi0, phi0, sigma = 0.25, 145, np.radians(-9.764), np.radians(190.045), 0, np.radians(110.5)  # https://ntrs.nasa.gov/api/citations/20230017026/downloads/SciTech24_ECI_Trajectory_Paper_STRIVES.pdf
    atmoParams['Uranus']['rho_surf'] = 5.209878827646674e-06 # FALL AAS - also used for SFM
    atmoParams['Uranus']['H'] = 180719.42817051418  # FALL AAS - also used for SFM

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)

    gamma0 = np.radians(-16.3)
    # rho_ref = 5e-4
    ln_rho_ref = 20 # TODO

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 20*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0, gamma0, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)
    # V0_I = np.nan

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "c1": -1.01094680e+00,
        "c2": 8.67534770e+04,
        "c3": -1.05460738e+05,
        "c4": 4.01233924e-02,
        "c5": -2.72373694e+00,
        "c6": 7.82304642e-07,
        "c7": -8.37361698e-06,
        "c8": 2.56253449e-13,
        "c9": -1.06634136e-11,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }

    # UQ stuff
    # init_cov = np.diag(
    #     [500 ** 2, np.radians(0.1) ** 2, np.radians(0.1) ** 2, 100 ** 2, np.radians(0.1) ** 2, np.radians(0.1) ** 2,
    #      np.log(1e-8) ** 2])
    # proc_cov = 1e-4 * init_cov

    # TODO
    init_cov = convertFromNonDimDensCov(np.diag(1e-5*np.ones(len(x0))), params)
    init_cov[4,4] = 2e-5
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov, []


def uranus_weird_GMM(basePath, atmoTag):
    # Build a weird initial GMM / weights and propagate that through all methods
    # h0, V0_R = 1000e3, 29.36e3  # m, m/s
    h0 = 1000e3 # [m] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    V0_I = 24.936e3  # [m/s] from https://arc.aiaa.org/doi/10.2514/6.2024-0716
    # Assume
    LD, beta, theta0, phi0, psi0, sigma0 = 0.25, 140., np.radians(190.045), np.radians(-9.764), np.radians(45), np.radians(0)
    # gamma0_I = np.radians(-10.5569012447)  # Middle of FPA corridor
    gamma0_I = np.radians(-11.0136565975)  # Lift-up FPA
    V0_R, gamma0_R = inertial2planetRel('Uranus', V0_I, gamma0_I, h0, psi0, phi0)

    if atmoTag == 'single_exp':
        rho0 = exponentialAtmosphereUranusSingle(h0)
    if atmoTag == 'polynomial':
        rho0 = polynomialAtmosphereUranus(h0)
    ln_rho_0 = np.log(rho0)

    ln_rho_ref = 20

    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 14*60  # s

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    atmoParams['Uranus']['rho_surf'] = 6.07086000e-03,  # atmo fit 2/17/2025 from 200 to 1000 km
    atmoParams['Uranus']['H'] = 5.53847873e+04,  # atmo fit 2/17/2025 from 200 to 1000 km

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma0,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "H": 180719.42817051418,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim",
        "V0_I": V0_I,
        'r0': x0[0],
        'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        }


    proc_cov = convertFromNonDimDensCov(np.diag(1e-7*np.ones(len(x0))), params)
    init_cov = convertFromNonDimDensCov(np.diag(1e-5*np.ones(len(x0))), params)
    init_cov_cov_nd = np.diag(1e-7*np.ones(len(x0)))

    # Get weird initial means
    np.random.seed(42)
    numMixands = 5

    # 1. Generate Random Means in Nondimensional Space
    disp_inds = [0,3,4,6]  # only disperse position, velocity, FPA, and density
    x0s_dim = []
    for _ in range(numMixands):
        disp_vals = np.random.multivariate_normal(x0[disp_inds], init_cov[disp_inds][:,disp_inds], 1)
        x0_disp = deepcopy(x0)
        x0_disp[disp_inds] = disp_vals
        x0s_dim.append(x0_disp)

    # 2. Generate Random Covariance Matrices in Nondimensional Space
    init_covs_nd = []
    for _ in range(numMixands):
        # Create a base random covariance matrix
        cov_nd = init_cov_cov_nd
        # Add random cross-correlations for states in disp_inds
        for i in disp_inds:
            for j in disp_inds:
                if i != j:
                    cov_nd[i,j] = np.random.uniform(-1e-8, 1e-8)

        # Make symmetric
        cov_nd = (cov_nd + cov_nd.T) / 2

        # Custom scaling: reduce variance for angles, increase for others
        scales = np.ones(len(x0))  # Default scale factor = 1
        for i, state in enumerate(disp_inds):
            if i in [1,2,4,5]:  # Function to check if a state is an angle
                scales[state] = np.random.uniform(1e-8, 1e-4)  # Keep small variance
            else:
                scales[state] = np.random.uniform(0.1, 3)  # Allow larger variance

        scale_matrix = np.diag(scales)
        cov_nd = scale_matrix @ cov_nd @ scale_matrix  # Apply per-state scaling

        # Apply random rotation
        R = special_ortho_group.rvs(len(x0))  # Random orthogonal rotation matrix
        cov_nd = R @ cov_nd @ R.T  # Rotate covariance matrix

        init_covs_nd.append(cov_nd)

    # Convert covariance matrices back to dimensional space
    init_covs_dim = [convertFromNonDimDensCov(cov_nd, params) for cov_nd in init_covs_nd]

    # Check that all covariances are PSD
    for cov in init_covs_dim:
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            print("Covariance matrix is not PSD")

    # 3. Generate Random Component Weights
    # Generate initial weights
    raw_weights = np.random.dirichlet(alpha=np.random.uniform(0.2, 2, size=numMixands))
    # Ensure all weights are at least 0.1
    min_weight = 0.1
    adjusted_weights = raw_weights + min_weight
    # Renormalize to sum to 1
    weights = adjusted_weights / np.sum(adjusted_weights)

    return ts, x0s_dim, params, init_covs_dim, proc_cov, weights

def uranus_inputs_JP_aero_stts():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 13*60  # s

    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    h0 = 1000e3  # [m]
    V0_I = 24.936e3  # [m/s]
    theta0 = np.radians(190.045)  # [deg]
    phi0 = np.radians(-9.764)  # [deg]
    psi0 = np.radians(45)  # [deg]
    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    LD = 0.25
    beta = 145  # [kg/m^2]
    # From arbitrary selection so not full LU or LD
    sigma = np.radians(78) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    atmoParams['Uranus']['rho_surf'] = 6.40934589e-03
    atmoParams['Uranus']['H'] = 5.47204083e+04

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)
    ln_rho_ref = 20

    # TODO check m_ref
    m_ref = R_eq**3*rho0**(1-1/ln_rho_ref)

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim", # EOMs with density
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        'rho_surf': atmoParams['Uranus']['rho_surf'],
        'H': atmoParams['Uranus']['H'],
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-14*np.ones(len(x0))), params)
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov

def uranus_inputs_JP_aero_stts_larger_perts():
    # Revised 5/5 to have larger initial perturbation covariance
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 13*60  # s

    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    h0 = 1000e3  # [m]
    V0_I = 24.936e3  # [m/s]
    theta0 = np.radians(190.045)  # [deg]
    phi0 = np.radians(-9.764)  # [deg]
    psi0 = np.radians(45)  # [deg]
    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    LD = 0.25
    beta = 145  # [kg/m^2]
    # From arbitrary selection so not full LU or LD
    sigma = np.radians(78) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    atmoParams['Uranus']['rho_surf'] = 6.40934589e-03
    atmoParams['Uranus']['H'] = 5.47204083e+04

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)
    ln_rho_ref = 20

    # TODO check m_ref
    m_ref = R_eq**3*rho0**(1-1/ln_rho_ref)

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim", # EOMs with density
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        'rho_surf': atmoParams['Uranus']['rho_surf'],
        'H': atmoParams['Uranus']['H'],
        }

    # Increased from 1e-14 to 1e-5
    # nd_vars = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    base_cov = convertFromNonDimDensCov(np.diag(1e-14*np.ones(len(x0))), params)
    # init_cov = np.diag([(26.281e3/3)**2, np.radians(0.116/3)**2, np.radians(0.227/3)**2, (9.215/3)**2, np.radians(0.150/3)**2, np.radians(0.063/3)**2, (0.3)**2])
    init_cov = np.diag([(10e3/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (5/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (0)**2])
    # init_cov = np.diag([0,0,0,0,0,0,(0.005 * abs(ln_rho_0) / 3) ** 2])
    print("Dimensional Initial Variances:", np.diag(init_cov))
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov

def uranus_inputs_JP_aero_stts_drag_mod():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 13*60  # s

    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    h0 = 1000e3  # [m]
    V0_I = 24.936e3  # [m/s]
    theta0 = np.radians(190.045)  # [deg]
    phi0 = np.radians(-9.764)  # [deg]
    psi0 = np.radians(45)  # [deg]
    # SRC: "Flight control methodologies for Neptune aerocapture trajectories" Acta
    LD = 0
    betas = [10, 100]  # [kg/m^2]
    # From arbitrary selection so not full LU or LD
    sigma = 0 # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-9.29) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    atmoParams['Uranus']['rho_surf'] = 6.40934589e-03
    atmoParams['Uranus']['H'] = 5.47204083e+04

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)
    ln_rho_ref = 20

    # TODO check m_ref
    m_ref = R_eq**3*rho0**(1-1/ln_rho_ref)

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "betas": betas,
        "t_deploy": 429, # Drag skirt deploy time
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim", # EOMs with density
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        'rho_surf': atmoParams['Uranus']['rho_surf'],
        'H': atmoParams['Uranus']['H'],
        }

    init_cov = np.diag([(10e3/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (5/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (0)**2])
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov


def uranus_inputs_JP_aero_stts_larger_perts_fast():
    # Revised 5/5 to have larger initial perturbation covariance
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 5  # s
    Tf = 13*60  # s

    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    h0 = 1000e3  # [m]
    # Papers:  Mission Design and Navigation Solutions for a Uranus  Flagship-class Orbiter and Probe Using Aerocapture
    V_inf = 17.5e3 # m/s
    # using altitude that makes the inertial vel from the Performance analysis paper match V infinity of 13.6 km/s
    V0_I = np.sqrt(V_inf**2 + 2*gravParams[planetName]['mu']/(R_eq + 1140e3))  # [m/s]


    # V0_I = 24.936e3  # [m/s]
    theta0 = np.radians(190.045)  # [deg]
    phi0 = np.radians(-9.764)  # [deg]
    psi0 = np.radians(45)  # [deg]
    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    LD = 0.25
    beta = 145  # [kg/m^2]
    # From arbitrary selection so not full LU or LD
    sigma = np.radians(75.5) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-11.1902087423) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    atmoParams['Uranus']['rho_surf'] = 6.40934589e-03
    atmoParams['Uranus']['H'] = 5.47204083e+04

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)
    ln_rho_ref = 20

    # TODO check m_ref
    m_ref = R_eq**3*rho0**(1-1/ln_rho_ref)

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0, ln_rho_0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "rho_ref": ln_rho_ref, # use this to normalize, NOT FOR EXPONENTIAL ATMOSPHERE
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFLnNonDim", # EOMs with density
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        'rho_surf': atmoParams['Uranus']['rho_surf'],
        'H': atmoParams['Uranus']['H'],
        }

    # Increased from 1e-14 to 1e-5
    # nd_vars = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    base_cov = convertFromNonDimDensCov(np.diag(1e-14*np.ones(len(x0))), params)
    # init_cov = np.diag([(26.281e3/3)**2, np.radians(0.116/3)**2, np.radians(0.227/3)**2, (9.215/3)**2, np.radians(0.150/3)**2, np.radians(0.063/3)**2, (0.3)**2])
    init_cov = np.diag([(10e3/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (5/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (0)**2])
    # init_cov = np.diag([0,0,0,0,0,0,(0.005 * abs(ln_rho_0) / 3) ** 2])
    print("Dimensional Initial Variances:", np.diag(init_cov))
    proc_cov = 1e-4 * init_cov

    return ts, x0, params, init_cov, proc_cov

