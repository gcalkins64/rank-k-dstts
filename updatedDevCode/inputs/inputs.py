# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for inputting initial conditions and parameters

import sys
import numpy as np
import os
from dev_dstt_gmm.util.nonDim import *
from dev_dstt_gmm.util.params_planet import gravParams, atmoParams #
from dev_dstt_gmm.util.atmospheres import *
from dev_dstt_gmm.util.aerocapture import *
from dev_dstt_gmm.util.eoms_aerocapture import *
from scipy.stats import special_ortho_group
from copy import deepcopy

def uranus_inputs_JP_aero_stts_longdT():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 60  # s
    Tf = 14*60  # s

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
        'rho_surf': 6.40934586e-03,
        'H': 5.47204083e+04,
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-4*np.ones(len(x0))), params)
    proc_cov =  np.zeros_like(init_cov)

    return ts, x0, params, init_cov, proc_cov


def uranus_inputs_JP_aero_stts_shortdT():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 10  # s
    Tf = 14*60  # s

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
        'rho_surf': 6.40934586e-03,
        'H': 5.47204083e+04,
        }

    init_cov = convertFromNonDimDensCov(np.diag(1e-4*np.ones(len(x0))), params)
    proc_cov =  np.zeros_like(init_cov)

    return ts, x0, params, init_cov, proc_cov


def uranus_inputs_IPPW():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 10  # s
    Tf = 14*60  # s

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
    sigma = np.radians(45) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753) # [rad]
    # gamma0_I = np.radians(-16.3) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    atmoParams['Uranus']['rho_surf'] = 6.40934589e-03
    atmoParams['Uranus']['H'] = 5.47204083e+04

    atmoTag = 'single_exp'
    rho0 = exponentialAtmosphereUranusSingle(h0)
    ln_rho_0 = np.log(rho0)
    ln_rho_ref = 20

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
        'rho_surf': 6.40934586e-03,
        'H': 5.47204083e+04,
        }

    vars = 4e-6*np.ones(len(x0))
    vars[4] = 5e-6
    # vars[0] = 1e-6
    # vars[3] = 1e-6
    init_cov = convertFromNonDimDensCov(np.diag(vars), params)
    #

    # init_cov = np.diag([(10e3/3)**2, np.radians(0.1/3)**2, np.radians(0.1/3)**2, (500/3)**2, np.radians(0.2/3)**2, np.radians(0.1/3)**2, (0.1/3)**2])
    proc_cov =  np.zeros_like(init_cov)

    return ts, x0, params, init_cov, proc_cov


def uranus_inputs_disp_10state():
    """
    10-state dispersion input function: 6 physical states
    [r, theta, phi, V, gamma, psi] + 4 dispersed parameters [LD, beta, sigma, H].

    Uses eoms3DOFParams (NOT the ln-density augmented EOMs). Atmospheric
    dispersion is achieved by varying H and rho_surf directly through the
    `params` dict -- no global atmoParams mutation, so this is safe to
    disperse per-case under parallel execution without any race conditions.

    Returns
    -------
    ts       : time vector
    m0_full  : 10-dim mean vector [r0, theta0, phi0, V0, gamma0, psi0, LD, beta, sigma, H]
    params   : nominal params dict
    init_cov : 10x10 covariance matrix over [states, LD, beta, sigma, H]
    proc_cov : zeros (placeholder, unused for this study)
    """
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 10  # s
    Tf = 14 * 60  # s

    # SRC: "Performance Analysis of Aerocapture Systems for Uranus Orbiters" SciTECH 2024
    h0 = 1000e3  # [m]
    V0_I = 24.936e3  # [m/s]
    theta0 = np.radians(190.045)  # [rad]
    phi0 = np.radians(-9.764)  # [rad]
    psi0 = np.radians(45)  # [rad]

    LD = 0.25
    beta = 145  # [kg/m^2]
    sigma = np.radians(45)  # [rad]

    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753)  # [rad]
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # From 1a_get_atmo_fit, fit from 200 to 1000 km
    # NOTE: rho_surf and H now live only in `params` (used directly by
    # eoms3DOFParams) -- the global atmoParams dict is no longer touched.
    rho_surf_nom = 6.40934589e-03
    H_nom = 5.47204083e+04

    atmoTag = 'single_exp'

    # 6-state initial condition (NO ln-density state)
    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "3DOFParams",  # eoms3DOFParams: H/rho_surf read from params, not global atmoParams
        "V0_I": V0_I,
        'r0': x0[0],
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),
        'rho_surf': rho_surf_nom,
        'H': H_nom,
    }

    # ---- 10-dim mean vector: [r, theta, phi, V, gamma, psi, LD, beta, sigma, H] ----
    p0 = np.array([LD, beta, sigma, H_nom])
    m0_full = np.concatenate([x0, p0])

    # ---- 1-sigma uncertainties -- EDIT THESE to match your actual uncertainty budget ----
    state_1sig = np.array([
        10e3 / 3,  # r (altitude proxy), m
        np.radians(1 / 3),  # theta, rad
        np.radians(1 / 3),  # phi, rad
        1000 / 3,  # V, m/s
        np.radians(1 / 3),  # gamma, rad
        np.radians(1 / 3),  # psi, rad
    ])
    param_1sig = np.array([
        0.2 * LD,  # LD, ~2% 1-sigma
        0.2 * beta,  # beta, ~5% 1-sigma
        np.radians(20 / 3),  # sigma, 1 deg 1-sigma
        0.22 * H_nom,  # H, ~2% 1-sigma
    ])

    full_1sig = np.concatenate([state_1sig, param_1sig])
    init_cov = np.diag(full_1sig ** 2)  # replace with full off-diagonal cov if correlations are known

    proc_cov = np.zeros_like(init_cov)

    return ts, m0_full, params, init_cov, proc_cov



def uranus_inputs_KLE():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 10  # s
    Tf = 14*60  # s

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
    sigma = np.radians(45) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753) # [rad]
    # gamma0_I = np.radians(-16.3) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # Not using exponential atmosphere
    atmoParams['Uranus']['rho_surf'] = np.nan
    atmoParams['Uranus']['H'] = np.nan

    # If using KLE tag, need to set atmosphere parameters and d later
    atmoTag = 'kle'

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "IntAtmoNonDim_KLE", # EOMs with density
        "eomFunc": eoms3DOFIntAtmoNonDim_KLE,
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        }

    state_1sig = np.array([
        # (0.05 * h0) / 3,  # r (altitude proxy), m Original Run
        (0.1 * h0) / 3,  # r (altitude proxy), m
        np.radians(0.15 / 3),  # theta, rad
        np.radians(0.15 / 3),  # phi, rad
        # (0.05 * V0_R) / 3,  # V, m/s Original Run
        (0.1 * V0_R) / 3,  # V, m/s
        np.radians(0.15 / 3),  # gamma, rad
        np.radians(0.15 / 3),  # psi, rad
    ])

    init_cov = np.diag(state_1sig ** 2)
    proc_cov =  np.zeros_like(init_cov)

    return ts, x0, params, init_cov, proc_cov

def uranus_inputs_KLE_10p_cov():
    planetName = 'Uranus'
    R_eq = gravParams[planetName].get('R_eq')
    T0, dT = 0, 10  # s
    Tf = 14*60  # s

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
    sigma = np.radians(45) # [deg]
    # From 1b_get_fpa_corridor
    gamma0_I = np.radians(-10.5825108753) # [rad]
    # gamma0_I = np.radians(-16.3) # [rad]
    # Convert gamma and V from inertial to planet-relative
    V0_R, gamma0_R = inertial2planetRel(planetName, V0_I, gamma0_I, h0, psi0, phi0)

    # Not using exponential atmosphere
    atmoParams['Uranus']['rho_surf'] = np.nan
    atmoParams['Uranus']['H'] = np.nan

    # If using KLE tag, need to set atmosphere parameters and d later
    atmoTag = 'kle'

    x0 = np.array([R_eq + h0, theta0, phi0, V0_R, gamma0_R, psi0])
    ts = np.arange(T0, Tf, dT)

    params = {
        "LD": LD,
        "beta": beta,
        "sigma": sigma,
        "planetName": planetName,
        "atmoTag": atmoTag,
        "gdualFlag": False,
        "eomTag": "IntAtmoNonDim_KLE", # EOMs with density
        "eomFunc": eoms3DOFIntAtmoNonDim_KLE,
        "V0_I": V0_I,
        'r0': x0[0],
        # 'terminationMode': 'Time',  # 'Altitude' or 'Time'
        'ra_target': 550_000e3 + gravParams[planetName].get('R_eq'),  # From SciTECH 2024 paper, SciTECH 2025 paper has 200_000ee3 m apo alt
        }

    state_1sig = 0.1 * np.array([
        # (0.05 * h0) / 3,  # r (altitude proxy), m Original Run
        (0.1 * h0) / 3,  # r (altitude proxy), m
        np.radians(0.15 / 3),  # theta, rad
        np.radians(0.15 / 3),  # phi, rad
        # (0.05 * V0_R) / 3,  # V, m/s Original Run
        (0.1 * V0_R) / 3,  # V, m/s
        np.radians(0.15 / 3),  # gamma, rad
        np.radians(0.15 / 3),  # psi, rad
    ])

    init_cov = np.diag(state_1sig ** 2)
    proc_cov =  np.zeros_like(init_cov)

    return ts, x0, params, init_cov, proc_cov
