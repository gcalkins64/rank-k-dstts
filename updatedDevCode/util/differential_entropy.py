# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for differential entropy calculations

import numpy as np
import math
import sys
from scipy.integrate import solve_ivp
import copy
from .stt import A_tensor_general  # generic gdual Jacobian, any eomTag
from .params_planet import *

def lin_entropy_ode_general(t, H, means_nd, params, ii=0, dimTag=3):
    """dH/dt = trace(A(t)), A(t) = d(eom)/dx via A_tensor_general — works for
    any eomTag already wired into A_tensor_general, no per-EOM symbolic partials."""
    x_nd = means_nd[ii]
    A_first, _, _, _ = A_tensor_general(t, order=1, x=x_nd, params=params,
                                         eomTag=params['eomTag'], dimTag=dimTag)
    return np.array([np.trace(A_first)])

def get_differential_entropy(ts, mus_nd, Ps, params):
    # get linearized entropy over the trajectory
    ts_nd = ts / np.sqrt(gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get('g_surf'))
    H_lin_0 = 0.5*np.log(np.linalg.det(2*np.pi*np.e*Ps[0,:,:]))  # set initial guess to be nonlinear initial entropy
    print(H_lin_0)
    # H_lin_0 = 0
    H_lin = np.zeros(len(ts))
    H_lin[0] = H_lin_0
    for ii in range(len(ts)-1):
        sol = solve_ivp(lambda t, X: lin_entropy_ode_general(t, X, mus_nd, params, ii), y0 = np.array([0]), t_span=[ts_nd[ii], ts_nd[ii+1]], method='RK45', rtol=1e-7, atol=1e-8)
        H_lin[ii+1] = sol.y[-1,-1]
    H_lin = np.array(H_lin)

    # Get nonlinear entropy over the trajectory
    H_nonlin = []
    for ii in range(len(ts)):
        H_nonlin.append(0.5*np.log(np.linalg.det(2*np.pi*np.e*Ps[ii,:,:])))
    H_nonlin = np.array(H_nonlin)

    DE = 1- np.abs((H_nonlin-H_lin)/H_lin_0)

    return DE[1:]

def get_nonlin_entropy(P):
    tmp = np.linalg.det(2 * np.pi * np.e * P)
    H_nonlin = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * P))
    return H_nonlin

def get_instantaneous_DE(P_nd, H_lin_k, H0):
    # Get the DE at time k
    # Get current nonlinear entropy
    tmp = np.linalg.det(2 * np.pi * np.e * P_nd)
    # if tmp <= 0:
    #     raise ValueError('Determinant of covariance matrix is too small')
    H_nonlin_k = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * P_nd))
    # print(H_nonlin_k)
    DE = np.abs((H_nonlin_k - H_lin_k) / H0)
    return DE, H_nonlin_k


