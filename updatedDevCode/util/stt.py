# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses helper functions for computing stt / trajectory quantities

from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from .eoms_aerocapture import *
from .params_planet import gravParams
from pyaudi import gdual_double as gdual # type: ignore
from .nonDim import *
from opt_einsum import contract

def getNominalTraj(ts, x0, params):
    Xs = np.zeros((len(ts), len(x0)))
    Xs[0, :] = x0
    # print(x0)

    for ii, t in enumerate(tqdm(ts[:-1], desc="Nominal Propagation")):
        tspan = [t, ts[ii + 1]]
        # print(tspan, Xs[ii,:])
        sol = solve_ivp(lambda t, X: eoms3DOF(t, X, params), tspan, Xs[ii, :],'RK45', rtol=1e-13, atol=1e-18)

        Xs[ii + 1, :] = np.transpose(sol.y)[-1]

    return Xs

def getNominalTrajLnDens(ts, x0, params):
    Xs = np.zeros((len(ts), len(x0)))
    Xs[0, :] = x0
    # print(x0)

    for ii, t in enumerate((ts[:-1])):
        tspan = [t, ts[ii + 1]]
        # print(tspan, Xs[ii,:])
        sol = solve_ivp(lambda t, X: eoms3DOFLnDens(t, X, params), tspan, Xs[ii, :],'RK45', rtol=1e-13, atol=1e-18)

        Xs[ii + 1, :] = np.transpose(sol.y)[-1]

    return Xs

def getNominalTrajNonDim(ts, x0, params):
    Xs = np.zeros((len(ts), len(x0)))
    Xs[0, :] = x0
    Xs_nd = np.zeros((len(ts), len(x0)))
    Xs_nd[0, :] = convertToNonDim(x0, params)
    planetName = params['planetName']

    for ii, t in enumerate(tqdm(ts[:-1], desc="Non Dim Propagation")):
        tspan = [t, ts[ii + 1]]
        tspan_nd = [t / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf')),
                 ts[ii + 1] / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf'))]
        sol = solve_ivp(lambda t, X: eoms3DOFNonDim(t, X, params), tspan_nd, Xs_nd[ii, :],'RK45', rtol=1e-13, atol=1e-18)

        Xs_nd[ii + 1, :] = np.transpose(sol.y)[-1]
        Xs[ii + 1, :] = convertFromNonDim(Xs_nd[ii + 1, :], params)

    return Xs

def getNominalTrajNonDimLnDens(ts, x0, params):
    Xs = np.zeros((len(ts), len(x0)))
    Xs[0, :] = x0
    Xs_nd = np.zeros((len(ts), len(x0)))
    Xs_nd[0, :] = convertToNonDimDens(x0, params)
    planetName = params['planetName']

    for ii, t in enumerate(tqdm(ts[:-1], desc="Non Dim Propagation")):
        tspan = [t, ts[ii + 1]]
        tspan_nd = [t / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf')),
                 ts[ii + 1] / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf'))]
        sol = solve_ivp(lambda t, X: eoms3DOFLnDensNonDim(t, X, params), tspan_nd, Xs_nd[ii, :],'RK45', rtol=1e-13, atol=1e-18)

        Xs_nd[ii + 1, :] = np.transpose(sol.y)[-1]
        Xs[ii + 1, :] = convertFromNonDimDens(Xs_nd[ii + 1, :], params)

    return Xs

def getNominalTrajNonDimKLE(ts, x0, params):
    Xs = np.zeros((len(ts), len(x0)))
    Xs[0, :] = x0
    Xs_nd = np.zeros((len(ts), len(x0)))
    Xs_nd[0, :] = convertToNonDim(x0, params)
    planetName = params['planetName']

    for ii, t in enumerate(tqdm(ts[:-1], desc="Non Dim Propagation")):
        tspan_nd = [t / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf')), ts[ii + 1] / np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf'))]
        sol = solve_ivp(lambda t, X: eoms3DOFIntAtmoNonDim_KLE(t, X, params), tspan_nd, Xs_nd[ii, :], 'RK45', rtol=1e-13, atol=1e-18)

        Xs_nd[ii + 1, :] = np.transpose(sol.y)[-1]
        Xs[ii + 1, :] = convertFromNonDim(Xs_nd[ii + 1, :], params)

    return Xs

def A_tensor_general(t, order, x, params, eomTag, dimTag, t_ref=None, X_ref=None):
    stateSize = len(x)
    A_first = np.zeros((stateSize, stateSize))
    A_second = np.zeros((stateSize, stateSize, stateSize))
    A_third = np.zeros((stateSize, stateSize, stateSize, stateSize))
    A_fourth = np.zeros((stateSize, stateSize, stateSize, stateSize, stateSize))

    # Interpolate state to time t if needed
    if t_ref is not None:
        # Interpolate each component separately
        x = np.array([np.interp(t, t_ref, X_ref[:, i]) for i in range(X_ref.shape[1])])

    # set up autodifferentiation stuff with gduals
    g_order = order+1
    d_KLE = params.get('d_KLE', 0)
    if dimTag == 2:
        base_names = ["V", "gamma", "z", "theta", "rho"]
    elif dimTag == 3:
        if eomTag == '3DOFNonDim':
            base_names = ["r", "theta", "phi", "V", "gamma", "psi", "rho"]
        elif eomTag in ('3DOFLnNonDim',
                        'eoms3DOFLnDensNonDim_CONSERVATIVE_eval',
                        'eoms3DOFLnDensNonDim_AERODYNAMIC_eval'):
            base_names = ["r", "theta", "phi", "V", "gamma", "psi", "ln_rho"]
        elif eomTag == 'IntAtmoNonDim_KLE':
            base_names = ["r", "theta", "phi", "V", "gamma", "psi"]
        else:
            base_names = ["r", "theta", "phi", "V", "gamma", "psi"]

    kle_names = [f"Y{i}" for i in range(d_KLE)]
    dual_str = base_names + kle_names
    X_i_g = [gdual(x[k], dual_str[k], g_order) for k in range(len(dual_str))]

    params['gdualFlag'] = True
    if eomTag == '3DOF':
        F = eoms3DOFDens(t, X_i_g, params)
    elif eomTag == '3DOFNonDim':
        F = eoms3DOFNonDimDens(t, X_i_g, params)
    elif eomTag == '3DOFLn':
        F = eoms3DOFLnDens(t, X_i_g, params)
    elif eomTag == '3DOFLnNonDim':
        F = eoms3DOFLnDensNonDim(t, X_i_g, params)
    elif eomTag == '3DOFNonDimNoDens':
        F = eoms3DOFNonDim(t, X_i_g, params)
    elif eomTag == '3DOFNoDens':
        F = eoms3DOF(t, X_i_g, params)
    elif eomTag == 'eoms3DOFLnDensNonDim_CONSERVATIVE_eval':
        F = eoms3DOFLnDensNonDim_CONSERVATIVE_eval(t, X_i_g, params)
    elif eomTag == 'eoms3DOFLnDensNonDim_AERODYNAMIC_eval':
        F = eoms3DOFLnDensNonDim_AERODYNAMIC_eval(t, X_i_g, params)
    elif eomTag == 'IntAtmoNonDim_KLE':
        F = eoms3DOFIntAtmoNonDim_KLE(t, X_i_g, params)


    for i in range(stateSize):
        for a in range(stateSize):
            A_first_gdual = F[i].partial(dual_str[a])
            A_first[i, a] = A_first_gdual.constant_cf
            if order >= 2:
                for b in range(stateSize):
                    A_second_gdual = A_first_gdual.partial(dual_str[b])
                    A_second[i, a, b] = A_second_gdual.constant_cf
                    if order >= 3:
                        for c in range(stateSize):
                            A_third_gdual = A_second_gdual.partial(dual_str[c])
                            A_third[i, a, b, c] = A_third_gdual.constant_cf
                            if order >= 4:
                                for d in range(stateSize):
                                    A_fourth_gdual = A_third_gdual.partial(dual_str[d])
                                    A_fourth[i, a, b, c, d] = A_fourth_gdual.constant_cf

    return A_first, A_second, A_third, A_fourth

def aerocaptureSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, t_ref, X_ref):
    # print(t) # TODO
    # Unpack State and Phi
    if order >= 1:
        second_start = stateSize + stateSize ** 2
        phi_first = Phi_stack[stateSize:second_start].reshape((stateSize, stateSize))
        third_start = second_start + stateSize ** 3
        if order >= 2:
            phi_second = Phi_stack[second_start:third_start].reshape((stateSize, stateSize, stateSize))
            fourth_start = third_start + stateSize ** 4
            if order >= 3:
                phi_third = Phi_stack[third_start:fourth_start].reshape((stateSize, stateSize, stateSize, stateSize))
                if order >= 4:
                    phi_fourth = Phi_stack[fourth_start:].reshape((stateSize, stateSize, stateSize, stateSize, stateSize))

    x = Phi_stack[0:stateSize]
    if eomTag == '3DOF':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFDens(t, x, params)
    if eomTag == '3DOFNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        params['gdualFlag'] = False
        xdot = eoms3DOFNonDimDens(t, x, params)
    if eomTag == '3DOFLn':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDens(t, x, params)
    if eomTag == '3DOFLnNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDensNonDim(t, x, params)
    if eomTag == '3DOFNonDimNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFNonDim(t, x, params)
    if eomTag == '3DOFNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOF(t, x, params)
    if eomTag == 'eoms3DOFLnDensNonDim_CONSERVATIVE_eval':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3, t_ref, X_ref)
        xdot = eoms3DOFLnDensNonDim_CONSERVATIVE_eval(t, x, params)
    if eomTag == 'eoms3DOFLnDensNonDim_AERODYNAMIC_eval':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3, t_ref, X_ref)
        xdot = eoms3DOFLnDensNonDim_AERODYNAMIC_eval(t, x, params)
    elif eomTag == 'IntAtmoNonDim_KLE':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3, t_ref, X_ref)
        params['gdualFlag'] = False
        xdot = eoms3DOFIntAtmoNonDim_KLE(t, x, params)

    # Get STT ODE
    if order >= 1:
        phidot_first = np.zeros((stateSize, stateSize))
        phidot_first += np.einsum('ij,ja->ia', A_first, phi_first)
        phidot = np.hstack((xdot, phidot_first.reshape(-1)))
    if order >= 2:
        phidot_second = np.zeros((stateSize, stateSize, stateSize))
        phidot_second += np.einsum('ij,jab->iab', A_first, phi_second)
        phidot_second += np.einsum('ijk,ja,kb->iab', A_second, phi_first, phi_first, optimize=True)
        phidot = np.hstack([xdot, phidot_first.reshape(-1), phidot_second.reshape(-1)])
    if order >= 3:
        phidot_third = np.zeros((stateSize, stateSize, stateSize, stateSize))
        phidot_third += np.einsum('ij,jabc->iabc', A_first, phi_third)
        phidot_third += np.einsum('ijk,ja,kbc->iabc', A_second, phi_first, phi_second)
        phidot_third += np.einsum('ijk,jab,kc->iabc', A_second, phi_second, phi_first)
        phidot_third += np.einsum('ijk,jac,kb->iabc', A_second, phi_second, phi_first)
        phidot_third += np.einsum('ijkl,ja,kb,lc->iabc', A_third, phi_first, phi_first, phi_first)
        phidot = np.hstack([xdot, phidot_first.reshape(-1), phidot_second.reshape(-1), phidot_third.reshape(-1)])
    if order >= 4:
        phidot_fourth = np.zeros((stateSize, stateSize, stateSize, stateSize, stateSize))
        phidot_fourth += np.einsum('ij,jabcd->iabcd', A_first, phi_fourth)
        phidot_fourth += np.einsum('ijk,jabc,kd->iabcd', A_second, phi_third, phi_first)
        phidot_fourth += np.einsum('ijk,jabd,kc->iabcd', A_second, phi_third, phi_first)
        phidot_fourth += np.einsum('ijk,jacd,kb->iabcd', A_second, phi_third, phi_first)
        phidot_fourth += np.einsum('ijk,jab,kcd->iabcd', A_second, phi_second, phi_second)
        phidot_fourth += np.einsum('ijk,jac,kbd->iabcd', A_second, phi_second, phi_second)
        phidot_fourth += np.einsum('ijk,jad,kbc->iabcd', A_second, phi_second, phi_second)
        phidot_fourth += np.einsum('ijk,ja,kbcd->iabcd', A_second, phi_first, phi_third)
        phidot_fourth += np.einsum('ijkl,jab,kc,ld->iabcd', A_third, phi_second, phi_first, phi_first)
        phidot_fourth += np.einsum('ijkl,jac,kb,ld->iabcd', A_third, phi_second, phi_first, phi_first)
        phidot_fourth += np.einsum('ijkl,jad,kb,lc->iabcd', A_third, phi_second, phi_first, phi_first)
        phidot_fourth += np.einsum('ijkl,ja,kbc,ld->iabcd', A_third, phi_first, phi_second, phi_first)
        phidot_fourth += np.einsum('ijkl,ja,kbd,lc->iabcd', A_third, phi_first, phi_second, phi_first)
        phidot_fourth += np.einsum('ijkl,ja,kb,lcd->iabcd', A_third, phi_first, phi_first, phi_second)
        phidot_fourth += np.einsum('ijklm,ja,kb,lc,md->iabcd', A_fourth, phi_first, phi_first, phi_first, phi_first)
        phidot = np.hstack([xdot, phidot_first.reshape(-1), phidot_second.reshape(-1), phidot_third.reshape(-1), phidot_fourth.reshape(-1)])

    return phidot

def getSTTs_general(ts_list, x0, order, params, eomTag, mode='t0tk', t_ref=None, X_ref=None, X_stt_ref_nd=None):
    """
    Pass in initial DIMENSIONAL state and nondimensionalize here if applicable.
    Output is nondimensionalized STTs and state for nondimensional case (need to nondimensionalize the state to use them)
            or dimensionalized STTs and state for dimensional case

    Mode can be t0tk, tktkp1, or tktf depending on time domain for mapping
    """
    if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim' or eomTag == 'eoms3DOFLnDensNonDim_CONSERVATIVE_eval' or eomTag == 'eoms3DOFLnDensNonDim_AERODYNAMIC_eval':
        # Non dimensionalize the state
        x0_nd = convertToNonDimDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '2DNonDim':
        x0_nd = convertToNonDim2DDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
        x0_nd = convertToNonDim(x0, params)
        x0_setup = x0_nd
    else: # dimensional
        x0_setup = x0

    stateSize = len(x0)
    phi_0_first = np.eye(stateSize)
    phi_0_second = np.zeros((stateSize, stateSize, stateSize))
    phi_0_third= np.zeros((stateSize, stateSize, stateSize, stateSize))
    phi_0_fourth = np.zeros((stateSize, stateSize, stateSize, stateSize, stateSize))

    # Stack State and Phi together to pass into integrator
    if order >= 1:
        Phi_stack = np.hstack((x0_setup, phi_0_first.flatten()))
        STTs = np.zeros((len(ts_list)+1, stateSize**2))
    if order >= 2:
        Phi_stack = np.hstack((x0_setup, phi_0_first.flatten(), phi_0_second.flatten()))
        STTs = np.zeros((len(ts_list)+1, stateSize**2 + stateSize ** 3))
    if order >= 3:
        Phi_stack = np.hstack((x0_setup, phi_0_first.flatten(), phi_0_second.flatten(), phi_0_third.flatten()))
        STTs = np.zeros((len(ts_list)+1, stateSize**2 + stateSize ** 3 + stateSize ** 4))
    if order >= 4:
        Phi_stack = np.hstack((x0_setup, phi_0_first.flatten(), phi_0_second.flatten(), phi_0_third.flatten(), phi_0_fourth.flatten()))
        STTs = np.zeros((len(ts_list), stateSize**2 + stateSize ** 3 + stateSize ** 4 + stateSize ** 5))

    if 'NonDim' in eomTag:
        Xs_nd = np.zeros((len(ts_list)+1, stateSize))
        Xs_nd[0, :] = x0_nd

    STTs[0, :] = Phi_stack[stateSize:]
    Xs = np.zeros((len(ts_list)+1, stateSize))
    Xs[0, :] = x0  # Always dimensional

    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')

    # for ii, ts in enumerate(tqdm(ts_list, desc=f'getSTTs_{eomTag}_{mode}')):
    for ii, ts in enumerate(ts_list):
        if 'NonDim' in eomTag:
            tspan = [ts[0] / np.sqrt(Rp / g0),
                    ts[1] / np.sqrt(Rp / g0)]
            if mode == 'tktkp1':
                x = Xs_nd[ii, :]
            if mode == 'tktf':
                x = X_stt_ref_nd[ii, :]
            if mode == 't0tk':
                x = Xs_nd[0, :]
        else: # dimensional
            tspan = ts
            if mode == 'tktkp1':
                x = Xs[ii, :]
            if mode == 'tktf': # TODO
                x = X_stt_ref_nd[ii, :]
            if mode == 't0tk':
                x = Xs[0, :]

        if order >= 1:
            Phi_stack = np.hstack((x, phi_0_first.flatten()))
        if order >= 2:
            Phi_stack = np.hstack((x, phi_0_first.flatten(), phi_0_second.flatten()))
        if order >= 3:
            Phi_stack = np.hstack((x, phi_0_first.flatten(), phi_0_second.flatten(), phi_0_third.flatten()))
        if order >= 4:
            Phi_stack = np.hstack((x, phi_0_first.flatten(), phi_0_second.flatten(), phi_0_third.flatten(), phi_0_fourth.flatten()))

        # print('Integrating over tspan ', tspan)
        # X_stack_sol = solve_ivp(lambda t, Phi_stack: aerocaptureSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, t_ref, X_ref), tspan, Phi_stack, 'DOP853', rtol=1e-13, atol=1e-13)
        X_stack_sol = solve_ivp(lambda t, Phi_stack: aerocaptureSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, t_ref, X_ref), tspan, Phi_stack, 'DOP853', rtol=1e-8, atol=1e-8)
        Phi_stack = X_stack_sol.y[:,-1]
        STTs[ii+1, :] = Phi_stack[stateSize:]
        if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim':
            if mode != 'tktf':
                Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
                Xs[ii + 1, :] = convertFromNonDimDens(Phi_stack[0:stateSize], params)
            else:
                Xs_nd[ii+1, :] = X_stt_ref_nd[ii+1, :]
                Xs[ii + 1, :] = convertFromNonDimDens(X_stt_ref_nd[ii+1, :], params)
        elif eomTag == '2DNonDim':
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim2DDens(Phi_stack[0:stateSize], params)
        elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim(Phi_stack[0:stateSize], params)
        else:  # dimensional
            Xs[ii + 1, :] = Phi_stack[0:stateSize]

    if 'NonDim' in eomTag:
        return Xs_nd, Xs, STTs  # pass out nondimensionalized nominal state and nondimensionalized STTs
    else:  # dimensional
        return Xs, STTs

def aerocaptureDSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, R, t_ref, X_ref):
    # Reshape DSTTs appropriately
    ndim = np.ndim(R)
    if ndim == 1: # Only a single direction passed in, use same R for all orders
        R_2 = R
        R_3 = R
    else:
        R_2 = R[0,:]
        R_3 = R[1,:]

    l = 1 # Latent dimension, hardcoded as 1 for now TODO
    if order >= 1:
        second_start = stateSize + stateSize ** 2
        phi_first = Phi_stack[stateSize:second_start].reshape((stateSize, stateSize))
    if order >= 2:  # Our ODEs for the higher order DSTTs are just columns corresponding to the 6 DEs for each higher order DSTT
        third_start = second_start + l*stateSize
        DSTT_second = Phi_stack[second_start:third_start].reshape(stateSize, l, l)

    if order >= 3:
        fourth_start = third_start + l*stateSize
        DSTT_third = Phi_stack[second_start:third_start].reshape(stateSize, l, l, l)
    if order >= 4:
        DSTT_fourth = Phi_stack[fourth_start:]

    x = Phi_stack[0:stateSize]
    if eomTag == '3DOF':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFDens(t, x, params)
    if eomTag == '3DOFNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFNonDimDens(t, x, params)
    if eomTag == '3DOFLn':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDens(t, x, params)
    if eomTag == '3DOFLnNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDensNonDim(t, x, params)
    if eomTag == '3DOFNonDimNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFNonDim(t, x, params)
    if eomTag == '3DOFNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOF(t, x, params)
    elif eomTag == 'IntAtmoNonDim_KLE':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3, t_ref, X_ref)
        params['gdualFlag'] = False
        xdot = eoms3DOFIntAtmoNonDim_KLE(t, x, params)

    # Get STM ODE
    if order >= 1:
        phidot_first = np.zeros((stateSize, stateSize))
        phidot_first += np.einsum('ij,ja->ia', A_first, phi_first)
        phidot = np.hstack((xdot, phidot_first.reshape(-1)))

    # NEED A DIFFERENT DSTM FOR EACH ORDER!!!!!!!!!
    if order >= 2:  # Second is modified by the DSTT truncation
        DSTM_2 = phi_first @ R_2
        DSTM_2 = DSTM_2.reshape(stateSize, l)
        DSTTdot_second = np.zeros(stateSize).reshape(stateSize, l, l)

        DSTTdot_second += np.einsum('ia,ajk->ijk', A_first, DSTT_second)
        DSTTdot_second += np.einsum('iab,aj,bk->ijk', A_second, DSTM_2, DSTM_2, optimize=True)

        DSTTdot_second = np.squeeze(DSTTdot_second)

        phidot = np.hstack([phidot, DSTTdot_second])

    # Third order
    if order >= 3:
        DSTTdot_third = np.zeros(stateSize).reshape(stateSize, l, l, l)
        DSTM_3 = phi_first @ R_3
        DSTM_3 = DSTM_3.reshape(stateSize, l)

        DSTTdot_third += np.einsum('ia,ajkl->ijkl', A_first, DSTT_third)
        DSTTdot_third += np.einsum('iab,aj,bkl->ijkl', A_second, DSTM_3, DSTT_second, optimize=True)
        DSTTdot_third += np.einsum('iab,ajk,bl->ijkl', A_second, DSTT_second, DSTM_3, optimize=True)
        DSTTdot_third += np.einsum('iab,ajl,bk->ijkl', A_second, DSTT_second, DSTM_3, optimize=True)
        DSTTdot_third += np.einsum('iabc,aj,bk,cl->ijkl', A_third, DSTM_3, DSTM_3, DSTM_3, optimize=True)

        DSTTdot_third = np.squeeze(DSTTdot_third)

        phidot = np.hstack([phidot, DSTTdot_third])

    # Fourth order
    if order >= 4:
        DSTTdot_fourth = np.zeros(stateSize)
        # 7/2/2026 pretty sure indexing is wrong, commenting out for now since we will not go to fourth order
        # DSTTdot_fourth += np.einsum('ij,j', A_first, DSTT_fourth)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTM, DSTT_third, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTT_second, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTT_second, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTM, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijklm,j,k,l,m->i', A_fourth, DSTM, DSTM, DSTM, DSTM, optimize=True)

        phidot = np.hstack([phidot, DSTTdot_fourth])
        raise NotImplementedError("Fourth Order DSTTs not implemented")
    return phidot

def getDSTTs_general(ts_list, x0, Rs, order, params, eomTag, latentDim = 1, mode='t0tk', Xs_int=None, t_ref=None, X_ref=None):
    raise NotImplementedError("Need to implement larger latent dimension")
    # Pass in as many R's as times
    stateSize = len(x0)
    # from DSTT_integration.py
    if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim' or eomTag == 'eoms3DOFLnDensNonDim_CONSERVATIVE_eval' or eomTag == 'eoms3DOFLnDensNonDim_AERODYNAMIC_eval':
        # Non dimensionalize the state
        x0_nd = convertToNonDimDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '2DNonDim':
        x0_nd = convertToNonDim2DDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
        x0_nd = convertToNonDim(x0, params)
        x0_setup = x0_nd
    else:  # dimensional
        x0_setup = x0

    # Formulate initial STM and DSTT columns
    STM_0 = np.eye(stateSize)
    DSTT_0_second = np.zeros(stateSize)
    DSTT_0_third = np.zeros(stateSize)
    DSTT_0_fourth = np.zeros(stateSize)

    if order >= 1:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize ** 2))
    if order >= 2:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize + stateSize ** 2))
    if order >= 3:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, 2 * stateSize + stateSize ** 2))
    if order >= 4:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten(), DSTT_0_fourth.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, 3 * stateSize + stateSize ** 2))

    if 'NonDim' in eomTag:
        Xs_nd = np.zeros((len(ts_list)+1, stateSize))
        Xs_nd[0, :] = x0_nd

    DSTTs[0, :] = Phi_stack[stateSize:]
    Xs = np.zeros((len(ts_list)+1, stateSize))
    Xs[0, :] = x0  # Always dimensional

    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')

    # for ii, ts in enumerate(tqdm(ts_list, desc=f'getDSTTs_{eomTag}')):
    for ii, ts in enumerate(ts_list):
        if 'NonDim' in eomTag:
            tspan = [ts[0] / np.sqrt(Rp / g0),
                    ts[1] / np.sqrt(Rp / g0)]
            if mode == 'tktkp1':
                x = Xs_nd[ii, :]
            if mode == 'tktf':
                x = Xs_int[ii, :]
            if mode == 't0tk':
                x = Xs_nd[0, :]
        else: # dimensional
            tspan = ts
            if mode == 'tktkp1':
                x = Xs[ii, :]
            if mode == 'tktf': # TODO
                x = Xs_int[ii, :]
            if mode == 't0tk':
                x = Xs[0, :]
        # Debug check dx
        if Xs_int is not None:
            dx = Xs[ii, :] - Xs_int[ii, :]

        if order >= 1:
            Phi_stack = np.hstack((x, STM_0.flatten()))
        if order >= 2:
            Phi_stack = np.hstack((x, STM_0.flatten(), DSTT_0_second.flatten()))
        if order >= 3:
            Phi_stack = np.hstack((x, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten()))
        if order >= 4:
            Phi_stack = np.hstack(
                (x, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten(), DSTT_0_fourth.flatten()))

        R = np.squeeze(Rs[ii])
        X_stack_sol = solve_ivp(
            lambda t, Phi_stack: aerocaptureDSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, R, t_ref, X_ref), tspan, Phi_stack, 'DOP853', rtol=1e-13, atol=1e-13)

        Phi_stack = X_stack_sol.y[:, -1]
        DSTTs[ii + 1, :] = Phi_stack[stateSize:]
        if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim':
            if mode != 'tktf':
                Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
                Xs[ii + 1, :] = convertFromNonDimDens(Phi_stack[0:stateSize], params)
            else:
                Xs_nd[ii+1, :] = Xs_int[ii+1, :]
                Xs[ii + 1, :] = convertFromNonDimDens(Xs_int[ii+1, :], params)
        elif eomTag == '2DNonDim':
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim2DDens(Phi_stack[0:stateSize], params)
        elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
            # Non dimensionalize the state
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim(Phi_stack[0:stateSize], params)
        else:  # dimensional
            Xs[ii + 1, :] = Phi_stack[0:stateSize]

        # state_check = X_stack_state.y[:, -1]
        # dx_check = Xs[ii + 1, :] - state_check
        # blah = 42

    if 'NonDim' in eomTag:
        return Xs_nd, Xs, DSTTs  # pass out nondimensionalized nominal state and nondimensionalized STTs
    else:  # dimensional
        return Xs, DSTTs

def aerocapturexDSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, R, latentDim, t_ref, X_ref):
    # Reshape DSTTs appropriately
    R_2 = R
    R_3 = R  # TODO

    l = latentDim # Latent dimension, hardcoded as 1 for now
    if order >= 1:
        second_start = stateSize + stateSize ** 2
        phi_first = Phi_stack[stateSize:second_start].reshape((stateSize, stateSize))
    if order >= 2:  # Our ODEs for the higher order DSTTs are just columns corresponding to the 6 DEs for each higher order DSTT
        third_start = second_start + l*l*stateSize
        DSTT_second = Phi_stack[second_start:third_start].reshape(stateSize, l, l)

    if order >= 3:
        fourth_start = third_start + l*l*l*stateSize
        DSTT_third = Phi_stack[second_start:third_start].reshape(stateSize, l, l, l)
    if order >= 4:
        DSTT_fourth = Phi_stack[fourth_start:]

    x = Phi_stack[0:stateSize]
    if eomTag == '3DOF':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFDens(t, x, params)
    if eomTag == '3DOFNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFNonDimDens(t, x, params)
    if eomTag == '3DOFLn':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDens(t, x, params)
    if eomTag == '3DOFLnNonDim':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFLnDensNonDim(t, x, params)
    if eomTag == '3DOFNonDimNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOFNonDim(t, x, params)
    if eomTag == '3DOFNoDens':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3)
        xdot = eoms3DOF(t, x, params)
    elif eomTag == 'IntAtmoNonDim_KLE':
        A_first, A_second, A_third, A_fourth = A_tensor_general(t, order, x, params, eomTag, 3, t_ref, X_ref)
        params['gdualFlag'] = False
        xdot = eoms3DOFIntAtmoNonDim_KLE(t, x, params)

    # Get STM ODE
    if order >= 1:
        phidot_first = np.zeros((stateSize, stateSize))
        phidot_first += np.einsum('ij,ja->ia', A_first, phi_first)
        phidot = np.hstack((xdot, phidot_first.reshape(-1)))

    # NEED A DIFFERENT DSTM FOR EACH ORDER!!!!!!!!!
    if order >= 2:  # Second is modified by the DSTT truncation
        DSTM_2 = np.einsum('ij,kj->ik', phi_first, R_2)
        DSTM_2 = DSTM_2.reshape(stateSize, l)
        DSTTdot_second = np.zeros(stateSize*l*l).reshape(stateSize, l, l)

        DSTTdot_second += np.einsum('ia,ajk->ijk', A_first, DSTT_second)
        DSTTdot_second += np.einsum('iab,aj,bk->ijk', A_second, DSTM_2, DSTM_2, optimize=True)

        DSTTdot_second = np.squeeze(DSTTdot_second)

        phidot = np.hstack([phidot, DSTTdot_second.reshape(-1)])

    # Third order
    if order >= 3:
        DSTTdot_third = np.zeros(stateSize).reshape(stateSize, l, l, l)
        DSTM_3 = phi_first @ R_3
        DSTM_3 = DSTM_3.reshape(stateSize, l)

        DSTTdot_third += np.einsum('ia,ajkl->ijkl', A_first, DSTT_third)
        DSTTdot_third += np.einsum('iab,aj,bkl->ijkl', A_second, DSTM_3, DSTT_second, optimize=True)
        DSTTdot_third += np.einsum('iab,ajk,bl->ijkl', A_second, DSTT_second, DSTM_3, optimize=True)
        DSTTdot_third += np.einsum('iab,ajl,bk->ijkl', A_second, DSTT_second, DSTM_3, optimize=True)
        DSTTdot_third += np.einsum('iabc,aj,bk,cl->ijkl', A_third, DSTM_3, DSTM_3, DSTM_3, optimize=True)

        DSTTdot_third = np.squeeze(DSTTdot_third)

        phidot = np.hstack([phidot, DSTTdot_third])

    # Fourth order
    if order >= 4:
        DSTTdot_fourth = np.zeros(stateSize)
        # 7/2/2026 pretty sure indexing is wrong, commenting out for now since we will not go to fourth order
        # DSTTdot_fourth += np.einsum('ij,j', A_first, DSTT_fourth)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_third, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTT_second, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijk,j,k->i', A_second, DSTM, DSTT_third, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTT_second, DSTM, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTT_second, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTT_second, DSTM, optimize=True)
        # DSTTdot_fourth += np.einsum('ijkl,j,k,l->i', A_third, DSTM, DSTM, DSTT_second, optimize=True)
        # DSTTdot_fourth += np.einsum('ijklm,j,k,l,m->i', A_fourth, DSTM, DSTM, DSTM, DSTM, optimize=True)

        phidot = np.hstack([phidot, DSTTdot_fourth])
        raise NotImplementedError("Fourth Order DSTTs not implemented")
    return phidot

def getxDSTTs_general(ts_list, x0, Rs, order, params, eomTag, latentDim=1, mode='t0tk', Xs_int=None, t_ref=None, X_ref=None):
    # Pass in as many R's as times
    stateSize = len(x0)
    # from DSTT_integration.py
    if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim' or eomTag == 'eoms3DOFLnDensNonDim_CONSERVATIVE_eval' or eomTag == 'eoms3DOFLnDensNonDim_AERODYNAMIC_eval':
        # Non dimensionalize the state
        x0_nd = convertToNonDimDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '2DNonDim':
        x0_nd = convertToNonDim2DDens(x0, params)
        x0_setup = x0_nd
    elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
        x0_nd = convertToNonDim(x0, params)
        x0_setup = x0_nd
    else:  # dimensional
        x0_setup = x0

    # Formulate initial STM and DSTT columns
    STM_0 = np.eye(stateSize)
    DSTT_0_second = np.zeros(stateSize*latentDim*latentDim)
    DSTT_0_third = np.zeros(stateSize*latentDim*latentDim*latentDim)
    DSTT_0_fourth = np.zeros(stateSize*latentDim*latentDim*latentDim*latentDim)

    if order >= 1:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize ** 2))
    if order >= 2:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize*latentDim*latentDim + stateSize ** 2))
    if order >= 3:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize*latentDim*latentDim + stateSize*latentDim*latentDim*latentDim + stateSize ** 2))
    if order >= 4:
        Phi_stack = np.hstack((x0_setup, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten(), DSTT_0_fourth.flatten()))
        DSTTs = np.zeros((len(ts_list)+1, stateSize*latentDim*latentDim + stateSize*latentDim*latentDim*latentDim + stateSize*latentDim*latentDim*latentDim*latentDim + stateSize ** 2))

    if 'NonDim' in eomTag:
        Xs_nd = np.zeros((len(ts_list)+1, stateSize))
        Xs_nd[0, :] = x0_nd

    DSTTs[0, :] = Phi_stack[stateSize:]
    Xs = np.zeros((len(ts_list)+1, stateSize))
    Xs[0, :] = x0  # Always dimensional

    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')

    # for ii, ts in enumerate(tqdm(ts_list, desc=f'getDSTTs_{eomTag}')):
    for ii, ts in enumerate(ts_list):
        if 'NonDim' in eomTag:
            tspan = [ts[0] / np.sqrt(Rp / g0),
                    ts[1] / np.sqrt(Rp / g0)]
            if mode == 'tktkp1':
                x = Xs_nd[ii, :]
            if mode == 'tktf':
                x = Xs_int[ii, :]
            if mode == 't0tk':
                x = Xs_nd[0, :]
        else: # dimensional
            tspan = ts
            if mode == 'tktkp1':
                x = Xs[ii, :]
            if mode == 'tktf': # TODO
                x = Xs_int[ii, :]
            if mode == 't0tk':
                x = Xs[0, :]
        # Debug check dx
        if Xs_int is not None:
            dx = Xs[ii, :] - Xs_int[ii, :]

        if order >= 1:
            Phi_stack = np.hstack((x, STM_0.flatten()))
        if order >= 2:
            Phi_stack = np.hstack((x, STM_0.flatten(), DSTT_0_second.flatten()))
        if order >= 3:
            Phi_stack = np.hstack((x, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten()))
        if order >= 4:
            Phi_stack = np.hstack(
                (x, STM_0.flatten(), DSTT_0_second.flatten(), DSTT_0_third.flatten(), DSTT_0_fourth.flatten()))

        R = Rs[ii]
        X_stack_sol = solve_ivp(
            lambda t, Phi_stack: aerocapturexDSTTsODE_general(t, Phi_stack, order, stateSize, params, eomTag, R, latentDim, t_ref, X_ref), tspan, Phi_stack, 'DOP853', rtol=1e-13, atol=1e-13)

        Phi_stack = X_stack_sol.y[:, -1]
        DSTTs[ii + 1, :] = Phi_stack[stateSize:]
        if eomTag == '3DOFNonDim' or eomTag == '3DOFLnNonDim':
            if mode != 'tktf':
                Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
                Xs[ii + 1, :] = convertFromNonDimDens(Phi_stack[0:stateSize], params)
            else:
                Xs_nd[ii+1, :] = Xs_int[ii+1, :]
                Xs[ii + 1, :] = convertFromNonDimDens(Xs_int[ii+1, :], params)
        elif eomTag == '2DNonDim':
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim2DDens(Phi_stack[0:stateSize], params)
        elif eomTag == '3DOFNonDimNoDens' or eomTag == 'IntAtmoNonDim_KLE':
            # Non dimensionalize the state
            Xs_nd[ii + 1, :] = Phi_stack[0:stateSize]
            Xs[ii + 1, :] = convertFromNonDim(Phi_stack[0:stateSize], params)
        else:  # dimensional
            Xs[ii + 1, :] = Phi_stack[0:stateSize]

        # state_check = X_stack_state.y[:, -1]
        # dx_check = Xs[ii + 1, :] - state_check
        # blah = 42

    if 'NonDim' in eomTag:
        return Xs_nd, Xs, DSTTs  # pass out nondimensionalized nominal state and nondimensionalized STTs
    else:  # dimensional
        return Xs, DSTTs



def STTstoDSTTs(ts, STTs, Rs, order, stateSize):
    # Rs is a times x order-1 x state size matrix
    # from stt_library/dstt_filter/STT_HNEKF.py
    # Eqns 13-16 from Spencer's paper
    if order >= 1:
        DSTTs = np.zeros((len(ts), stateSize**2))
    if order >= 2:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize))
    if order >= 3:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize*2))
    if order >= 4:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize*3))

    # for ii, t in enumerate(tqdm(ts, desc='computeDSTTs')):
    for ii, t in enumerate(ts):
        STT_list = reshape_STT(STTs[ii, :], order, stateSize)

        DSTT_first = STT_list[0].flatten()  # need to keep STM UNROTATED to use eqn 23 of DSTT paper
        DSTT_row = DSTT_first
        if order >= 2:
            R2 = Rs[ii,0,:][np.newaxis].T  # Get second order direction
            DSTT_second = np.einsum('ijk,ja,kb->iab', STT_list[1], R2, R2, optimize=True).flatten()
            DSTT_row = np.concatenate((DSTT_row, DSTT_second))
        if order >= 3:
            R3 = Rs[ii,1,:][np.newaxis].T  # Get third order direction
            DSTT_third = np.einsum('ijkl,ja,kb,lc->iabc', STT_list[2], R3, R3, R3, optimize=True).flatten()
            DSTT_row = np.concatenate((DSTT_row, DSTT_third))
        if order >= 4:
            R4 = Rs[ii,2,:][np.newaxis].T
            DSTT_fourth = np.einsum('ijklm,ja,kb,lc,md->iabcd', STT_list[3], R4, R4, R4, R4, optimize=True).flatten()
            DSTT_row = np.concatenate((DSTT_row, DSTT_fourth))
        DSTTs[ii, :] = np.array(DSTT_row)

    return DSTTs

def STTstoDSTTsLargerLatent(ts, STTs, Rs, order, stateSize):
    # R is (times, order - 1, latentDim, state_size) for larger latent dimension.
    # Eqns 13-15 from Spencer's paper
    tlen, _, latent_dim, stateSize= np.shape(Rs)

    if order >= 1:
        DSTTs = np.zeros((len(ts), stateSize**2))
    if order >= 2:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize*latent_dim*latent_dim))
    if order >= 3:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize*latent_dim*latent_dim + stateSize*latent_dim*latent_dim*latent_dim))
    if order >= 4:
        DSTTs = np.zeros((len(ts), stateSize**2 + stateSize*latent_dim*latent_dim*3))

    for ii, t in enumerate(tqdm(ts, desc='computeDSTTs')):
        STT_list = reshape_STT(STTs[ii, :], order, stateSize)

        DSTT_first = STT_list[0].flatten()  # need to keep STM UNROTATED to use eqn 23 of DSTT paper
        DSTT_row = DSTT_first
        if order >= 2:
            R2 = Rs[ii, 0, :, :]
            DSTT_second = np.einsum('ijk,aj,bk->iab', STT_list[1], R2, R2, optimize=True).flatten()
            DSTT_row = np.concatenate((DSTT_row, DSTT_second))
        if order >= 3:
            R3 = Rs[ii, 1, :, :]
            DSTT_third = np.einsum('ijkl,aj,bk,cl->iabc', STT_list[2], R3, R3, R3, optimize=True).flatten()
            DSTT_row = np.concatenate((DSTT_row, DSTT_third))
        if order >= 4:
            raise(NotImplementedError)
        DSTTs[ii, :] = np.array(DSTT_row)

    return DSTTs

def compute_dx_STT(stateSize, dx_0, phi_k_list, order):
    dx_k = np.zeros(stateSize)

    if order >= 1:
        phi_k_first = phi_k_list[0]
        # dx_k += phi_k_first @ dx_0
        dx_k += np.einsum('ij,j->i', phi_k_first, dx_0)

    if order >= 2:
        phi_k_second = phi_k_list[1]
        dx_k += 1 / 2 * np.einsum('ijk,j,k->i', phi_k_second, dx_0, dx_0)

    if order >= 3:
        phi_k_third = phi_k_list[2]
        dx_k += 1 / 6 * np.einsum('ijkl,j,k,l->i', phi_k_third, dx_0, dx_0, dx_0)

    if order >= 4:
        phi_k_fourth = phi_k_list[3]
        dx_k += 1 / 24 * np.einsum('ijklm,j,k,l,m->i', phi_k_fourth, dx_0, dx_0, dx_0, dx_0)

    return dx_k


def compute_dx_DSTT(stateSize, dx_0, phi_k_list, R, order):
    # Eqn. 23 from DSTT paper
    dx_k = np.zeros(stateSize)
    dy_0 = R @ dx_0.T  # linear transformation to alternative basis

    if order >= 1:  # Still STM times dx_0
        phi_k_first = phi_k_list[0]
        dx_k += np.einsum('ij,j->i', phi_k_first, dx_0)

    if order >= 2:  # All higher order terms are in the alternative basis
        phi_k_second = phi_k_list[1]
        # dx_k += 1 / 2 * np.einsum('ijk,j,k->i', phi_k_second, dy_0, dy_0)
        # dx_k += 1 / 2 * np.einsum('i, j->i', phi_k_second, dy_0)
        dx_k += 1 / 2 * np.squeeze(np.multiply(phi_k_second, np.multiply(dy_0, dy_0)))
        # print(1 / 2 * np.multiply(phi_k_second, np.multiply(dy_0, dy_0)))

    if order >= 3:
        phi_k_third = phi_k_list[2]
        # dx_k += 1 / 6 * np.einsum('ijkl,j,k,l->i', phi_k_third, dy_0, dy_0, dy_0)
        dx_k += 1 / 6 * np.squeeze(np.multiply(phi_k_third, np.multiply(dy_0, np.multiply(dy_0, dy_0))))

    if order >= 4:
        phi_k_fourth = phi_k_list[3]
        # dx_k += 1 / 24 * np.einsum('ijklm,j,k,l,m->i', phi_k_fourth, dy_0, dy_0, dy_0, dy_0)
        # dx_k += 1 / 24 * np.einsum('i, j->i', phi_k_fourth, dy_0)
        dx_k += 1 / 24 * np.squeeze(np.multiply(phi_k_fourth, np.multiply(dy_0, np.multiply(dy_0, np.multiply(dy_0, dy_0)))))

    return dx_k

def compute_dx_DSTT_latentDim(dx_0, phi_k_list, R, stateSize, order):
    # For DSTT with variable latent dimension, but the same R for each order
    # Eqn. 23 from DSTT paper
    dx_k = np.zeros(stateSize)
    dy_0 = np.einsum('ij,j->i', R, dx_0) # linear transformation to alternative basis

    if order >= 1:  # Still STM times dx_0
        phi_k_first = phi_k_list[0]
        dx_k += np.einsum('ij,j->i', phi_k_first, dx_0)

    if order >= 2:  # All higher order terms are in the alternative basis
        phi_k_second = phi_k_list[1]
        dx_k += 1 / 2 * np.einsum('ijk,j,k->i', phi_k_second, dy_0, dy_0)

    if order >= 3:
        phi_k_third = phi_k_list[2]
        dx_k += 1 / 6 * np.einsum('ijkl,j,k,l->i', phi_k_third, dy_0, dy_0, dy_0)

    if order >= 4:
        phi_k_fourth = phi_k_list[3]
        raise(NotImplementedError)

    return dx_k

def compute_dx_XDSTT(stateSize, dx_0, phi_k_list, Rs, order):
    # XDSTT means different R for each order
    # Variable r for each dimension
    # Rs is a order x stateSize
    dx_k = np.zeros(stateSize)
    if order >= 1:  # Still STM times dx_0
        phi_k_first = phi_k_list[0]
        dx_k += np.einsum('ij,j->i', phi_k_first, dx_0)

    if order >= 2:  # All higher order terms are in the alternative basis
        R2 = Rs[0,:]
        dy_2 = R2 @ dx_0.T  # linear transformation to alternative basis
        phi_k_second = phi_k_list[1]
        # dx_k += 1 / 2 * np.einsum('ijk,j,k->i', phi_k_second, dy_0, dy_0)
        # dx_k += 1 / 2 * np.einsum('i, j->i', phi_k_second, dy_0)
        dx_k += 1 / 2 * np.squeeze(np.multiply(phi_k_second, np.multiply(dy_2, dy_2)))
        # print(1 / 2 * np.multiply(phi_k_second, np.multiply(dy_0, dy_0)))

    if order >= 3:
        R3 = Rs[1,:]
        dy_3 = R3 @ dx_0.T  # linear transformation to alternative basis
        phi_k_third = phi_k_list[2]
        # dx_k += 1 / 6 * np.einsum('ijkl,j,k,l->i', phi_k_third, dy_0, dy_0, dy_0)
        dx_k += 1 / 6 * np.squeeze(np.multiply(phi_k_third, np.multiply(dy_3, np.multiply(dy_3, dy_3))))

    if order >= 4:
        R4 = Rs[2,:]
        dy_4 = R4 @ dx_0.T  # linear transformation to alternative basis
        phi_k_fourth = phi_k_list[3]
        # dx_k += 1 / 24 * np.einsum('ijklm,j,k,l,m->i', phi_k_fourth, dy_0, dy_0, dy_0, dy_0)
        # dx_k += 1 / 24 * np.einsum('i, j->i', phi_k_fourth, dy_0)
        dx_k += 1 / 24 * np.squeeze(np.multiply(phi_k_fourth, np.multiply(dy_4, np.multiply(dy_4, np.multiply(dy_4, dy_4)))))

    return dx_k


def reshape_STT(Phi_list, order, stateSize):
    phi_out_list = []

    if order >= 1:
        second_start = stateSize ** 2
        phi_first = Phi_list[0:second_start].reshape((stateSize, stateSize))
        phi_out_list.append(phi_first)

    if order >= 2:
        third_start = second_start + stateSize ** 3
        phi_second = Phi_list[second_start:third_start].reshape((stateSize, stateSize, stateSize))
        phi_out_list.append(phi_second)

    if order >= 3:
        fourth_start = third_start + stateSize ** 4
        phi_third = Phi_list[third_start:fourth_start].reshape((stateSize, stateSize, stateSize, stateSize))
        phi_out_list.append(phi_third)

    if order >= 4:
        phi_fourth = Phi_list[fourth_start:(fourth_start + stateSize ** 5)].reshape(
            (stateSize, stateSize, stateSize, stateSize, stateSize))
        phi_out_list.append(phi_fourth)

    return phi_out_list


def reshape_DSTT(X, order, stateSize, latentDim=1):
    STM = X[0:stateSize ** 2].reshape(stateSize, stateSize)
    phi_out_list = []
    phi_out_list.append(STM)

    if order >= 2:  # Our DSTT_second
        second_start = stateSize ** 2
        phi_second = X[second_start:second_start + stateSize*latentDim*latentDim].reshape(stateSize, latentDim, latentDim)
        phi_out_list.append(phi_second)

    if order >= 3:  # Our DSTT_third
        third_start = second_start + stateSize*latentDim*latentDim
        phi_third = X[third_start:third_start + stateSize*latentDim*latentDim*latentDim].reshape(stateSize, latentDim, latentDim, latentDim)
        phi_out_list.append(phi_third)

    if order >= 4:  # Our
        raise(NotImplementedError)

    return phi_out_list


def getPerturbedTrajSTTsNonDimTkTkp1(dx_0, order, STTs_nd, Xs_stt_nd, params, dxs_true_nd=None):
    # If dxs_true is not None, use the true dx from the integrated solution for tk
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(STTs_nd), stateSize))
    Xs_stt_perturb_nd = np.zeros((len(STTs_nd), stateSize))
    dxs_stt = np.zeros((len(STTs_nd), stateSize))
    Xs_stt_perturb = np.zeros((len(STTs_nd), stateSize))

    dxs_stt_nd[0,:] = dx_0_nd
    Xs_stt_perturb_nd[0, :] = Xs_stt_nd[0, :] + dxs_stt_nd[0, :]
    dxs_stt[0,:] = dx_0
    Xs_stt_perturb[0, :] = convertFromNonDimDens(Xs_stt_nd[0, :], params) + dxs_stt[0, :]
    d1, _ = Xs_stt_nd.shape

    for ii in range(d1-1):
        if dxs_true_nd is None:
            dx_km1_nd = dxs_stt_nd[ii,:]
        else:
            dx_km1_nd = dxs_true_nd[ii,:]
        STT_list = reshape_STT(STTs_nd[ii+1, :], order, stateSize)
        dx_k_nd = compute_dx_STT(stateSize, dx_km1_nd, STT_list, order)
        dxs_stt_nd[ii + 1, :] = dx_k_nd
        Xs_stt_perturb_nd[ii+1, :] = Xs_stt_nd[ii+1, :] + dxs_stt_nd[ii+1, :]
        dxs_stt[ii + 1, :] = convertFromNonDimDens(dx_k_nd, params)
        Xs_stt_perturb[ii+1, :] = convertFromNonDimDens(Xs_stt_nd[ii+1, :], params) + dxs_stt[ii+1, :]

    return Xs_stt_perturb, dxs_stt

def getPerturbedTrajSTTsNonDimT0Tk(dx_0, order, STTs_nd, Xs_stt_nd, params, dxs_true_nd=None):
    # Can pass in a different dx0 for each time step if desired
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(STTs_nd), stateSize))
    Xs_stt_perturb_nd = np.zeros((len(STTs_nd), stateSize))
    dxs_stt = np.zeros((len(STTs_nd), stateSize))
    Xs_stt_perturb = np.zeros((len(STTs_nd), stateSize))

    dxs_stt_nd[0,:] = dx_0_nd
    Xs_stt_perturb_nd[0, :] = Xs_stt_nd[0, :] + dxs_stt_nd[0, :]
    dxs_stt[0,:] = dx_0
    Xs_stt_perturb[0, :] = convertFromNonDimDens(Xs_stt_nd[0, :], params) + dxs_stt[0, :]
    d1, _ = Xs_stt_nd.shape

    for ii in range(d1-1):
        STT_list = reshape_STT(STTs_nd[ii+1, :], order, stateSize)
        if dxs_true_nd is None:
            dx_k_nd = compute_dx_STT(stateSize, dx_0_nd, STT_list, order)
        else:
            dx_k_nd = compute_dx_STT(stateSize, dxs_true_nd[ii,:], STT_list, order)
        dxs_stt_nd[ii + 1, :] = dx_k_nd
        Xs_stt_perturb_nd[ii+1, :] = Xs_stt_nd[ii+1, :] + dxs_stt_nd[ii+1, :]
        dxs_stt[ii + 1, :] = convertFromNonDimDens(dx_k_nd, params)
        Xs_stt_perturb[ii+1, :] = convertFromNonDimDens(Xs_stt_nd[ii+1, :], params) + dxs_stt[ii+1, :]

    return Xs_stt_perturb, dxs_stt

def getXfSTT(dx_0, order, STTs_nd_t0tf, Xsf_nd, params):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)
    stateSize = len(dx_0)

    STT_list = reshape_STT(STTs_nd_t0tf[1, :], order, stateSize)
    dx_f_nd = compute_dx_STT(stateSize, dx_0_nd, STT_list, order)
    Xf_nd = Xsf_nd + dx_f_nd
    Xf = convertFromNonDimDens(Xf_nd, params)
    dx_f = convertFromNonDimDens(dx_f_nd, params)

    return Xf, dx_f

def getPerturbedTrajSTTsNonDimT0TkTkTf(dx_0, order, STTs_nd_t0tk, STTs_nd_tktf, Xs_stt_nd, params):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(STTs_nd_t0tk)-1, stateSize))
    Xs_stt_perturb_nd = np.zeros((len(STTs_nd_t0tk)-1, stateSize))
    dxs_stt = np.zeros((len(STTs_nd_t0tk)-1, stateSize))
    Xs_stt_perturb = np.zeros((len(STTs_nd_t0tk)-1, stateSize))

    dxs_stt_nd[0,:] = dx_0_nd
    Xs_stt_perturb_nd[0, :] = Xs_stt_nd[0, :] + dxs_stt_nd[0, :]
    dxs_stt[0,:] = dx_0
    Xs_stt_perturb[0, :] = convertFromNonDimDens(Xs_stt_nd[0, :], params) + dxs_stt[0, :]
    d1, _ = Xs_stt_nd.shape

    for ii in range(d1-2):
        STT_t0tk_list = reshape_STT(STTs_nd_t0tk[ii+1, :], order, stateSize)  # skip first identity entry, then t0 to tk
        STT_tktf_list = reshape_STT(STTs_nd_tktf[ii+2, :], order, stateSize)  # skip first two identity entries, then tk>0 to tf
        dx_k_nd = compute_dx_STT(stateSize, dx_0_nd, STT_t0tk_list, order)
        dx_f_nd = compute_dx_STT(stateSize, dx_k_nd, STT_tktf_list, order)
        dxs_stt_nd[ii + 1, :] = dx_f_nd
        Xs_stt_perturb_nd[ii+1, :] = Xs_stt_nd[-1, :] + dxs_stt_nd[ii+1, :]
        dxs_stt[ii + 1, :] = convertFromNonDimDens(dx_f_nd, params)
        Xs_stt_perturb[ii+1, :] = convertFromNonDimDens(Xs_stt_nd[-1, :], params) + dxs_stt[ii+1, :]

    # Output is a array where the first entry is the perturbed state, and all other entries are the state or dx at the final time
    return Xs_stt_perturb, dxs_stt

def getPerturbedTrajSTTsNonDimTkTf(dx_0, order, STTs_nd_tktf, Xs_stt_nd, Xs_pert_nd, params):
    # Propagated from true perturbed state (Xs_pert_nd) through the STTs to time f for each input time k
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(STTs_nd_tktf)-1, stateSize))
    Xs_stt_perturb_nd = np.zeros((len(STTs_nd_tktf)-1, stateSize))
    dxs_stt = np.zeros((len(STTs_nd_tktf)-1, stateSize))
    Xs_stt_perturb = np.zeros((len(STTs_nd_tktf)-1, stateSize))

    d1, _ = Xs_stt_nd.shape

    for ii in range(d1-1):
        STT_tktf_list = reshape_STT(STTs_nd_tktf[ii+1, :], order, stateSize)
        d_k_nd_true = Xs_pert_nd[ii, :] - Xs_stt_nd[ii, :]
        dx_f_nd = compute_dx_STT(stateSize, d_k_nd_true, STT_tktf_list, order)
        dxs_stt_nd[ii, :] = dx_f_nd
        Xs_stt_perturb_nd[ii, :] = Xs_stt_nd[-1, :] + dxs_stt_nd[ii, :]
        dxs_stt[ii, :] = convertFromNonDimDens(dx_f_nd, params)
        Xs_stt_perturb[ii, :] = convertFromNonDimDens(Xs_stt_nd[-1, :], params) + dxs_stt[ii, :]

    # Output is a array where the first entry is the perturbed state, and all other entries are the state or dx at the final time
    # All dx-s are at time f!
    return Xs_stt_perturb, dxs_stt

def getPerturbedTrajDSTTsNonDim(dx_0, order, DSTTs_nd, Xs_dstt_nd, params, Rs):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_dstt_nd = np.zeros((len(DSTTs_nd), stateSize))
    Xs_dstt_perturb_nd = np.zeros((len(DSTTs_nd), stateSize))
    dxs_dstt = np.zeros((len(DSTTs_nd), stateSize))
    Xs_dstt_perturb = np.zeros((len(DSTTs_nd), stateSize))

    dxs_dstt_nd[0,:] = dx_0_nd
    Xs_dstt_perturb_nd[0, :] = Xs_dstt_nd[0, :] + dxs_dstt_nd[0, :]
    dxs_dstt[0,:] = dx_0
    Xs_dstt_perturb[0, :] = convertFromNonDimDens(Xs_dstt_nd[0, :], params) + dxs_dstt[0, :]
    d1, _ = Xs_dstt_nd.shape

    for ii in range(d1-1):
        dx_km1_nd = dxs_dstt_nd[ii,:]
        STT_list = reshape_DSTT(DSTTs_nd[ii+1, :], order, stateSize)
        dx_k_nd = compute_dx_DSTT(stateSize, dx_km1_nd, STT_list, Rs[ii,0,:], order)
        dxs_dstt_nd[ii + 1, :] = dx_k_nd
        Xs_dstt_perturb_nd[ii+1, :] = Xs_dstt_nd[ii+1, :] + dxs_dstt_nd[ii+1, :]
        dxs_dstt[ii + 1, :] = convertFromNonDimDens(dx_k_nd, params)
        Xs_dstt_perturb[ii+1, :] = convertFromNonDimDens(Xs_dstt_nd[ii+1, :], params) + dxs_dstt[ii+1, :]

    return Xs_dstt_perturb, dxs_dstt

def getPerturbedTrajXDSTTsNonDimTkTkp1(dx_0, order, DSTTs_nd, Xs_dstt_nd, Rs, params, dxs_true_nd=None, latent_dim=1):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(Xs_dstt_nd), stateSize))
    Xs_stt_perturb_nd = np.zeros((len(Xs_dstt_nd), stateSize))
    dxs_stt = np.zeros((len(Xs_dstt_nd), stateSize))
    Xs_stt_perturb = np.zeros((len(Xs_dstt_nd), stateSize))

    dxs_stt_nd[0,:] = dx_0_nd
    Xs_stt_perturb_nd[0, :] = Xs_dstt_nd[0, :] + dxs_stt_nd[0, :]
    dxs_stt[0,:] = dx_0
    Xs_stt_perturb[0, :] = convertFromNonDimDens(Xs_dstt_nd[0, :], params) + dxs_stt[0, :]
    d1, _ = Xs_dstt_nd.shape

    for ii in range(d1-1):
        if dxs_true_nd is None:
            dx_km1_nd = dxs_stt_nd[ii,:]
        else:
            dx_km1_nd = dxs_true_nd[ii,:]
        Rs_k = Rs[ii+1,:,:]

        if latent_dim == 1:  # Single y dimension, but different R for each order
            DSTT_list = reshape_DSTT(DSTTs_nd[ii+1, :], order, stateSize)
            dx_k_nd = compute_dx_XDSTT(stateSize, dx_km1_nd, DSTT_list, Rs_k, order)
        else: # larger latent dimension
            DSTT_list = reshape_DSTT(DSTTs_nd[ii+1, :], order, stateSize, latentDim=latent_dim)
            dx_k_nd = compute_dx_DSTT_latentDim(dx_km1_nd, DSTT_list, Rs_k, stateSize, order)
        dxs_stt_nd[ii + 1, :] = dx_k_nd
        Xs_stt_perturb_nd[ii+1, :] = Xs_dstt_nd[ii+1, :] + dxs_stt_nd[ii+1, :]
        dxs_stt[ii + 1, :] = convertFromNonDimDens(dx_k_nd, params)
        Xs_stt_perturb[ii+1, :] = convertFromNonDimDens(Xs_dstt_nd[ii+1, :], params) + dxs_stt[ii+1, :]

    return Xs_stt_perturb, dxs_stt

def getPerturbedTrajXDSTTsNonDimT0Tk(dx_0, order, DSTTs_nd, Xs_dstt_nd, Rs, params, dxs_true_nd=None, latent_dim=1):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(Xs_dstt_nd), stateSize))
    Xs_stt_perturb_nd = np.zeros((len(Xs_dstt_nd), stateSize))
    dxs_stt = np.zeros((len(Xs_dstt_nd), stateSize))
    Xs_stt_perturb = np.zeros((len(Xs_dstt_nd), stateSize))

    dxs_stt_nd[0,:] = dx_0_nd
    Xs_stt_perturb_nd[0, :] = Xs_dstt_nd[0, :] + dxs_stt_nd[0, :]
    dxs_stt[0,:] = dx_0
    Xs_stt_perturb[0, :] = convertFromNonDimDens(Xs_dstt_nd[0, :], params) + dxs_stt[0, :]
    d1, _ = Xs_dstt_nd.shape

    for ii in range(d1-1):
        DSTT_list = reshape_DSTT(DSTTs_nd[ii, :], order, stateSize, latent_dim)
        Rs_k = Rs[ii,:,:]
        if dxs_true_nd is None:
            if latent_dim == 1:
                dx_k_nd = compute_dx_XDSTT(stateSize, dx_0_nd, DSTT_list, Rs_k, order)
            else:
                dx_k_nd = compute_dx_DSTT_latentDim(dx_0_nd, DSTT_list, Rs_k, stateSize, order)
        else:
            if latent_dim == 1:
                dx_k_nd = compute_dx_XDSTT(stateSize, dxs_true_nd[ii,:], DSTT_list, Rs_k, order)
            else:
                dx_k_nd = compute_dx_DSTT_latentDim(stateSize, dxs_true_nd[ii,:], DSTT_list, Rs_k, order)
        dxs_stt_nd[ii + 1, :] = dx_k_nd
        Xs_stt_perturb_nd[ii+1, :] = Xs_dstt_nd[ii+1, :] + dxs_stt_nd[ii+1, :]
        dxs_stt[ii + 1, :] = convertFromNonDimDens(dx_k_nd, params)
        Xs_stt_perturb[ii+1, :] = convertFromNonDimDens(Xs_dstt_nd[ii+1, :], params) + dxs_stt[ii+1, :]

    return Xs_stt_perturb, dxs_stt

def getPerturbedTrajXDSTTsNonDimTkTf(dx_0, order, DSTTs_nd, Xs_dstt_nd, Xs_pert_nd, Rs, params, latent_dim=1):
    dx_0_nd = convertToNonDimDens(dx_0, params)

    stateSize = len(dx_0)
    dxs_stt_nd = np.zeros((len(Xs_dstt_nd)-1, stateSize))
    Xs_stt_perturb_nd = np.zeros((len(Xs_dstt_nd)-1, stateSize))
    dxs_stt = np.zeros((len(Xs_dstt_nd)-1, stateSize))
    Xs_stt_perturb = np.zeros((len(Xs_dstt_nd)-1, stateSize))

    d1, _ = Xs_dstt_nd.shape

    for ii in range(d1-1):
        DSTT_list = reshape_DSTT(DSTTs_nd[ii, :], order, stateSize, latent_dim)
        Rs_k = Rs[ii,:,:]
        dx_k_true = Xs_pert_nd[ii, :] - Xs_dstt_nd[ii, :]
        if latent_dim == 1:
            dx_f_nd = compute_dx_XDSTT(stateSize, dx_k_true, DSTT_list, Rs_k, order)
        else:
            dx_f_nd = compute_dx_DSTT_latentDim(dx_k_true, DSTT_list, Rs_k, stateSize, order)
        dxs_stt_nd[ii, :] = dx_f_nd
        Xs_stt_perturb_nd[ii, :] = Xs_dstt_nd[-1, :] + dxs_stt_nd[ii, :]
        dxs_stt[ii, :] = convertFromNonDimDens(dx_f_nd, params)
        Xs_stt_perturb[ii, :] = convertFromNonDimDens(Xs_dstt_nd[-1, :], params) + dxs_stt[ii, :]

    # All states and dx-s are at time f!
    return Xs_stt_perturb, dxs_stt

def getXfXDSTT(dx_0, order, DSTTs_nd_t0tf, Xsf_nd, Rs, params, latent_dim=1):
    # nondimensionalize dx_0
    dx_0_nd = convertToNonDimDens(dx_0, params)
    stateSize = len(dx_0)

    DSTT_list = reshape_DSTT(DSTTs_nd_t0tf[0, :], order, stateSize, latent_dim)
    if latent_dim == 1:
        dx_f_nd = compute_dx_XDSTT(stateSize, dx_0_nd, DSTT_list, Rs[0,:,:], order)
    else:
        dx_f_nd = compute_dx_DSTT_latentDim(dx_0_nd, DSTT_list, Rs[0,:,:], stateSize, order)


    Xf_nd = Xsf_nd + dx_f_nd
    Xf = convertFromNonDimDens(Xf_nd, params)
    dx_f = convertFromNonDimDens(dx_f_nd, params)

    return Xf, dx_f

def compute_expected_mixed(dm, P, order, R, stateSize):
    # R is (_, order - 1, state_size) for latentDim=1, or
    #      (_, order - 1, latentDim, state_size) for larger latent dimension. (deleted time index)
    # Output:
    # expected_dict = {'mean': [], 'cov': []} where each dict has a sub-dictionary where the key is the order of the STT / DSTTs multiplied together
    # example: expected_dict['cov']['12'] means STM x second-order DSTT

    expected_dict = {'mean': {}, 'cov': {}}
    _, latentDim, _ = R.shape

    if order >= 1:
        ## MEAN PROPAGATION
        # 1 = E[dx]
        expected_dict['mean']['1'] = dm

        ## COV PROPAGATION
        # 11 = E[dxdx]
        expected_dict['cov']['11'] = contract('i,j->ij', dm, dm) + P

    if order >= 2:
        dy2 = R[0, :, :] @ dm
        Py2 = R[0, :, :] @ P @ R[0, :, :].T
        Pxy2 = np.einsum('ij,kj->ik', P, R[0, :, :])
        ## MEAN PROPAGATION
        # 2 = E[dy2 dy2]
        expected_dict['mean']['2'] = contract('i,j->ij', dy2, dy2) + Py2

        ## COV PROPAGATION
        # 12 = E[dxdy2dy2]
        # 21 = E[dy2dy2dx]
        # 12 = 21 because multiplication of scalar RVs is commutative
        # These tensors will both be stateSize, latentDim, latentDim so tensor contractions need to keep track of that
        expected_dict['cov']['12'] = np.einsum('i,j,k->ijk', dm, dy2, dy2) + np.einsum('i,jk->ijk', dm, Py2) + np.einsum('j,ik->ijk', dy2, Pxy2) + np.einsum('k,ij->ijk', dy2, Pxy2)
        expected_dict['cov']['21'] = expected_dict['cov']['12']
        # 22 = E[dy2dy2dy2dy2]
        expected_fourth = np.zeros((latentDim, latentDim, latentDim, latentDim))
        expected_fourth += contract('i,j,k,l->ijkl', dy2, dy2, dy2, dy2)
        expected_fourth += contract('k,l,ij->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('j,l,ik->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('j,k,il->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('i,l,jk->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('i,k,jl->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('i,j,kl->ijkl', dy2, dy2, Py2)
        expected_fourth += contract('il,jk->ijkl', Py2, Py2)
        expected_fourth += contract('ik,jl->ijkl', Py2, Py2)
        expected_fourth += contract('ij,kl->ijkl', Py2, Py2)
        expected_dict['cov']['22'] = expected_fourth

    if order >= 3:
        raise NotImplementedError
    if order >= 4:
        raise NotImplementedError

    return expected_dict


def compute_expected(dm, P, order, stateSize):
    # dm -> change in mean
    # P -> covariance at time k
    expected_list = []
    if order >= 1:
        # FIRST ORDER
        # --------------------------------------------------------
        expected_first = dm
        expected_list.append(expected_first)

        # SECOND ORDER
        # --------------------------------------------------------
        expected_second = contract('i,j->ij', dm, dm) + P
        expected_list.append(expected_second)

    if order >= 2:
        # THIRD ORDER
        # --------------------------------------------------------
        expected_third = np.zeros((stateSize, stateSize, stateSize))
        expected_third += contract('i,j,k->ijk', dm, dm, dm)
        expected_third += contract('k,ij->ijk', dm, P)
        expected_third += contract('j,ik->ijk', dm, P)
        expected_third += contract('i,jk->ijk', dm, P)
        expected_list.append(expected_third)

    if order >= 2:
        # FOURTH ORDER
        # --------------------------------------------------------
        expected_fourth = np.zeros((stateSize, stateSize, stateSize, stateSize))
        expected_fourth += contract('i,j,k,l->ijkl', dm, dm, dm, dm)
        expected_fourth += contract('k,l,ij->ijkl', dm, dm, P)
        expected_fourth += contract('j,l,ik->ijkl', dm, dm, P)
        expected_fourth += contract('j,k,il->ijkl', dm, dm, P)
        expected_fourth += contract('i,l,jk->ijkl', dm, dm, P)
        expected_fourth += contract('i,k,jl->ijkl', dm, dm, P)
        expected_fourth += contract('i,j,kl->ijkl', dm, dm, P)
        expected_fourth += contract('il,jk->ijkl', P, P)
        expected_fourth += contract('ik,jl->ijkl', P, P)
        expected_fourth += contract('ij,kl->ijkl', P, P)
        expected_list.append(expected_fourth)

    if order >= 3:
        # FIFTH ORDER
        # --------------------------------------------------------
        expected_fifth = np.zeros((stateSize, stateSize,
                                   stateSize, stateSize, stateSize))
        expected_fifth += contract('m,ij,kl->ijklm', dm, P, P)
        expected_fifth += contract('l,ij,km->ijklm', dm, P, P)
        expected_fifth += contract('k,ij,lm->ijklm', dm, P, P)
        expected_fifth += contract('m,ik,jl->ijklm', dm, P, P)
        expected_fifth += contract('l,ik,jm->ijklm', dm, P, P)
        expected_fifth += contract('j,ik,lm->ijklm', dm, P, P)
        expected_fifth += contract('m,il,jk->ijklm', dm, P, P)
        expected_fifth += contract('l,im,jk->ijklm', dm, P, P)
        expected_fifth += contract('i,jk,lm->ijklm', dm, P, P)
        expected_fifth += contract('k,il,jm->ijklm', dm, P, P)
        expected_fifth += contract('j,il,km->ijklm', dm, P, P)
        expected_fifth += contract('k,im,jl->ijklm', dm, P, P)
        expected_fifth += contract('i,jl,km->ijklm', dm, P, P)
        expected_fifth += contract('j,im,kl->ijklm', dm, P, P)
        expected_fifth += contract('i,jm,kl->ijklm', dm, P, P)
        expected_fifth += contract('k,l,m,ij->ijklm', dm, dm, dm, P)
        expected_fifth += contract('j,l,m,ik->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,l,m,jk->ijklm', dm, dm, dm, P)
        expected_fifth += contract('j,k,m,il->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,k,m,jl->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,j,m,kl->ijklm', dm, dm, dm, P)
        expected_fifth += contract('j,k,l,im->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,k,l,jm->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,j,l,km->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,j,k,lm->ijklm', dm, dm, dm, P)
        expected_fifth += contract('i,j,k,l,m->ijklm', dm, dm, dm, dm, dm)
        expected_list.append(expected_fifth)

        # SIXTH ORDER
        # --------------------------------------------------------
        expected_sixth = np.zeros((stateSize, stateSize, stateSize,
                                   stateSize, stateSize, stateSize))
        expected_sixth += contract('ij,kl,mn->ijklmn', P, P, P)
        expected_sixth += contract('ij,km,ln->ijklmn', P, P, P)
        expected_sixth += contract('ij,kn,lm->ijklmn', P, P, P)
        expected_sixth += contract('ik,jl,mn->ijklmn', P, P, P)
        expected_sixth += contract('ik,jm,ln->ijklmn', P, P, P)
        expected_sixth += contract('ik,jn,lm->ijklmn', P, P, P)
        expected_sixth += contract('il,jk,mn->ijklmn', P, P, P)
        expected_sixth += contract('im,jk,ln->ijklmn', P, P, P)
        expected_sixth += contract('in,jk,lm->ijklmn', P, P, P)
        expected_sixth += contract('il,jm,kn->ijklmn', P, P, P)
        expected_sixth += contract('il,jn,km->ijklmn', P, P, P)
        expected_sixth += contract('im,jl,kn->ijklmn', P, P, P)
        expected_sixth += contract('in,jl,km->ijklmn', P, P, P)
        expected_sixth += contract('im,jn,kl->ijklmn', P, P, P)
        expected_sixth += contract('in,jm,kl->ijklmn', P, P, P)
        expected_sixth += contract('m,n,ij,kl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,n,ij,km->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,n,ij,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,m,ij,kn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,m,ij,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,l,ij,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('m,n,ik,jl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,n,ik,jm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,n,ik,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,m,ik,jn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,m,ik,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,l,ik,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('m,n,il,jk->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,n,im,jk->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,n,jk,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('l,m,in,jk->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,m,jk,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,l,jk,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,n,il,jm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,n,il,km->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,m,il,jn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,m,il,kn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,k,il,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,n,im,jl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,n,jl,km->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,m,in,jl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,m,jl,kn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,k,jl,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,n,im,kl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,n,jm,kl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,m,in,kl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,m,jn,kl->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,j,kl,mn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,l,im,jn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,l,im,kn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,k,im,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,l,in,jm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,l,jm,kn->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,k,jm,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,l,in,km->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,l,jn,km->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,j,km,ln->ijklmn', dm, dm, P, P)
        expected_sixth += contract('j,k,in,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,k,jn,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('i,j,kn,lm->ijklmn', dm, dm, P, P)
        expected_sixth += contract('k,l,m,n,ij->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('j,l,m,n,ik->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,l,m,n,jk->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('j,k,m,n,il->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,k,m,n,jl->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,m,n,kl->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('j,k,l,n,im->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,k,l,n,jm->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,l,n,km->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,k,n,lm->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('j,k,l,m,in->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,k,l,m,jn->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,l,m,kn->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,k,m,ln->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,k,l,mn->ijklmn', dm, dm, dm, dm, P)
        expected_sixth += contract('i,j,k,l,m,n->ijklmn', dm, dm, dm, dm, dm, dm)
        expected_list.append(expected_sixth)

    if order >=4:
        # SEVENTH ORDER
        # --------------------------------------------------------
        expected_seventh = np.zeros((stateSize, stateSize, stateSize, stateSize,
                                     stateSize, stateSize, stateSize))
        expected_seventh += contract('o,ij,kl,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ij,kl,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ij,kl,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,ij,km,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ij,km,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ij,km,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,ij,kn,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ij,ko,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,ij,lm,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ij,kn,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ij,kn,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ij,ko,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,ij,ln,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ij,ko,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,ij,lo,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,ik,jl,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ik,jl,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ik,jl,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,ik,jm,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ik,jm,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ik,jm,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,ik,jn,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,ik,jo,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,ik,lm,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ik,jn,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ik,jn,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,ik,jo,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,ik,ln,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,ik,jo,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,ik,lo,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,il,jk,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,il,jk,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,il,jk,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,im,jk,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,im,jk,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,im,jk,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,in,jk,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,io,jk,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jk,lm,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,in,jk,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,in,jk,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,io,jk,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jk,ln,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,io,jk,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jk,lo,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,il,jm,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,il,jm,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,il,jm,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,il,jn,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,il,jo,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,il,km,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,il,jn,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,il,jn,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,il,jo,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,il,kn,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,il,jo,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,il,ko,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,im,jl,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,im,jl,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,im,jl,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,in,jl,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,io,jl,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jl,km,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,in,jl,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,in,jl,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,io,jl,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jl,kn,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,io,jl,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jl,ko,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,im,jn,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,im,jo,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,im,kl,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('o,in,jm,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('n,io,jm,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jm,kl,no->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,in,jo,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,in,kl,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,io,jn,kl->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jn,kl,mo->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,io,kl,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jo,kl,mn->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,im,jn,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,im,jn,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,im,jo,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,im,kn,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,im,jo,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,im,ko,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,in,jm,ko->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,in,jm,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,io,jm,kn->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jm,kn,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,io,jm,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jm,ko,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,in,jo,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,in,km,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('l,io,jn,km->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jn,km,lo->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,io,km,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jo,km,ln->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,in,jo,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,in,ko,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('k,io,jn,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jn,ko,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('j,io,kn,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('i,jo,kn,lm->ijklmno', dm, P, P, P)
        expected_seventh += contract('m,n,o,ij,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,n,o,ij,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,n,o,ij,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,o,ij,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,o,ij,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,o,ij,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,n,ij,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,n,ij,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,n,ij,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,m,ij,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('m,n,o,ik,jl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,n,o,ik,jm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,n,o,ik,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,o,ik,jn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,o,ik,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,o,ik,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,n,ik,jo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,n,ik,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,n,ik,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,m,ik,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('m,n,o,il,jk->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,n,o,im,jk->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,n,o,jk,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,o,in,jk->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,o,jk,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,o,jk,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('l,m,n,io,jk->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,n,jk,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,n,jk,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,m,jk,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,n,o,il,jm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,n,o,il,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,o,il,jn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,o,il,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,o,il,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,n,il,jo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,n,il,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,n,il,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,m,il,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,n,o,im,jl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,n,o,jl,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,o,in,jl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,o,jl,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,o,jl,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,m,n,io,jl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,n,jl,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,n,jl,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,m,jl,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,n,o,im,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,n,o,jm,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,o,in,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,o,jn,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,o,kl,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,m,n,io,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,m,n,jo,kl->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,n,kl,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,m,kl,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,o,im,jn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,o,im,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,o,im,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,n,im,jo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,n,im,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,n,im,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,l,im,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,o,in,jm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,o,jm,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,o,jm,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,n,io,jm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,n,jm,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,n,jm,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,l,jm,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,o,in,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,o,jn,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,o,km,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,n,io,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,n,jo,km->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,n,km,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,l,km,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,o,in,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,o,jn,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,o,kn,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,n,io,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,n,jo,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,n,ko,lm->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,k,lm,no->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,m,in,jo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,m,in,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,m,in,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,l,in,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,m,io,jn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,m,jn,ko->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,m,jn,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,l,jn,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,l,m,io,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,l,m,jo,kn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,m,kn,lo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,l,kn,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,m,io,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,m,jo,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,m,ko,ln->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,k,ln,mo->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('j,k,l,io,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,k,l,jo,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,l,ko,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('i,j,k,lo,mn->ijklmno', dm, dm, dm, P, P)
        expected_seventh += contract('k,l,m,n,o,ij->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('j,l,m,n,o,ik->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,l,m,n,o,jk->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('j,k,m,n,o,il->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,k,m,n,o,jl->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,m,n,o,kl->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('j,k,l,n,o,im->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,k,l,n,o,jm->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,l,n,o,km->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,n,o,lm->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('j,k,l,m,o,in->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,k,l,m,o,jn->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,l,m,o,kn->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,m,o,ln->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,l,o,mn->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('j,k,l,m,n,io->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,k,l,m,n,jo->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,l,m,n,ko->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,m,n,lo->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,l,n,mo->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,l,m,no->ijklmno', dm, dm, dm, dm, dm, P)
        expected_seventh += contract('i,j,k,l,m,n,o->ijklmno', dm, dm, dm, dm, dm, dm, dm)
        expected_list.append(expected_seventh)

        # EIGHTH ORDER
        # --------------------------------------------------------
        expected_eighth = np.zeros((stateSize, stateSize, stateSize, stateSize,
                                    stateSize, stateSize, stateSize, stateSize))
        expected_eighth += contract('ij,kl,mn,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kl,mo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kl,mp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,km,ln,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,km,lo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,km,lp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kn,lm,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,ko,lm,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kp,lm,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kn,lo,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kn,lp,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,ko,ln,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kp,ln,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,ko,lp,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ij,kp,lo,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jl,mn,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jl,mo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jl,mp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jm,ln,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jm,lo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jm,lp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jn,lm,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jo,lm,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jp,lm,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jn,lo,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jn,lp,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jo,ln,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jp,ln,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jo,lp,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ik,jp,lo,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jk,mn,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jk,mo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jk,mp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jk,ln,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jk,lo,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jk,lp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jk,lm,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jk,lm,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jk,lm,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jk,lo,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jk,lp,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jk,ln,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jk,ln,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jk,lp,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jk,lo,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jm,kn,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jm,ko,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jm,kp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jn,km,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jo,km,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jp,km,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jn,ko,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jn,kp,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jo,kn,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jp,kn,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jo,kp,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('il,jp,ko,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jl,kn,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jl,ko,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jl,kp,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jl,km,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jl,km,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jl,km,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jl,ko,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jl,kp,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jl,kn,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jl,kn,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jl,kp,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jl,ko,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jn,kl,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jo,kl,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jp,kl,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jm,kl,op->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jm,kl,np->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jm,kl,no->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jo,kl,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jp,kl,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jn,kl,mp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jn,kl,mo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jp,kl,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jo,kl,mn->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jn,ko,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jn,kp,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jo,kn,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jp,kn,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jo,kp,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('im,jp,ko,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jm,ko,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jm,kp,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jm,kn,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jm,kn,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jm,kp,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jm,ko,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jo,km,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jp,km,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jn,km,lp->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jn,km,lo->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jp,km,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jo,km,ln->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jo,kp,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('in,jp,ko,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jn,kp,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jn,ko,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('io,jp,kn,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('ip,jo,kn,lm->ijklmnop', P, P, P, P)
        expected_eighth += contract('o,p,ij,kl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ij,kl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ij,kl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ij,kl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ij,kl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ij,kl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,ij,km,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ij,km,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ij,km,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ij,km,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ij,km,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ij,km,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,ij,kn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ij,ko,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,ij,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ij,kp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ij,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ij,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ij,kn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ij,kn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ij,kn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ij,kn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ij,kn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ij,ko,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,ij,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ij,kp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ij,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ij,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ij,ko,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,ij,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ij,kp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ij,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ij,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ij,ko,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ij,ko,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ij,ko,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ij,kp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ij,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ij,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ij,kp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ij,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ij,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ij,kp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ij,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ij,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,ik,jl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ik,jl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ik,jl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ik,jl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ik,jl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ik,jl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,ik,jm,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ik,jm,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ik,jm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ik,jm,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ik,jm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ik,jm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,ik,jn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,ik,jo,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,ik,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ik,jp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ik,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ik,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ik,jn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ik,jn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ik,jn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ik,jn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ik,jn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,ik,jo,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,ik,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ik,jp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ik,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ik,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,ik,jo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,ik,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ik,jp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ik,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ik,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ik,jo,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ik,jo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ik,jo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ik,jp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ik,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ik,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ik,jp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ik,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ik,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ik,jp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ik,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ik,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,il,jk,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,il,jk,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,il,jk,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,il,jk,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,il,jk,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,il,jk,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,im,jk,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,im,jk,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,im,jk,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,im,jk,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,im,jk,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,im,jk,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,in,jk,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,io,jk,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jk,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ip,jk,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jk,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jk,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,in,jk,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,in,jk,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,in,jk,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,in,jk,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,in,jk,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,io,jk,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jk,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ip,jk,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jk,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jk,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,io,jk,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jk,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ip,jk,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jk,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jk,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,io,jk,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,io,jk,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,io,jk,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ip,jk,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jk,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jk,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ip,jk,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jk,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jk,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ip,jk,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jk,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jk,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,il,jm,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,il,jm,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,il,jm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,il,jm,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,il,jm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,il,jm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,il,jn,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,il,jo,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,il,km,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,il,jp,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,il,km,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,il,km,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,il,jn,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,il,jn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,il,jn,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,il,jn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,il,jn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,il,jo,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,il,kn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,il,jp,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,il,kn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,il,kn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,il,jo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,il,ko,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,il,jp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,il,kp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,il,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,il,jo,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,il,jo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,il,jo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,il,jp,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,il,ko,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,il,ko,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,il,jp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,il,kp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,il,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,il,jp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,il,kp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,il,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,im,jl,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,im,jl,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,im,jl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,im,jl,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,im,jl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,im,jl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,in,jl,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,io,jl,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jl,km,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ip,jl,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jl,km,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jl,km,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,in,jl,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,in,jl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,in,jl,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,in,jl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,in,jl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,io,jl,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jl,kn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ip,jl,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jl,kn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jl,kn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,io,jl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jl,ko,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ip,jl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jl,kp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jl,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,io,jl,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,io,jl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,io,jl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ip,jl,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jl,ko,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jl,ko,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ip,jl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jl,kp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jl,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ip,jl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jl,kp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jl,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,im,jn,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,im,jo,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,im,kl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,im,jp,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,im,kl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,im,kl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('o,p,in,jm,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,p,io,jm,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jm,kl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('n,o,ip,jm,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jm,kl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jm,kl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,in,jo,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,in,kl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,in,jp,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,in,kl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,in,kl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,p,io,jn,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jn,kl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,o,ip,jn,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jn,kl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jn,kl,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,io,kl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jo,kl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ip,kl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jp,kl,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kl,mn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,io,jp,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,io,kl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,io,kl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,ip,jo,kl->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jo,kl,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jo,kl,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ip,kl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jp,kl,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kl,mo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ip,kl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jp,kl,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kl,mp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,im,jn,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,im,jn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,im,jn,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,im,jn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,im,jn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,im,jo,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,im,kn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,im,jp,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,im,kn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,im,kn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,im,jo,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,im,ko,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,im,jp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,im,kp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,im,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,im,jo,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,im,jo,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,im,jo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,im,jp,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,im,ko,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,im,ko,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,im,jp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,im,kp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,im,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,im,jp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,im,kp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,im,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,in,jm,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,in,jm,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,in,jm,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,in,jm,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,in,jm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,io,jm,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jm,kn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ip,jm,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jm,kn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jm,kn,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,io,jm,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jm,ko,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ip,jm,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jm,kp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jm,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,io,jm,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,io,jm,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,io,jm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ip,jm,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jm,ko,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jm,ko,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ip,jm,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jm,kp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jm,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ip,jm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jm,kp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jm,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,in,jo,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,in,km,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,in,jp,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,in,km,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,in,km,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,p,io,jn,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jn,km,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,o,ip,jn,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jn,km,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jn,km,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,io,km,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jo,km,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ip,km,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jp,km,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,km,ln,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,io,jp,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,io,km,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,io,km,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,n,ip,jo,km->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jo,km,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jo,km,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ip,km,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jp,km,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,km,lo,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ip,km,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jp,km,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,km,lp,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,in,jo,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,in,ko,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,in,jp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,in,kp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,in,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,p,io,jn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jn,ko,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,o,ip,jn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jn,kp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jn,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,p,io,kn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,p,jo,kn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,o,ip,kn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,o,jp,kn,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kn,lm,op->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,io,jp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,io,kp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,io,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,n,ip,jo,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jo,kp,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jo,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,n,ip,ko,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,n,jp,ko,lm->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,ko,lm,np->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,ip,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jp,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kp,lm,no->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,in,jo,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,in,jo,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,in,jo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,in,jp,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,in,ko,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,in,ko,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,in,jp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,in,kp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,in,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,in,jp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,in,kp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,in,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,io,jn,kp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,io,jn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,io,jn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ip,jn,ko->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jn,ko,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jn,ko,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ip,jn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jn,kp,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jn,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ip,jn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jn,kp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jn,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,io,jp,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,io,kn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,io,kn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('l,m,ip,jo,kn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jo,kn,lp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jo,kn,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ip,kn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jp,kn,lo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kn,lo,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ip,kn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jp,kn,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kn,lp,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,io,jp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,io,kp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,io,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,m,ip,jo,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jo,kp,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jo,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,m,ip,ko,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,m,jp,ko,ln->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,ko,ln,mp->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,ip,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jp,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kp,ln,mo->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,io,jp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,io,kp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,io,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('k,l,ip,jo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jo,kp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jo,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,l,ip,ko,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,l,jp,ko,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,ko,lp,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('j,k,ip,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,k,jp,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('i,j,kp,lo,mn->ijklmnop', dm, dm, P, P, P)
        expected_eighth += contract('m,n,o,p,ij,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,n,o,p,ij,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,n,o,p,ij,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,o,p,ij,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,o,p,ij,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,o,p,ij,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,p,ij,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,p,ij,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,p,ij,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,p,ij,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,o,ij,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,o,ij,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,o,ij,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,o,ij,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,n,ij,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('m,n,o,p,ik,jl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,n,o,p,ik,jm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,n,o,p,ik,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,o,p,ik,jn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,o,p,ik,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,o,p,ik,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,p,ik,jo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,p,ik,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,p,ik,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,p,ik,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,o,ik,jp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,o,ik,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,o,ik,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,o,ik,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,n,ik,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('m,n,o,p,il,jk->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,n,o,p,im,jk->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,n,o,p,jk,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,o,p,in,jk->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,o,p,jk,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,o,p,jk,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,p,io,jk->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,p,jk,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,p,jk,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,p,jk,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('l,m,n,o,ip,jk->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,o,jk,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,o,jk,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,o,jk,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,n,jk,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,n,o,p,il,jm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,n,o,p,il,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,o,p,il,jn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,o,p,il,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,o,p,il,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,p,il,jo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,p,il,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,p,il,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,p,il,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,o,il,jp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,o,il,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,o,il,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,o,il,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,n,il,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,n,o,p,im,jl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,n,o,p,jl,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,o,p,in,jl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,o,p,jl,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,o,p,jl,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,p,io,jl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,p,jl,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,p,jl,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,p,jl,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,m,n,o,ip,jl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,o,jl,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,o,jl,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,o,jl,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,n,jl,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,n,o,p,im,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,n,o,p,jm,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,o,p,in,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,o,p,jn,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,o,p,kl,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,p,io,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,p,jo,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,p,kl,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,p,kl,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,m,n,o,ip,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,m,n,o,jp,kl->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,o,kl,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,o,kl,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,n,kl,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,o,p,im,jn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,o,p,im,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,o,p,im,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,p,im,jo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,p,im,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,p,im,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,p,im,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,o,im,jp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,o,im,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,o,im,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,o,im,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,n,im,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,o,p,in,jm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,o,p,jm,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,o,p,jm,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,p,io,jm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,p,jm,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,p,jm,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,p,jm,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,n,o,ip,jm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,o,jm,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,o,jm,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,o,jm,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,n,jm,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,o,p,in,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,o,p,jn,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,o,p,km,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,p,io,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,p,jo,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,p,km,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,p,km,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,n,o,ip,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,n,o,jp,km->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,o,km,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,o,km,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,n,km,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,o,p,in,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,o,p,jn,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,o,p,kn,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,p,io,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,p,jo,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,p,ko,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,p,lm,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,n,o,ip,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,n,o,jp,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,n,o,kp,lm->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,o,lm,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,n,lm,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,p,in,jo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,p,in,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,p,in,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,p,in,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,o,in,jp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,o,in,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,o,in,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,o,in,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,m,in,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,p,io,jn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,p,jn,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,p,jn,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,p,jn,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,o,ip,jn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,o,jn,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,o,jn,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,o,jn,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,m,jn,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,p,io,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,p,jo,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,p,kn,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,p,kn,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,o,ip,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,o,jp,kn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,o,kn,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,o,kn,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,m,kn,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,p,io,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,p,jo,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,p,ko,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,p,ln,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,o,ip,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,o,jp,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,o,kp,ln->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,o,ln,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,m,ln,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,p,io,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,p,jo,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,p,ko,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,p,lo,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,o,ip,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,o,jp,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,o,kp,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,o,lp,mn->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,l,mn,op->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,n,io,jp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,n,io,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,n,io,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,n,io,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,m,io,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,n,ip,jo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,n,jo,kp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,n,jo,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,n,jo,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,m,jo,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,l,m,n,ip,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,l,m,n,jp,ko->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,n,ko,lp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,n,ko,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,m,ko,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,m,n,ip,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,m,n,jp,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,m,n,kp,lo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,n,lo,mp->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,m,lo,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,n,ip,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,n,jp,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,n,kp,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,n,lp,mo->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,l,mo,np->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('j,k,l,m,ip,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,k,l,m,jp,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,l,m,kp,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,m,lp,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('i,j,k,l,mp,no->ijklmnop', dm, dm, dm, dm, P, P)
        expected_eighth += contract('k,l,m,n,o,p,ij->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,l,m,n,o,p,ik->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,l,m,n,o,p,jk->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,k,m,n,o,p,il->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,k,m,n,o,p,jl->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,m,n,o,p,kl->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,k,l,n,o,p,im->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,k,l,n,o,p,jm->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,l,n,o,p,km->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,n,o,p,lm->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,k,l,m,o,p,in->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,k,l,m,o,p,jn->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,l,m,o,p,kn->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,m,o,p,ln->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,o,p,mn->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,k,l,m,n,p,io->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,k,l,m,n,p,jo->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,l,m,n,p,ko->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,m,n,p,lo->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,n,p,mo->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,m,p,no->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('j,k,l,m,n,o,ip->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,k,l,m,n,o,jp->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,l,m,n,o,kp->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,m,n,o,lp->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,n,o,mp->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,m,o,np->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,m,n,op->ijklmnop', dm, dm, dm, dm, dm, dm, P)
        expected_eighth += contract('i,j,k,l,m,n,o,p->ijklmnop', dm, dm, dm, dm, dm, dm, dm, dm)
        expected_list.append(expected_eighth)

    return expected_list

def compute_expected_dm_DSTT_larger_ld(DSTT_list, order, stateSize, expected_dict):
    # R is (_, order - 1, state_size) for latentDim=1, or
    #      (_, order - 1, latentDim, state_size) for larger latent dimension. (deleted time index)
    # This is for a variable direction between all orders
    # Expected dict is the list of E[dx/dy] for each order
    dm = np.zeros(stateSize)

    if order >= 1: # Still use STM for first order
        dm += np.einsum('ia,a->i', DSTT_list[0], expected_dict['mean']['1'])

    if order >= 2:
        dm += 1 / 2 * np.einsum('iab,ab->i', DSTT_list[1], expected_dict['mean']['2'])

    if order >= 3:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")

    if order >= 4:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")

    return dm

def compute_expected_dm_xDSTT(DSTT_list, order, stateSize, Rs, expected_list, latentDim):
    # This is for a variable direction between all orders
    # Expected list is the list of E[dx] for each order
    dm = np.zeros(stateSize)

    if order >= 1: # Still use STM for first order
        dm += np.einsum('ia,a->i', DSTT_list[0], expected_list[0])

    if order >= 2:
        R2 = Rs[0, :]
        dm += 1 / 2 * np.einsum('iab,a,b,ab->i', DSTT_list[1], R2, R2, expected_list[1])

    if order >= 3:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")
        R3 = Rs[1,:]
        dm += 1/6 * np.einsum('iabc,a,b,c,abc->i',DSTT_list[2], R3, R3, R3, expected_list[2])

    if order >= 4:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")
        R4 = Rs[2,:]
        dm += 1/24 * np.einsum('iabcd,a,b,c,d,abcd->i',DSTT_list[3], R4, R4, R4, R4, expected_list[3])

    return dm

def compute_covariance_DSTT_larger_ld(DSTT_list, expected_dict, dm_kp1, order):
    # R is (_, order - 1, state_size) for latentDim=1, or
    #      (_, order - 1, latentDim, state_size) for larger latent dimension. (deleted time index)
    # This is for a variable direction between all orders
    # Expected dict is the list of E[dx/dy] for each order
    P_prop = -np.einsum('i,j->ij', dm_kp1, dm_kp1)

    if order >= 1:  # still use STM for first order
        phi = DSTT_list[0]
        term_1 = np.einsum('ia,je,ae->ij', phi, phi, expected_dict['cov']['11'])
        P_prop += term_1

    if order >= 2:
        psi_2 = DSTT_list[1]
        term_p1q1 = 0.5 * np.einsum('ia,jef,aef->ij', phi, psi_2, expected_dict['cov']['12'], optimize=True)
        term_p2q1 = 0.5 * np.einsum('iab,je,eab->ij', psi_2, phi, expected_dict['cov']['21'], optimize=True)
        term_p2q2 = 0.25 * np.einsum('iab,jef,abef->ij', psi_2, psi_2, expected_dict['cov']['22'],
                                  optimize=True)
        P_prop += term_p1q1 + term_p2q1 + term_p2q2

    if order >= 3:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")
    if order >= 4:
        raise NotImplementedError("4th order not implemented yet - need larger latent dimension")

    return P_prop

def compute_covariance_xDSTT(DSTT_list, expected_list, dm_kp1, order, Rs, latentDim):
    # Rs is order x stateSize
    # DSTTs have already been rotated - just need to rotate the E[x] terms
    P_prop = -np.einsum('i,j->ij', dm_kp1, dm_kp1)

    if order >= 1:  # still use STM for first order
        phi = DSTT_list[0]
        term_1 = np.einsum('ia,je,ae->ij', phi, phi, expected_list[1])
        P_prop += term_1

    if order >= 2:
        # print("phi shape: ", phi.shape)
        R2 = Rs[0, :]
        # print("R2 shape: ", R2.shape)
        psi_2 = DSTT_list[1]
        # print("psi_2 shape: ", psi_2.shape)
        term_2 = 0.5 * np.einsum('ia,jef,e,f,aef->ij', phi, psi_2, R2, R2, expected_list[2], optimize=True)
        # print("term_2 shape", term_2.shape)
        term_5 = 0.5 * np.einsum('iab,a,b,je,abe->ij', psi_2, R2, R2, phi, expected_list[2], optimize=True)
        # print("term_5 shape", term_5.shape)
        term_6 = 0.25 * np.einsum('iab,jef,a,b,e,f,abef->ij', psi_2, psi_2, R2, R2, R2, R2, expected_list[3],
                                  optimize=True)
        # print("term_6 shape", term_6.shape)
        P_prop += term_2 + term_5 + term_6


    if order >= 3:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")
        R3 = Rs[1,:]
        psi_3 = DSTT_list[2]
        term_3 = (1 / 6) * np.einsum('ia,jefg,e,f,g,aefg->ij', phi, psi_3, R3, R3, R3, expected_list[3], optimize=True)
        term_7 = (1 / 12) * np.einsum('iab,jefg,a,b,e,f,g,abefg->ij', psi_2, psi_3, R2, R2, R3, R3, R3, expected_list[4], optimize=True)
        term_9 = (1 / 6) * np.einsum('iabc,a,b,c,je,abce->ij', psi_3, R3, R3, R3, phi, expected_list[3], optimize=True)
        term_10 = (1 / 12) * np.einsum('iabc,jef,a,b,c,e,f,abcef->ij', psi_3, psi_2, R3, R3, R3, R2, R2, expected_list[4], optimize=True)
        term_11 = (1 / 36) * np.einsum('iabc,jefg,a,b,c,e,f,g,abcefg->ij', psi_3, psi_3, R3, R3, R3, R3, R3, R3, expected_list[5], optimize=True)
        P_prop += term_3 + term_7 + term_9 + term_10 + term_11

    if order >= 4:
        raise NotImplementedError("3rd order not implemented yet - need larger latent dimension")
        R4 = Rs[2,:]
        psi_4 = DSTT_list[3]
        term_4 = (1 / 24) * np.einsum('ia,jefgh,e,f,g,h,aefgh->ij', phi, psi_4, R4, R4, R4, R4, expected_list[4])
        term_8 = (1 / 48) * np.einsum('iab,jefgh,a,b,e,f,g,h,abefgh->ij', psi_2, psi_4, R2, R2, R4, R4, R4, R4, expected_list[5])
        term_12 = (1 / 144) * np.einsum('iabc,jefgh,a,b,c,e,f,g,h,abcefgh->ij', psi_3, psi_4, R3, R3, R3, R4, R4, R4, R4, expected_list[6])
        term_13 = (1 / 24) * np.einsum('iabcd,a,b,c,d,je,abcde->ij', psi_4, R4, R4, R4, R4, phi, expected_list[4])
        term_14 = (1 / 48) * np.einsum('iabcd,jef,a,b,c,d,e,f,abcdef->ij', psi_4, psi_2, R4, R4, R4, R4, R2, R2, expected_list[5])
        term_15 = (1 / 144) * np.einsum('iabcd,jefg,a,b,c,d,e,f,g,abcdefg->ij', psi_4, psi_3, R4, R4, R4, R4, R3, R3, R3, expected_list[6])
        term_16 = (1 / 576) * np.einsum('iabcd,jefgh,a,b,c,d,e,f,g,h,abcdefgh->ij', psi_4, psi_4, R4, R4, R4, R4, R4, R4, R4, R4, expected_list[7])
        P_prop += term_4 + term_8 + term_12 + term_13 + term_14 + term_15 + term_16

    return P_prop

def compute_expected_dm_DSTT(DSTT_list, order, stateSize, R,  expected_list):
    dm = np.zeros(stateSize)

    if order >= 1: # Still use STM for first order
        dm += np.einsum('ia,a->i', DSTT_list[0], expected_list[0])

    if order >= 2:
        dm += 1/2 * np.einsum('iab,a,b,ab->i',DSTT_list[1], R, R, expected_list[1])

    if order >= 3:
        dm += 1/6 * np.einsum('iabc,a,b,c,abc->i',DSTT_list[2], R, R, R, expected_list[2])

    if order >= 4:
        dm += 1/24 * np.einsum('iabcd,a,b,c,d,abcd->i',DSTT_list[3], R, R, R, R, expected_list[3])

    # # from Spencer's DSTT paper, Eqns. 55 - 61
    # dm = np.zeros(stateSize)
    # expected_dy = compute_expected_y(order, R, dm_k, P)
    #
    # if order >= 1: # still use STM for first order
    #     tmp1 = np.einsum('ia,a->i', DSTT_list[0], dm_k)
    #     dm += np.einsum('ia,a->i', DSTT_list[0], dm_k)
    #
    #     if order >= 2:
    #         psi_2 = DSTT_list[1][:,0,0]
    #         E_y2 = expected_dy[1]
    #         tmp2 = 1 / 2 * E_y2 * psi_2
    #         tmp2a = 1 / 2 * ( R @ dm_k ** 2 + R.T @ P @ R )  * psi_2
    #         dm += 1 / 2 * E_y2 * psi_2
    #
    #         if order >= 3:
    #             psi_3 = DSTT_list[2][:,0,0,0]
    #             E_y3 = expected_dy[2]
    #             tmp3 = 1 / 6 * E_y3 * psi_3
    #             dm += 1 / 6 * E_y3 * psi_3
    #
    #             if order >= 4:
    #                 # TODO not implemented yet
    #                 psi_4 = DSTT_list[3][:,0,0,0,0]
    #                 E_y4 = expected_dy[3]
    #                 tmp4 = 1 / 24 * E_y4 * psi_4
    #                 dm += 1 / 24 * E_y4 * psi_4

    # if order >= 1: # still use STM for first order
    #     dm += np.einsum('ia,a->i', DSTT_list[0], dm_k)
    #
    #     if order >= 2:
    #         psi_2 = DSTT_list[1]
    #         dm += 1 / 2 * psi_2 * E_dy**2
    #         dm += 1 / 2 * psi_2 * sigma_xi_k**2
    #
    #         # Check
    #         check1 = 1 / 2 * psi_2 * E_dy**2 + 1 / 2 * psi_2 * sigma_xi_k**2
    #         check2 = 1/2 * psi_2 * (np.einsum('i,i->', R, dm_k)**2 + sigma_xi_k**2)
    #         tmp = check2 - check1
    #         if order >= 3:
    #             psi_3 = DSTT_list[2]
    #             dm += 1/6 * psi_3 * E_dy**3
    #             dm += 1/6 * psi_3 * R @ P @ dm_k
    #             check = 0
    #
    #             if order >= 4:
    #                 psi_4 = DSTT_list[3]
    #                 dm += 1 / 8 * psi_4 * sigma_xi_k**4

    return dm


def compute_covariance_DSTT(DSTT_list, expected_list, dm_kp1, order, R):
    P_prop = -np.einsum('i,j->ij', dm_kp1, dm_kp1)
    # TODO
    # P_prop = np.zeros((dm_k.shape[0], dm_k.shape[0]))

    if order >= 1:  # still use STM for first order
        phi = DSTT_list[0]
        term_1 = np.einsum('ia,je,ae->ij', phi, phi, expected_list[1])
        P_prop += term_1

    if order >= 2:
        psi_2 = DSTT_list[1]
        term_2 = 0.5 * np.einsum('ia,jef,e,f,aef->ij', phi, psi_2, R, R, expected_list[2], optimize=True)
        term_5 = 0.5 * np.einsum('iab,a,b,je,abe->ij', psi_2, R, R, phi, expected_list[2], optimize=True)
        term_6 = 0.25 * np.einsum('iab,jef,a,b,e,f,abef->ij', psi_2, psi_2, R, R, R, R, expected_list[3], optimize=True)
        P_prop += term_2 + term_5 + term_6

    if order >= 3:
        psi_3 = DSTT_list[2]
        term_3 = (1 / 6) * np.einsum('ia,jefg,e,f,g,aefg->ij', phi, psi_3, R, R, R, expected_list[3], optimize=True)
        term_7 = (1 / 12) * np.einsum('iab,jefg,a,b,e,f,g,abefg->ij', psi_2, psi_3, R, R, R,R,R, expected_list[4], optimize=True)
        term_9 = (1 / 6) * np.einsum('iabc,a,b,c,je,abce->ij', psi_3, R, R, R, phi, expected_list[3], optimize=True)
        term_10 = (1 / 12) * np.einsum('iabc,jef,a,b,c,e,f,abcef->ij', psi_3, psi_2, R, R, R, R, R, expected_list[4], optimize=True)
        term_11 = (1 / 36) * np.einsum('iabc,jefg,a,b,c,e,f,g,abcefg->ij', psi_3, psi_3, R, R, R, R, R, R, expected_list[5], optimize=True)
        P_prop += term_3 + term_7 + term_9 + term_10 + term_11

    if order >= 4:
        # TODO need to update the indices
        psi_4 = DSTT_list[3]
        term_4 = (1 / 24) * np.einsum('ia,jefgh,efgh,ae->ij', phi, psi_4, R, R, R, R, expected_list[4])
        term_8 = (1 / 48) * np.einsum('iab,jefgh,ab,efgh,abefgh->ij', psi_2, psi_4, R, R, R, R, expected_list[5])
        term_12 = (1 / 144) * np.einsum('iabc,jefgh,abc,efgh,abcefgh->ij', psi_3, psi_4, R, R, R, R, expected_list[6])
        term_13 = (1 / 24) * np.einsum('iabcd,abcd,je,abcde->ij', psi_4, R, R, R, R, phi, expected_list[4])
        term_14 = (1 / 48) * np.einsum('iabcd,jef,abcd,ef,abcdef->ij', psi_4, psi_2, R, R, expected_list[5])
        term_15 = (1 / 144) * np.einsum('iabcd,jefg,abcd,efg,abcdefg->ij', psi_4, psi_3, R, R, R, expected_list[6])
        term_16 = (1 / 576) * np.einsum('iabcd,jefgh,abcd,efgh,abcdefgh->ij', psi_4, psi_4, R, R, R, R, expected_list[7])
        P_prop += term_4 + term_8 + term_12 + term_13 + term_14 + term_15 + term_16


    # if order >= 1:  # still use STM for first order
    #     # p=1, q=1
    #     phi_first = DSTT_list[0]
    #     P_prop += np.einsum('ip,ja,pa->ij', phi_first, phi_first, expected_list[1])
    #
    # if order >= 2:
    #     DSTT_2 = DSTT_list[1]
    #     P_prop += 1/2 * sigma_xi_k**4 * np.einsum('i,j->ij', DSTT_2, DSTT_2)
    #
    # if order >= 3:  #
    #     DSTT_3 = DSTT_list[2]
    #     P_prop += 1/2 * sigma_xi_k**4 * np.einsum('i,j->ij', DSTT_3, DSTT_1)
    #     P_prop += 1/2 * sigma_xi_k**4 * np.einsum('i,j->ij', DSTT_1, DSTT_3)
    #     P_prop += 5/12 * sigma_xi_k**6 * np.einsum('i,j->ij', DSTT_3, DSTT_3)
    #
    # if order >= 4:
    #     DSTT_4 = DSTT_list[3]
    #     P_prop += 1/4 * sigma_xi_k**6 * np.einsum('i,j->ij', DSTT_4, DSTT_2)
    #     P_prop += 1/4 * sigma_xi_k**6 * np.einsum('i,j->ij', DSTT_2, DSTT_4)
    #     P_prop += 1/6 * sigma_xi_k**8 * np.einsum('i,j->ij', DSTT_4, DSTT_4)

    return P_prop

def compute_expected_dm(phi_list, expected_list, order, stateSize):
    # dm = np.zeros((stateSize,1))
    dm = np.zeros(stateSize)

    if order >= 1:
        dm += np.einsum('ia,a->i', phi_list[0], expected_list[0])

        if order >= 2:
            dm += 1 / 2 * np.einsum('iab,ab->i', phi_list[1], expected_list[1])

            if order >= 3:
                dm += 1 / 6 * np.einsum('iabc,abc->i', phi_list[2], expected_list[2])

                if order >= 4:
                    dm += 1 / 24 * np.einsum('iabcd,abcd->i', phi_list[3], expected_list[3])

    return dm

def compute_covariance(phi_list, expected_list, dm, order, stateSize):
    P_prop = -np.einsum('i,j->ij', dm, dm)

    if order >= 1:
        # p=1, q=1
        phi_first = phi_list[0]
        P_prop += np.einsum('ip,ja,pa->ij', phi_first, phi_first, expected_list[1])

    if order >= 2:
        phi_second = phi_list[1]
        # p=2, q=1
        P_prop += 1 / 2 * np.einsum('ipq,ja,pqa->ij', phi_second, phi_first, expected_list[2])

        # p=1, q=2
        P_prop += 1 / 2 * np.einsum('ip,jab,pab->ij', phi_first, phi_second, expected_list[2])

        # p=2, q=2
        P_prop += 1 / 4 * np.einsum('ipq,jab,pqab->ij', phi_second, phi_second, expected_list[3])

    if order >= 3:  #
        phi_third = phi_list[2]
        # p=3, q=1
        P_prop += 1 / 6 * np.einsum('ipqr,ja,pqra->ij', phi_third, phi_first, expected_list[3])
        # p=1, q=3
        P_prop += 1 / 6 * np.einsum('ip,jabc,pabc->ij', phi_first, phi_third, expected_list[3])
        # p=2, q=3
        P_prop += 1 / 12 * np.einsum('ipq,jabc,pqabc->ij', phi_second, phi_third, expected_list[4])
        # p=3, q=2
        P_prop += 1 / 12 * np.einsum('ipqr,jab,pqrab->ij', phi_third, phi_second, expected_list[4])
        # p=3, q=3
        P_prop += 1 / 36 * np.einsum('ipqr,jabc,pqrabc->ij', phi_third, phi_third, expected_list[5])

    if order >= 4:
        phi_fourth = phi_list[3]
        # p=4, q=1
        P_prop += 1 / 24 * np.einsum('ipqrs,ja,pqrsa->ij', phi_fourth, phi_first, expected_list[4])
        # p=1, q=4
        P_prop += 1 / 24 * np.einsum('ip,jabcd,pabcd->ij', phi_first, phi_fourth, expected_list[4])
        # p=4, q=2
        P_prop += 1 / 48 * np.einsum('ipqrs,jab,pqrsab->ij', phi_fourth, phi_second, expected_list[5])
        # p=2, q=4
        P_prop += 1 / 48 * np.einsum('ipq,jabcd,pqabcd->ij', phi_second, phi_fourth, expected_list[5])
        # p=4, q=3
        P_prop += 1 / 144 * np.einsum('ipqrs,jabc,pqrsabc->ij', phi_fourth, phi_third, expected_list[6])
        # p=3, q=4
        P_prop += 1 / 144 * np.einsum('ipqr,jabcd,pqrabcd->ij', phi_third, phi_fourth, expected_list[6])
        # p=4, q=4
        P_prop += 1 / 576 * np.einsum('ipqrs,jabcd,pqrsabcd->ij', phi_fourth, phi_fourth, expected_list[7])

    return P_prop

