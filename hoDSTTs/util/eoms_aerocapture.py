# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses the equations of motion for aerocapture

import numpy as np
from .atmospheres import *
from .gravity import zonalJ2Gravity, invSquareGravity, J2Gravity
from .params_planet import gravParams, rotParams, atmoParams
from .aerodynamics import *

def eoms3DOF(t, X, params):
    '''

    :param t: time (s)
    :param X: State vector [r, theta, phi, V, gamma, psi]


    :param planetName: string of planet name
    :return: time derivative of the state vector
    '''
    sigma, LD, beta, planetName = params['sigma'], params['LD'], params['beta'], params['planetName']

    # print(X)
    # X is [r, theta, phi, V, gamma, psi]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    # print(r)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]

    alt = X[0] - gravParams[planetName].get('R_eq')

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(planetName, r, phi)
    if params['gdualFlag'] == False:
        rho = exponentialAtmosphere(planetName, alt)
    else:
        rho = exponentialAtmosphereGdual(planetName, alt)
    Omega = rotParams[planetName].get('Omega')

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho, V)

    r_dot = V*np.sin(gamma)
    theta_dot = V*np.cos(gamma)*np.sin(psi) / (r*np.cos(phi))
    phi_dot = V*np.cos(gamma)*np.cos(psi) / r
    V_dot = -D - gr*np.sin(gamma) - gphi*np.cos(gamma)*np.cos(psi) + Omega**2 * r*np.cos(phi)*(np.sin(gamma) * np.cos(phi) - np.cos(gamma)*np.sin(phi)*np.cos(psi))
    gamma_dot = 1/V * (L*np.cos(sigma) + (V**2/r - gr)*np.cos(gamma) + gphi*np.sin(gamma)*np.cos(psi) + 2*Omega*V*np.cos(phi)*np.sin(psi) +
                       Omega**2*r*np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.cos(psi)*np.sin(phi)))
    # psi_dot = 1/V * (L*np.sin(sigma) / np.cos(gamma) + V**2/r*np.cos(gamma)*np.sin(psi)*np.tan(phi) + gphi*np.sin(psi)/np.cos(gamma)
    #                  - 2*Omega*V*(np.tan(gamma)*np.cos(psi)*np.cos(phi) - np.sin(phi))
    #                  + Omega**2*r/np.cos(gamma) *np.sin(psi)*np.sin(phi)*np.cos(phi))

    psi_dot = (1 / V) * (L * np.sin(sigma) / np.cos(gamma) - V ** 2 / r * np.cos(gamma) * np.cos(psi) * np.tan(phi)
                         - gphi * np.cos(psi) / np.cos(gamma)
                         + 2 * Omega * V * (np.tan(gamma) * np.cos(phi) * np.sin(psi) - np.sin(phi))
                         - Omega ** 2 * r * np.sin(phi) * np.cos(phi) * np.cos(psi) / np.cos(gamma))

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot])
    # print(dX)

    return dX

def eoms3DOFNonDim(t, X_nd, params):
    '''

    :param t: time (s)
    :param X: State vector [r, theta, phi, V, gamma, psi]


    :param planetName: string of planet name
    :return: time derivative of the state vector
    '''
    sigma, LD, beta, planetName, gdualFlag = params['sigma'], params['LD'], params['beta'], params['planetName'], params['gdualFlag']

    # X is [r, theta, phi, V, gamma, psi]
    r = X_nd[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X_nd[1]
    phi = X_nd[2]
    V = X_nd[3]
    gamma = X_nd[4]
    psi = X_nd[5]

    # Convert position back to dimensional space to compute alt
    alt_d = r * gravParams[planetName].get('R_eq') - gravParams[planetName].get('R_eq')

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(planetName, r * gravParams[planetName].get('R_eq'), phi)
    gr = gr / gravParams[planetName].get('g_surf')
    gphi = gphi / gravParams[planetName].get('g_surf')
    if gdualFlag:
        rho = util_atmospheres.exponentialAtmosphereGdual(planetName, alt_d)
    else:
        rho = util_atmospheres.exponentialAtmosphere(planetName, alt_d)
    Omega = rotParams[planetName].get('Omega')
    Omega = Omega * (np.sqrt(gravParams[planetName].get('R_eq') / gravParams[planetName].get('g_surf')))

    # Compute aerodynamics (THESE ARE ACCELERATIONS NOT FORCES)
    L, D = aerodynamicForcesBeta(LD, beta, rho, V * np.sqrt(gravParams[planetName].get('g_surf') * gravParams[planetName].get('R_eq')) )
    L = L / gravParams[planetName].get('g_surf')
    D = D / gravParams[planetName].get('g_surf')

    r_dot = V*np.sin(gamma)
    theta_dot = V*np.cos(gamma)*np.sin(psi) / (r*np.cos(phi))
    phi_dot = V*np.cos(gamma)*np.cos(psi) / r
    V_dot = -D - gr*np.sin(gamma) - gphi*np.cos(gamma)*np.cos(psi) + Omega**2 * r*np.cos(phi)*(np.sin(gamma) * np.cos(phi)
                                                                                               - np.cos(gamma)*np.sin(phi)*np.cos(psi))
    gamma_dot = 1/V * (L*np.cos(sigma) + (V**2/r - gr)*np.cos(gamma) + gphi*np.sin(gamma)*np.cos(psi) + 2*Omega*V*np.cos(phi)*np.sin(psi) +
                       Omega**2*r*np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.cos(psi)*np.sin(phi)))
    psi_dot = 1/V * (L*np.sin(sigma) / np.cos(gamma) + V**2/r*np.cos(gamma)*np.sin(psi)*np.tan(phi) + gphi*np.sin(psi)/np.cos(gamma)
                     - 2*Omega*V*(np.tan(gamma)*np.cos(psi)*np.cos(phi) - np.sin(phi))
                     + Omega**2*r/np.cos(gamma) *np.sin(psi)*np.sin(phi)*np.cos(phi))

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot])
    # print(dX)

    return dX

def eoms3DOFDens(t, X, params):
    # X is [r, theta, phi, V, gamma, psi, rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    rho = X[6]

    Rp = gravParams[params['planetName']].get('R_eq')
    h = X[0] - Rp
    sigma = params['sigma']

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r, phi)
    Omega = rotParams[params['planetName']].get('Omega')

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho, V)

    r_dot = V*np.sin(gamma)
    theta_dot = V*np.cos(gamma)*np.sin(psi) / (r*np.cos(phi))
    phi_dot = V*np.cos(gamma)*np.cos(psi) / r
    V_dot = -D - gr*np.sin(gamma) - gphi*np.cos(gamma)*np.cos(psi) + Omega**2 * r*np.cos(phi)*(np.sin(gamma) * np.cos(phi)
                                                                                               - np.cos(gamma)*np.sin(phi)*np.cos(psi))
    gamma_dot = 1/V * (L*np.cos(sigma) + (V**2/r - gr)*np.cos(gamma) + gphi*np.sin(gamma)*np.cos(psi) + 2*Omega*V*np.cos(phi)*np.sin(psi) +
                       Omega**2*r*np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.cos(psi)*np.sin(phi)))
    psi_dot = 1/V * (L*np.sin(sigma) / np.cos(gamma) + V**2/r*np.cos(gamma)*np.sin(psi)*np.tan(phi) + gphi*np.sin(psi)/np.cos(gamma)
                     - 2*Omega*V*(np.tan(gamma)*np.cos(psi)*np.cos(phi) - np.sin(phi))
                     + Omega**2*r/np.cos(gamma) *np.sin(psi)*np.sin(phi)*np.cos(phi))

    if params['atmoTag'] == 'single_exp':
        rho0 = atmoParams['Uranus'].get('rho_surf')
        H = atmoParams['Uranus'].get('H')
        h0 = 0
        drho_dt = -V*rho0*np.exp((Rp + h0 - r)/H)*np.sin(gamma)/H
    if params['atmoTag'] == 'polynomial':
        c1 = atmoParams['Uranus'].get('c1')
        c2 = atmoParams['Uranus'].get('c2')
        c3 = atmoParams['Uranus'].get('c3')
        c4 = atmoParams['Uranus'].get('c4')
        c5 = atmoParams['Uranus'].get('c5')
        c6 = atmoParams['Uranus'].get('c6')
        c7 = atmoParams['Uranus'].get('c7')
        c8 = atmoParams['Uranus'].get('c8')
        c9 = atmoParams['Uranus'].get('c9')
        drho_dt = V*((-c2 - c4*(-2*Rp + 2*r) - 3*c6*(-Rp + r)**2 - 4*c8*(-Rp + r)**3)*(c1 + c3*(-Rp + r) + c5*(-Rp + r)**2 + c7*(-Rp + r)**3 + c9*(-Rp + r)**4)/(c2*(-Rp + r) + c4*(-Rp + r)**2 + c6*(-Rp + r)**3 + c8*(-Rp + r)**4 + 1)**2 + (c3 + c5*(-2*Rp + 2*r) + 3*c7*(-Rp + r)**2 + 4*c9*(-Rp + r)**3)/(c2*(-Rp + r) + c4*(-Rp + r)**2 + c6*(-Rp + r)**3 + c8*(-Rp + r)**4 + 1))*np.exp((c1 + c3*(-Rp + r) + c5*(-Rp + r)**2 + c7*(-Rp + r)**3 + c9*(-Rp + r)**4)/(c2*(-Rp + r) + c4*(-Rp + r)**2 + c6*(-Rp + r)**3 + c8*(-Rp + r)**4 + 1))*np.sin(gamma)

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, drho_dt])
    # print(dX)

    return dX

def eoms3DOFNonDimDens(t, X, params):
    # X is [r, theta, phi, V, gamma, psi, rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    rho = X[6]

    rho_d = rho * params['rho_ref']
    Rp = gravParams[params['planetName']].get('R_eq')
    r_d = r * Rp
    alt_d = r_d - Rp
    V_d = V * np.sqrt(gravParams[params['planetName']].get('g_surf') * gravParams[params['planetName']].get('R_eq'))
    sigma = params['sigma']

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r_d, phi)
    gr = gr / gravParams[params['planetName']].get('g_surf')
    gphi = gphi / gravParams[params['planetName']].get('g_surf')
    Omega = rotParams[params['planetName']].get('Omega')
    Omega = Omega * (np.sqrt(gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get('g_surf')))

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho_d, V_d)
    L = L / gravParams[params['planetName']].get('g_surf')
    D = D / gravParams[params['planetName']].get('g_surf')

    r_dot = V * np.sin(gamma)
    theta_dot = V * np.cos(gamma) * np.sin(psi) / (r * np.cos(phi))
    phi_dot = V * np.cos(gamma) * np.cos(psi) / r
    V_dot = -D - gr * np.sin(gamma) - gphi * np.cos(gamma) * np.cos(psi) + Omega ** 2 * r * np.cos(phi) * (
                np.sin(gamma) * np.cos(phi)
                - np.cos(gamma) * np.sin(phi) * np.cos(psi))
    gamma_dot = 1 / V * (L * np.cos(sigma) + (V ** 2 / r - gr) * np.cos(gamma) + gphi * np.sin(gamma) * np.cos(
        psi) + 2 * Omega * V * np.cos(phi) * np.sin(psi) +
                         Omega ** 2 * r * np.cos(phi) * (
                                     np.cos(gamma) * np.cos(phi) + np.sin(gamma) * np.cos(psi) * np.sin(phi)))
    psi_dot = 1 / V * (L * np.sin(sigma) / np.cos(gamma) + V ** 2 / r * np.cos(gamma) * np.sin(psi) * np.tan(
        phi) + gphi * np.sin(psi) / np.cos(gamma)
                       - 2 * Omega * V * (np.tan(gamma) * np.cos(psi) * np.cos(phi) - np.sin(phi))
                       + Omega ** 2 * r / np.cos(gamma) * np.sin(psi) * np.sin(phi) * np.cos(phi))

    if params['atmoTag'] == 'single_exp':
        rho0 = atmoParams['Uranus'].get('rho_surf')
        H = atmoParams['Uranus'].get('H')
        h0 = 0
        drho_dt = -V * rho0 * np.exp((Rp + h0 - r_d) / H) * np.sin(gamma) / (H * params['rho_ref'] / Rp)
    if params['atmoTag'] == 'polynomial':
        rho_ref = params['rho_ref']
        c1 = atmoParams['Uranus'].get('c1')
        c2 = atmoParams['Uranus'].get('c2')
        c3 = atmoParams['Uranus'].get('c3')
        c4 = atmoParams['Uranus'].get('c4')
        c5 = atmoParams['Uranus'].get('c5')
        c6 = atmoParams['Uranus'].get('c6')
        c7 = atmoParams['Uranus'].get('c7')
        c8 = atmoParams['Uranus'].get('c8')
        c9 = atmoParams['Uranus'].get('c9')
        drho_dt = V * ((-c2 - c4 * (-2 * Rp + 2 * r_d) - 3 * c6 * (-Rp + r_d) ** 2 - 4 * c8 * (-Rp + r_d) ** 3) * (
                    c1 + c3 * (-Rp + r_d) + c5 * (-Rp + r_d) ** 2 + c7 * (-Rp + r_d) ** 3 + c9 * (-Rp + r_d) ** 4) / (
                                   c2 * (-Rp + r_d) + c4 * (-Rp + r_d) ** 2 + c6 * (-Rp + r_d) ** 3 + c8 * (
                                       -Rp + r_d) ** 4 + 1) ** 2 + (
                                   c3 + c5 * (-2 * Rp + 2 * r_d) + 3 * c7 * (-Rp + r_d) ** 2 + 4 * c9 * (-Rp + r_d) ** 3) / (
                                   c2 * (-Rp + r_d) + c4 * (-Rp + r_d) ** 2 + c6 * (-Rp + r_d) ** 3 + c8 * (
                                       -Rp + r_d) ** 4 + 1)) * np.exp(
            (c1 + c3 * (-Rp + r_d) + c5 * (-Rp + r_d) ** 2 + c7 * (-Rp + r_d) ** 3 + c9 * (-Rp + r_d) ** 4) / (
                        c2 * (-Rp + r_d) + c4 * (-Rp + r_d) ** 2 + c6 * (-Rp + r_d) ** 3 + c8 * (-Rp + r_d) ** 4 + 1)) * np.sin(
            gamma) / (rho_ref / Rp)

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, drho_dt])
    return dX

def eoms3DOFLnDens(t, X, params):
    # Uses natural log of density as a state instead of density
    # X is [r, theta, phi, V, gamma, psi, rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    ln_rho = X[6]
    rho = np.exp(ln_rho)

    Rp = gravParams[params['planetName']].get('R_eq')
    h = X[0] - Rp
    sigma = params['sigma']

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r, phi)
    Omega = rotParams[params['planetName']].get('Omega')

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho, V, t)

    r_dot = V*np.sin(gamma)
    theta_dot = V*np.cos(gamma)*np.sin(psi) / (r*np.cos(phi))
    phi_dot = V*np.cos(gamma)*np.cos(psi) / r
    V_dot = -D - gr*np.sin(gamma) - gphi*np.cos(gamma)*np.cos(psi) + Omega**2 * r*np.cos(phi)*(np.sin(gamma) * np.cos(phi) - np.cos(gamma)*np.sin(phi)*np.cos(psi))
    gamma_dot = 1/V * (L*np.cos(sigma) + (V**2/r - gr)*np.cos(gamma) + gphi*np.sin(gamma)*np.cos(psi) + 2*Omega*V*np.cos(phi)*np.sin(psi) +
                       Omega**2*r*np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.cos(psi)*np.sin(phi)))
    psi_dot = 1/V * (L*np.sin(sigma) / np.cos(gamma) + V**2/r*np.cos(gamma)*np.sin(psi)*np.tan(phi) + gphi*np.sin(psi)/np.cos(gamma)
                     - 2*Omega*V*(np.tan(gamma)*np.cos(psi)*np.cos(phi) - np.sin(phi))
                     + Omega**2*r/np.cos(gamma) *np.sin(psi)*np.sin(phi)*np.cos(phi))

    if params['atmoTag'] == 'single_exp':
        H = atmoParams[params['planetName']].get('H')
        dln_rho_dt = -V * np.sin(gamma) / H
    if params['atmoTag'] == 'polynomial':
        raise ValueError('Polynomial atmosphere not implemented for natural log density state')

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, dln_rho_dt])
    # print(dX)

    return dX

def eoms3DOFLnDensNonDim(t, X, params):
    # ln_rho is the final state
    # X is [r, theta, phi, V, gamma, psi, ln_rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    ln_rho = X[6]
    ln_rho_d = ln_rho * params['rho_ref']

    rho_d = np.exp(ln_rho_d)
    Rp = gravParams[params['planetName']].get('R_eq')
    r_d = r * Rp
    alt_d = r_d - Rp
    V_d = V * np.sqrt(gravParams[params['planetName']].get('g_surf') * gravParams[params['planetName']].get('R_eq'))
    sigma = params['sigma']

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r_d, phi)
    gr = gr / gravParams[params['planetName']].get('g_surf')
    gphi = gphi / gravParams[params['planetName']].get('g_surf')
    Omega = rotParams[params['planetName']].get('Omega')
    Omega = Omega * (np.sqrt(gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get('g_surf')))

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho_d, V_d, t)
    L = L / gravParams[params['planetName']].get('g_surf')
    D = D / gravParams[params['planetName']].get('g_surf')

    r_dot = V * np.sin(gamma)
    theta_dot = V * np.cos(gamma) * np.sin(psi) / (r * np.cos(phi))
    phi_dot = V * np.cos(gamma) * np.cos(psi) / r
    V_dot = -D - gr * np.sin(gamma) - gphi * np.cos(gamma) * np.cos(psi) + Omega ** 2 * r * np.cos(phi) * (
                np.sin(gamma) * np.cos(phi)
                - np.cos(gamma) * np.sin(phi) * np.cos(psi))
    gamma_dot = 1 / V * (L * np.cos(sigma) + (V ** 2 / r - gr) * np.cos(gamma) + gphi * np.sin(gamma) * np.cos(
        psi) + 2 * Omega * V * np.cos(phi) * np.sin(psi) +
                         Omega ** 2 * r * np.cos(phi) * (
                                     np.cos(gamma) * np.cos(phi) + np.sin(gamma) * np.cos(psi) * np.sin(phi)))
    psi_dot = 1 / V * (L * np.sin(sigma) / np.cos(gamma) + V ** 2 / r * np.cos(gamma) * np.sin(psi) * np.tan(
        phi) + gphi * np.sin(psi) / np.cos(gamma)
                       - 2 * Omega * V * (np.tan(gamma) * np.cos(psi) * np.cos(phi) - np.sin(phi))
                       + Omega ** 2 * r / np.cos(gamma) * np.sin(psi) * np.sin(phi) * np.cos(phi))

    if params['atmoTag'] == 'single_exp':
        H = atmoParams['Uranus'].get('H')
        d_ln_rho_dt = -V * np.sin(gamma) / (H/Rp) / params['rho_ref']
    if params['atmoTag'] == 'polynomial':
        raise ValueError('Polynomial atmosphere not implemented for natural log density state')

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, d_ln_rho_dt])
    return dX

def terminate(t, x, params):
    if x[3] <= 1000:  # if velocity is below 1000 m/s
        return 1
    if x[0] - gravParams[params['planetName']].get('R_eq') <= 0:  # if altitude is negative
        return 1
    # This is not getting altitude! This is getting the difference from the initial position!
    return x[0] - params['r0']
terminate.terminal = True  # Stop the integration when this event is triggered
terminate.direction = 1  # Trigger the event when the sign of the return value changes

def terminateTime(t, x, params):
    if x[3] <= 1000:  # if velocity is below 1000 m/s
        return 1
    elif x[0] - gravParams[params['planetName']].get('R_eq') <= 0:  # if altitude is negative
        return 1
    return x[0] - params['r0']
terminateTime.direction = 1  # Trigger the event when the sign of the return value changes

def eoms3DOFLnDensNonDim_CONSERVATIVE_eval(t, X, params):
    # X is [r, theta, phi, V, gamma, psi, ln_rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    ln_rho = X[6]

    ln_rho_d = ln_rho * params['rho_ref']

    rho_d = np.exp(ln_rho_d)
    R_eq = gravParams[params['planetName']].get('R_eq')
    r_d = r * R_eq
    alt_d = r_d - R_eq
    g_surf = gravParams[params['planetName']].get('g_surf')
    V_d = V * np.sqrt(g_surf * R_eq)

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r_d, phi)
    gr = gr / g_surf
    gphi = gphi / g_surf
    Omega = rotParams[params['planetName']].get('Omega')
    Omega_nd = Omega * (np.sqrt(R_eq / g_surf))

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho_d, V_d)
    L = L / g_surf
    D = D / g_surf

    r_dot = V * np.sin(gamma)
    theta_dot = V * np.cos(gamma) * np.sin(psi) / (r * np.cos(phi))
    phi_dot = V * np.cos(gamma) * np.cos(psi) / r
    V_dot = - gr * np.sin(gamma) - gphi * np.cos(gamma) * np.cos(psi) + Omega_nd ** 2 * r * np.cos(phi) * (np.sin(gamma) * np.cos(phi) - np.cos(gamma) * np.sin(phi) * np.cos(psi))
    gamma_dot = 1 / V * ( (V ** 2 / r - gr) * np.cos(gamma) + gphi * np.sin(gamma) * np.cos( psi) + 2 * Omega_nd * V * np.cos(phi) * np.sin(psi) + Omega_nd ** 2 * r * np.cos(phi) * (np.cos(gamma) * np.cos(phi) + np.sin(gamma) * np.cos(psi) * np.sin(phi)))
    psi_dot = 1 / V * (V ** 2 / r * np.cos(gamma) * np.sin(psi) * np.tan(phi) + gphi * np.sin(psi) / np.cos(gamma)- 2 * Omega_nd * V * (np.tan(gamma) * np.cos(psi) * np.cos(phi) - np.sin(phi))+ Omega_nd ** 2 * r / np.cos(gamma) * np.sin(psi) * np.sin(phi) * np.cos(phi))
    d_ln_rho_dt = 0. * ln_rho  # No change in density in the conservative case

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, d_ln_rho_dt])
    return dX

def eoms3DOFLnDensNonDim_AERODYNAMIC_eval(t, X, params):
    # X is [r, theta, phi, V, gamma, psi, ln_rho]
    r = X[0]  # distance from center of planet (NOT ALTITUDE)
    theta = X[1]
    phi = X[2]
    V = X[3]
    gamma = X[4]
    psi = X[5]
    ln_rho = X[6]
    ln_rho_d = ln_rho * params['rho_ref']
    sigma = params['sigma']

    rho_d = np.exp(ln_rho_d)
    R_eq = gravParams[params['planetName']].get('R_eq')
    r_d = r * R_eq
    alt_d = r_d - R_eq
    V_d = V * np.sqrt(gravParams[params['planetName']].get('g_surf') * gravParams[params['planetName']].get('R_eq'))

    # get gravity, density, and rotation rate
    gr, gphi = zonalJ2Gravity(params['planetName'], r_d, phi)
    g_surf = gravParams[params['planetName']].get('g_surf')
    gr = gr / g_surf
    gphi = gphi / g_surf
    Omega = rotParams[params['planetName']].get('Omega')
    Omega_nd = Omega * (np.sqrt(R_eq / g_surf))

    # Compute aerodynamics
    L, D = aerodynamicAccelsBeta(params, rho_d, V_d)
    L = L / g_surf
    D = D / g_surf

    r_dot = 0. * r
    theta_dot = 0. * theta
    phi_dot = 0. * phi
    V_dot = -D
    gamma_dot = 1 / V * (L * np.cos(sigma))
    psi_dot = 1 / V * (L * np.sin(sigma) / np.cos(gamma))

    if params['atmoTag'] == 'single_exp':
        H = atmoParams['Uranus'].get('H')
        d_ln_rho_dt = -V * np.sin(gamma) / (H / R_eq) / params['rho_ref']
    if params['atmoTag'] == 'polynomial':
        raise ValueError('Polynomial atmosphere not implemented for natural log density state')

    dX = np.array([r_dot, theta_dot, phi_dot, V_dot, gamma_dot, psi_dot, d_ln_rho_dt])
    return dX