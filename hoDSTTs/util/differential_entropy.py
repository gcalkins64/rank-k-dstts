# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for differential entropy calculations

import numpy as np
import math
import sys
from scipy.integrate import solve_ivp
from .params_planet import *
def dynamics_partials_single_exp_nd(m_nd, params):
    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')
    mu = gravParams[params['planetName']].get('mu') # * G
    J2 = gravParams[params['planetName']].get('J2')
    Omega = rotParams[params['planetName']].get('Omega')
    rho_ref = params.get('rho_ref')
    beta = params.get('beta')
    LD = params.get('LD')
    H = params.get('H')
    sigma = params.get('sigma')
    h0 = 0
    rho0 = atmoParams['Uranus'].get('rho_surf')

    r_star, theta, phi, V_star, gamma, psi, rho_star = m_nd
    F = np.array(([[0, 0, 0, math.sqrt(Rp*g0)*math.sin(gamma), V_star*math.sqrt(Rp*g0)*math.cos(gamma), 0, 0], [-V_star*math.sqrt(Rp*g0)*math.sin(psi)*math.cos(gamma)/(Rp*r_star**2*math.cos(phi)), 0, V_star*math.sqrt(Rp*g0)*math.sin(phi)*math.sin(psi)*math.cos(gamma)/(Rp*r_star*math.cos(phi)**2), math.sqrt(Rp*g0)*math.sin(psi)*math.cos(gamma)/(Rp*r_star*math.cos(phi)), -V_star*math.sqrt(Rp*g0)*math.sin(gamma)*math.sin(psi)/(Rp*r_star*math.cos(phi)), V_star*math.sqrt(Rp*g0)*math.cos(gamma)*math.cos(psi)/(Rp*r_star*math.cos(phi)), 0], [-V_star*math.sqrt(Rp*g0)*math.cos(gamma)*math.cos(psi)/(Rp*r_star**2), 0, 0, math.sqrt(Rp*g0)*math.cos(gamma)*math.cos(psi)/(Rp*r_star), -V_star*math.sqrt(Rp*g0)*math.sin(gamma)*math.cos(psi)/(Rp*r_star), -V_star*math.sqrt(Rp*g0)*math.sin(psi)*math.cos(gamma)/(Rp*r_star), 0], [2*J2*mu*(1.5 - 4.5*math.sin(phi)**2)*math.sin(gamma)/(Rp**2*g0*r_star**5) + 12*J2*mu*math.sin(phi)*math.cos(gamma)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**5) + Omega**2*Rp*(math.sin(gamma)*math.cos(phi) - math.sin(phi)*math.cos(gamma)*math.cos(psi))*math.cos(phi) + 2*mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)*math.sin(gamma)/(Rp**2*g0*r_star**3), 0, 9.0*J2*mu*math.sin(gamma)*math.sin(phi)*math.cos(phi)/(Rp**2*g0*r_star**4) + 3*J2*mu*math.sin(phi)**2*math.cos(gamma)*math.cos(psi)/(Rp**2*g0*r_star**4) - 3*J2*mu*math.cos(gamma)*math.cos(phi)**2*math.cos(psi)/(Rp**2*g0*r_star**4) + Omega**2*Rp*r_star*(-math.sin(gamma)*math.sin(phi) - math.cos(gamma)*math.cos(phi)*math.cos(psi))*math.cos(phi) - Omega**2*Rp*r_star*(math.sin(gamma)*math.cos(phi) - math.sin(phi)*math.cos(gamma)*math.cos(psi))*math.sin(phi), -1.0*Rp*V_star*rho_ref*rho_star/beta, 3*J2*mu*math.sin(gamma)*math.sin(phi)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4) + Omega**2*Rp*r_star*(math.sin(gamma)*math.sin(phi)*math.cos(psi) + math.cos(gamma)*math.cos(phi))*math.cos(phi) - mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)*math.cos(gamma)/(Rp**2*g0*r_star**2), 3*J2*mu*math.sin(phi)*math.sin(psi)*math.cos(gamma)*math.cos(phi)/(Rp**2*g0*r_star**4) + Omega**2*Rp*r_star*math.sin(phi)*math.sin(psi)*math.cos(gamma)*math.cos(phi), -0.5*Rp*V_star**2*rho_ref/beta], [(-12*J2*mu*math.sin(gamma)*math.sin(phi)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**5) + Omega**2*Rp*(math.sin(gamma)*math.sin(phi)*math.cos(psi) + math.cos(gamma)*math.cos(phi))*math.cos(phi) + (2*J2*mu*(1.5 - 4.5*math.sin(phi)**2)/(Rp**2*g0*r_star**5) - V_star**2*g0/r_star**2 + 2*mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)/(Rp**2*g0*r_star**3))*math.cos(gamma))/(V_star*math.sqrt(Rp*g0)), 0, (-3*J2*mu*math.sin(gamma)*math.sin(phi)**2*math.cos(psi)/(Rp**2*g0*r_star**4) + 3*J2*mu*math.sin(gamma)*math.cos(phi)**2*math.cos(psi)/(Rp**2*g0*r_star**4) + 9.0*J2*mu*math.sin(phi)*math.cos(gamma)*math.cos(phi)/(Rp**2*g0*r_star**4) - Omega**2*Rp*r_star*(math.sin(gamma)*math.sin(phi)*math.cos(psi) + math.cos(gamma)*math.cos(phi))*math.sin(phi) + Omega**2*Rp*r_star*(math.sin(gamma)*math.cos(phi)*math.cos(psi) - math.sin(phi)*math.cos(gamma))*math.cos(phi) - 2*Omega*V_star*math.sqrt(Rp*g0)*math.sin(phi)*math.sin(psi))/(V_star*math.sqrt(Rp*g0)), (1.0*LD*Rp*V_star*rho_ref*rho_star*math.cos(sigma)/beta + 2*Omega*math.sqrt(Rp*g0)*math.sin(psi)*math.cos(phi) + 2*V_star*g0*math.cos(gamma)/r_star)/(V_star*math.sqrt(Rp*g0)) - (3*J2*mu*math.sin(gamma)*math.sin(phi)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4) + 0.5*LD*Rp*V_star**2*rho_ref*rho_star*math.cos(sigma)/beta + Omega**2*Rp*r_star*(math.sin(gamma)*math.sin(phi)*math.cos(psi) + math.cos(gamma)*math.cos(phi))*math.cos(phi) + 2*Omega*V_star*math.sqrt(Rp*g0)*math.sin(psi)*math.cos(phi) + (V_star**2*g0/r_star - mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)/(Rp**2*g0*r_star**2))*math.cos(gamma))/(V_star**2*math.sqrt(Rp*g0)), (3*J2*mu*math.sin(phi)*math.cos(gamma)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4) + Omega**2*Rp*r_star*(-math.sin(gamma)*math.cos(phi) + math.sin(phi)*math.cos(gamma)*math.cos(psi))*math.cos(phi) - (V_star**2*g0/r_star - mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)/(Rp**2*g0*r_star**2))*math.sin(gamma))/(V_star*math.sqrt(Rp*g0)), (-3*J2*mu*math.sin(gamma)*math.sin(phi)*math.sin(psi)*math.cos(phi)/(Rp**2*g0*r_star**4) - Omega**2*Rp*r_star*math.sin(gamma)*math.sin(phi)*math.sin(psi)*math.cos(phi) + 2*Omega*V_star*math.sqrt(Rp*g0)*math.cos(phi)*math.cos(psi))/(V_star*math.sqrt(Rp*g0)), 0.5*LD*Rp*V_star*rho_ref*math.cos(sigma)/(beta*math.sqrt(Rp*g0))], [(-12*J2*mu*math.sin(phi)*math.sin(psi)*math.cos(phi)/(Rp**2*g0*r_star**5*math.cos(gamma)) + Omega**2*Rp*math.sin(phi)*math.sin(psi)*math.cos(phi)/math.cos(gamma) - V_star**2*g0*math.sin(psi)*math.cos(gamma)*math.tan(phi)/r_star**2)/(V_star*math.sqrt(Rp*g0)), 0, (-3*J2*mu*math.sin(phi)**2*math.sin(psi)/(Rp**2*g0*r_star**4*math.cos(gamma)) + 3*J2*mu*math.sin(psi)*math.cos(phi)**2/(Rp**2*g0*r_star**4*math.cos(gamma)) - Omega**2*Rp*r_star*math.sin(phi)**2*math.sin(psi)/math.cos(gamma) + Omega**2*Rp*r_star*math.sin(psi)*math.cos(phi)**2/math.cos(gamma) - 2*Omega*V_star*math.sqrt(Rp*g0)*(-math.sin(phi)*math.cos(psi)*math.tan(gamma) - math.cos(phi)) + V_star**2*g0*(math.tan(phi)**2 + 1)*math.sin(psi)*math.cos(gamma)/r_star)/(V_star*math.sqrt(Rp*g0)), (1.0*LD*Rp*V_star*rho_ref*rho_star*math.sin(sigma)/(beta*math.cos(gamma)) - 2*Omega*math.sqrt(Rp*g0)*(-math.sin(phi) + math.cos(phi)*math.cos(psi)*math.tan(gamma)) + 2*V_star*g0*math.sin(psi)*math.cos(gamma)*math.tan(phi)/r_star)/(V_star*math.sqrt(Rp*g0)) - (3*J2*mu*math.sin(phi)*math.sin(psi)*math.cos(phi)/(Rp**2*g0*r_star**4*math.cos(gamma)) + 0.5*LD*Rp*V_star**2*rho_ref*rho_star*math.sin(sigma)/(beta*math.cos(gamma)) + Omega**2*Rp*r_star*math.sin(phi)*math.sin(psi)*math.cos(phi)/math.cos(gamma) - 2*Omega*V_star*math.sqrt(Rp*g0)*(-math.sin(phi) + math.cos(phi)*math.cos(psi)*math.tan(gamma)) + V_star**2*g0*math.sin(psi)*math.cos(gamma)*math.tan(phi)/r_star)/(V_star**2*math.sqrt(Rp*g0)), (3*J2*mu*math.sin(gamma)*math.sin(phi)*math.sin(psi)*math.cos(phi)/(Rp**2*g0*r_star**4*math.cos(gamma)**2) + 0.5*LD*Rp*V_star**2*rho_ref*rho_star*math.sin(gamma)*math.sin(sigma)/(beta*math.cos(gamma)**2) + Omega**2*Rp*r_star*math.sin(gamma)*math.sin(phi)*math.sin(psi)*math.cos(phi)/math.cos(gamma)**2 - 2*Omega*V_star*math.sqrt(Rp*g0)*(math.tan(gamma)**2 + 1)*math.cos(phi)*math.cos(psi) - V_star**2*g0*math.sin(gamma)*math.sin(psi)*math.tan(phi)/r_star)/(V_star*math.sqrt(Rp*g0)), (3*J2*mu*math.sin(phi)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4*math.cos(gamma)) + Omega**2*Rp*r_star*math.sin(phi)*math.cos(phi)*math.cos(psi)/math.cos(gamma) + 2*Omega*V_star*math.sqrt(Rp*g0)*math.sin(psi)*math.cos(phi)*math.tan(gamma) + V_star**2*g0*math.cos(gamma)*math.cos(psi)*math.tan(phi)/r_star)/(V_star*math.sqrt(Rp*g0)), 0.5*LD*Rp*V_star*rho_ref*math.sin(sigma)/(beta*math.sqrt(Rp*g0)*math.cos(gamma))], [Rp**2*V_star*rho0*math.sqrt(Rp*g0)*math.exp((-Rp*r_star + Rp + h0)/H)*math.sin(gamma)/(H**2*rho_ref), 0, 0, -Rp*rho0*math.sqrt(Rp*g0)*math.exp((-Rp*r_star + Rp + h0)/H)*math.sin(gamma)/(H*rho_ref), -Rp*V_star*rho0*math.sqrt(Rp*g0)*math.exp((-Rp*r_star + Rp + h0)/H)*math.cos(gamma)/(H*rho_ref), 0, 0]])
)
    return F

def trace_dynamics_partials_single_exp_nd(m_nd, params):
    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')
    mu = gravParams[params['planetName']].get('mu') # * G
    J2 = gravParams[params['planetName']].get('J2')
    Omega = rotParams[params['planetName']].get('Omega')
    Omega_star = Omega * (np.sqrt(gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get('g_surf')))
    rho_ref = params.get('rho_ref')
    beta = params.get('beta')
    beta_star = beta / (rho_ref * Rp)
    LD = params.get('LD')
    H = params.get('H')
    sigma = params.get('sigma')
    h0 = 0
    rho0 = atmoParams['Uranus'].get('rho_surf')

    r_star, theta, phi, V_star, gamma, psi, rho_star = m_nd

    # trace_F = (-1.0 * Rp * V_star * rho_ref * rho_star / beta +
    #            (3 * J2 * mu * math.sin(phi) * math.cos(gamma) * math.cos(phi) * math.cos(psi) / (Rp ** 2 * g0 * r_star ** 4) + Omega ** 2 * Rp * r_star * (-math.sin(gamma) * math.cos(phi) + math.sin(phi) * math.cos(gamma) * math.cos(psi)) * math.cos(phi) - (V_star ** 2 * g0 / r_star - mu * (J2 * (1.5 - 4.5 * math.sin(phi) ** 2) / r_star ** 2 + 1) / (Rp ** 2 * g0 * r_star ** 2)) * math.sin(gamma)) / (V_star * math.sqrt(Rp * g0))
    #            + (3 * J2 * mu * math.sin(phi) * math.cos(phi) * math.cos(psi) / (Rp ** 2 * g0 * r_star ** 4 * math.cos(gamma)) + Omega ** 2 * Rp * r_star * math.sin(phi) * math.cos(phi) * math.cos(psi) / math.cos(gamma) + 2 * Omega * V_star * math.sqrt(Rp * g0) * math.sin(psi) * math.cos(phi) * math.tan(gamma) + V_star ** 2 * g0 * math.cos(gamma) * math.cos(psi) * math.tan(phi) / r_star) / (V_star * math.sqrt(Rp * g0)))

    # Recomputed on 7/26
    trace_F = (-1.0*V_star*rho_star/beta_star
               + (3*J2*mu*math.sin(phi)*math.cos(gamma)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4) + Omega_star**2*r_star*(-math.sin(gamma)*math.cos(phi) + math.sin(phi)*math.cos(gamma)*math.cos(psi))*math.cos(phi) - (V_star**2/r_star - mu*(J2*(1.5 - 4.5*math.sin(phi)**2)/r_star**2 + 1)/(Rp**2*g0*r_star**2))*math.sin(gamma))/V_star
               + (3*J2*mu*math.sin(phi)*math.cos(phi)*math.cos(psi)/(Rp**2*g0*r_star**4*math.cos(gamma)) + Omega_star**2*r_star*math.sin(phi)*math.cos(phi)*math.cos(psi)/math.cos(gamma) + 2*Omega_star*V_star*math.sin(psi)*math.cos(phi)*math.tan(gamma) + V_star**2*math.cos(gamma)*math.cos(psi)*math.tan(phi)/r_star)/V_star)

    return trace_F

def trace_dynamics_partials_single_exp_dim(m, params):
    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')
    mu = gravParams[params['planetName']].get('mu') #* G
    J2 = gravParams[params['planetName']].get('J2')
    Omega = rotParams[params['planetName']].get('Omega')
    rho_ref = params.get('rho_ref')
    beta = params.get('beta')
    LD = params.get('LD')
    H = params.get('H')
    sigma = params.get('sigma')
    h0 = 0
    rho0 = atmoParams['Uranus'].get('rho_surf')

    r, theta, phi, V, gamma, psi, rho = m

    trace_F_d = (-1.0*V*rho/(beta*g0)
               + (3*J2*Rp**2*mu*math.sin(phi)*math.cos(gamma)*math.cos(phi)*math.cos(psi)/(g0*r**4) + Omega**2*r*(-math.sin(gamma)*math.cos(phi) + math.sin(phi)*math.cos(gamma)*math.cos(psi))*math.cos(phi) - (V**2/r - mu*(J2*Rp**2*(1.5 - 4.5*math.sin(phi)**2)/r**2 + 1)/(g0*r**2))*math.sin(gamma))/V
               + (3*J2*Rp**2*mu*math.sin(phi)*math.cos(phi)*math.cos(psi)/(g0*r**4*math.cos(gamma)) + Omega**2*r*math.sin(phi)*math.cos(phi)*math.cos(psi)/math.cos(gamma) + 2*Omega*V*math.sin(psi)*math.cos(phi)*math.tan(gamma) + V**2*math.cos(gamma)*math.cos(psi)*math.tan(phi)/r)/V)

    return trace_F_d

def lin_entropy_ode(t, X, mus_nd, params, ii):
    if params['atmoTag'] == 'single_exp':
        # F = dynamics_partials_single_exp_nd(mus_nd[ii], params)
        trace_F = trace_dynamics_partials_single_exp_nd(mus_nd[ii], params)
    else:
        raise ValueError('Only single exponential atmosphere supported for now')
    # H_dot = np.trace(F)
    H_dot = trace_F
    return H_dot

def lin_entropy_ode_dim(t, X, mus, params, ii):
    if params['atmoTag'] == 'single_exp':
        # F = dynamics_partials_single_exp_nd(mus_nd[ii], params)
        trace_F_d = trace_dynamics_partials_single_exp_dim(mus[ii], params)
    else:
        raise ValueError('Only single exponential atmosphere supported for now')
    # H_dot = np.trace(F)
    H_dot = trace_F_d
    return H_dot

def get_differential_entropy(ts, mus_nd, Ps, params):
    # get linearized entropy over the trajectory
    ts_nd = ts / np.sqrt(gravParams[params['planetName']].get('R_eq') / gravParams[params['planetName']].get('g_surf'))
    H_lin_0 = 0.5*np.log(np.linalg.det(2*np.pi*np.e*Ps[0,:,:]))  # set initial guess to be nonlinear initial entropy
    print(H_lin_0)
    # H_lin_0 = 0
    H_lin = np.zeros(len(ts))
    H_lin[0] = H_lin_0
    for ii in range(len(ts)-1):
        sol = solve_ivp(lambda t, X: lin_entropy_ode(t, X, mus_nd, params, ii), y0 = np.array([0]), t_span=[ts_nd[ii], ts_nd[ii+1]], method='RK45', rtol=1e-7, atol=1e-8)
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


