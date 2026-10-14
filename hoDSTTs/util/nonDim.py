# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions to convert between non-dimensional and dimensional states and covariances

import numpy as np
from .params_planet import gravParams # type: ignore

def convertFromNonDim(Xs_nd, params):
    Xs = Xs_nd.copy()
    Xs[0] = Xs_nd[0] * gravParams[params.get('planetName')].get('R_eq')
    Xs[3] = Xs_nd[3] * np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    return Xs
def convertToNonDim(Xs, params):
    Xs_nd = Xs.copy()
    Xs_nd[0] = Xs[0] / gravParams[params.get('planetName')].get('R_eq')
    Xs_nd[3] = Xs[3] / np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    return Xs_nd

def convertFromNonDim2D(Xs_nd, params):
    # X is V, gamma, z, theta
    Xs = Xs_nd.copy()
    Xs[2] = Xs_nd[2] * gravParams[params.get('planetName')].get('R_eq')
    Xs[0] = Xs_nd[0] * np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    return Xs
def convertToNonDim2D(Xs, params):
    # X is V, gamma, z, theta
    Xs_nd = Xs.copy()
    Xs_nd[2] = Xs[2] / gravParams[params.get('planetName')].get('R_eq')
    Xs_nd[0] = Xs[0] / np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    return Xs_nd

def convertFromNonDim2DDens(Xs_nd, params):
    # X is V, gamma, z, theta, rho
    Xs = Xs_nd.copy()
    Xs[2] = Xs_nd[2] * gravParams[params.get('planetName')].get('R_eq')
    Xs[0] = Xs_nd[0] * np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    Xs[4] = Xs_nd[4] * params.get('rho_ref')
    return Xs
def convertToNonDim2DDens(Xs, params):
    # X is V, gamma, z, theta, rhos
    Xs_nd = Xs.copy()
    Xs_nd[2] = Xs[2] / gravParams[params.get('planetName')].get('R_eq')
    Xs_nd[0] = Xs[0] / np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    Xs_nd[4] = Xs[4] / params.get('rho_ref')
    return Xs_nd

def convertFromNonDimDens(Xs_nd, params):
    Xs = Xs_nd.copy()
    Xs[0] = Xs_nd[0] * gravParams[params.get('planetName')].get('R_eq')
    Xs[3] = Xs_nd[3] * np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    Xs[6] = Xs_nd[6] * params.get('rho_ref')
    return Xs

def convertToNonDimDens(Xs, params):
    Xs_nd = Xs.copy()
    Xs_nd[0] = Xs[0] / gravParams[params.get('planetName')].get('R_eq')
    Xs_nd[3] = Xs[3] / np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    Xs_nd[6] = Xs[6] / params.get('rho_ref')
    return Xs_nd

def convertFromNonDimDensCov(P_nd, params):
    P = P_nd.copy()
    r_nd_factor = gravParams[params.get('planetName')].get('R_eq')
    V_nd_factor = np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    rho_nd_factor = params.get('rho_ref')

    # Row 1 - r
    P[0,0] = P_nd[0,0] * r_nd_factor * r_nd_factor
    P[0,1] = P_nd[0,1] * r_nd_factor
    P[0,2] = P_nd[0,2] * r_nd_factor
    P[0,3] = P_nd[0,3] * r_nd_factor * V_nd_factor
    P[0,4] = P_nd[0,4] * r_nd_factor 
    P[0,5] = P_nd[0,5] * r_nd_factor 
    P[0,6] = P_nd[0,6] * r_nd_factor * rho_nd_factor
    # Row 2 - theta
    P[1,0] = P_nd[1,0] * r_nd_factor
    P[1,1] = P_nd[1,1] 
    P[1,2] = P_nd[1,2] 
    P[1,3] = P_nd[1,3] * V_nd_factor
    P[1,4] = P_nd[1,4] 
    P[1,5] = P_nd[1,5] 
    P[1,6] = P_nd[1,6] * rho_nd_factor
    # Row 3 - phi
    P[2,0] = P_nd[2,0] * r_nd_factor
    P[2,1] = P_nd[2,1] 
    P[2,2] = P_nd[2,2] 
    P[2,3] = P_nd[2,3] * V_nd_factor
    P[2,4] = P_nd[2,4] 
    P[2,5] = P_nd[2,5] 
    P[2,6] = P_nd[2,6] * rho_nd_factor
    # Row 4 - V
    P[3,0] = P_nd[3,0] * V_nd_factor * r_nd_factor
    P[3,1] = P_nd[3,1] * V_nd_factor 
    P[3,2] = P_nd[3,2] * V_nd_factor 
    P[3,3] = P_nd[3,3] * V_nd_factor * V_nd_factor
    P[3,4] = P_nd[3,4] * V_nd_factor 
    P[3,5] = P_nd[3,5] * V_nd_factor 
    P[3,6] = P_nd[3,6] * V_nd_factor * rho_nd_factor
    # Row 5 - gamma
    P[4,0] = P_nd[4,0] * r_nd_factor
    P[4,1] = P_nd[4,1] 
    P[4,2] = P_nd[4,2] 
    P[4,3] = P_nd[4,3] * V_nd_factor
    P[4,4] = P_nd[4,4] 
    P[4,5] = P_nd[4,5] 
    P[4,6] = P_nd[4,6] * rho_nd_factor
    # Row 5 - psi
    P[5,0] = P_nd[5,0] * r_nd_factor
    P[5,1] = P_nd[5,1] 
    P[5,2] = P_nd[5,2] 
    P[5,3] = P_nd[5,3] * V_nd_factor
    P[5,4] = P_nd[5,4] 
    P[5,5] = P_nd[5,5] 
    P[5,6] = P_nd[5,6] * rho_nd_factor
    # Row 6 - rho
    P[6,0] = P_nd[6,0] * rho_nd_factor * r_nd_factor
    P[6,1] = P_nd[6,1] * rho_nd_factor 
    P[6,2] = P_nd[6,2] * rho_nd_factor 
    P[6,3] = P_nd[6,3] * rho_nd_factor * V_nd_factor
    P[6,4] = P_nd[6,4] * rho_nd_factor 
    P[6,5] = P_nd[6,5] * rho_nd_factor 
    P[6,6] = P_nd[6,6] * rho_nd_factor * rho_nd_factor

    return P

def convertToNonDimDensCov(P, params):
    # state is r, theta, phi, V, gamma, psi, rho 
    # cov is 
    # [r,r r,theta r,phi r,V r,gamma r,psi r,rho]
    # [theta,r theta,theta theta,phi theta,V theta,gamma theta,psi theta,rho]
    # [phi,r phi,theta phi,phi phi,V, phi,gamma phi,psi phi,rho]
    # [V,r V,theta V,phi V,V V,gamma V,psi V,rho]
    # [gamma,r gamma,theta gamma,phi gamma,V, gamma,gamma gamma,psi gamma,rho]
    # [psi,r psi,theta psi,phi psi,V psi,gamma psi,psi psi,rho]
    # [rho,r rho,theta rho,phi rho,V rho,gamma rho,psi rho,rho]\

    P_nd = P.copy()
    r_nd_factor = gravParams[params.get('planetName')].get('R_eq')
    V_nd_factor = np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    rho_nd_factor = params.get('rho_ref')

    # Row 1 - r
    P_nd[0,0] = P[0,0] / r_nd_factor / r_nd_factor
    P_nd[0,1] = P[0,1] / r_nd_factor 
    P_nd[0,2] = P[0,2] / r_nd_factor 
    P_nd[0,3] = P[0,3] / r_nd_factor / V_nd_factor
    P_nd[0,4] = P[0,4] / r_nd_factor 
    P_nd[0,5] = P[0,5] / r_nd_factor 
    P_nd[0,6] = P[0,6] / r_nd_factor / rho_nd_factor
    # Row 2 - theta
    P_nd[1,0] = P[1,0] / r_nd_factor
    P_nd[1,1] = P[1,1] 
    P_nd[1,2] = P[1,2] 
    P_nd[1,3] = P[1,3] / V_nd_factor
    P_nd[1,4] = P[1,4] 
    P_nd[1,5] = P[1,5] 
    P_nd[1,6] = P[1,6] / rho_nd_factor
    # Row 3 - phi
    P_nd[2,0] = P[2,0] / r_nd_factor
    P_nd[2,1] = P[2,1] 
    P_nd[2,2] = P[2,2] 
    P_nd[2,3] = P[2,3] / V_nd_factor
    P_nd[2,4] = P[2,4] 
    P_nd[2,5] = P[2,5] 
    P_nd[2,6] = P[2,6] / rho_nd_factor
    # Row 4 - V
    P_nd[3,0] = P[3,0] / V_nd_factor / r_nd_factor
    P_nd[3,1] = P[3,1] / V_nd_factor 
    P_nd[3,2] = P[3,2] / V_nd_factor 
    P_nd[3,3] = P[3,3] / V_nd_factor / V_nd_factor
    P_nd[3,4] = P[3,4] / V_nd_factor 
    P_nd[3,5] = P[3,5] / V_nd_factor 
    P_nd[3,6] = P[3,6] / V_nd_factor / rho_nd_factor
    # Row 5 - gamma
    P_nd[4,0] = P[4,0] / r_nd_factor
    P_nd[4,1] = P[4,1] 
    P_nd[4,2] = P[4,2] 
    P_nd[4,3] = P[4,3] / V_nd_factor
    P_nd[4,4] = P[4,4] 
    P_nd[4,5] = P[4,5] 
    P_nd[4,6] = P[4,6] / rho_nd_factor
    # Row 5 - psi
    P_nd[5,0] = P[5,0] / r_nd_factor
    P_nd[5,1] = P[5,1] 
    P_nd[5,2] = P[5,2] 
    P_nd[5,3] = P[5,3] / V_nd_factor
    P_nd[5,4] = P[5,4] 
    P_nd[5,5] = P[5,5] 
    P_nd[5,6] = P[5,6] / rho_nd_factor
    # Row 6 - rho
    P_nd[6,0] = P[6,0] / rho_nd_factor / r_nd_factor
    P_nd[6,1] = P[6,1] / rho_nd_factor 
    P_nd[6,2] = P[6,2] / rho_nd_factor 
    P_nd[6,3] = P[6,3] / rho_nd_factor / V_nd_factor
    P_nd[6,4] = P[6,4] / rho_nd_factor 
    P_nd[6,5] = P[6,5] / rho_nd_factor 
    P_nd[6,6] = P[6,6] / rho_nd_factor / rho_nd_factor

    return P_nd


def convertFromNonDim2DDensCov(P_nd, params):
    P = P_nd.copy()
    z_nd_factor = gravParams[params.get('planetName')].get('R_eq')
    V_nd_factor = np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    rho_nd_factor = params.get('rho_ref')

    # Row 1 - V
    P[0,0] = P_nd[0,0] * V_nd_factor * V_nd_factor
    P[0,1] = P_nd[0,1] * V_nd_factor
    P[0,2] = P_nd[0,2] * V_nd_factor * z_nd_factor
    P[0,3] = P_nd[0,3] * V_nd_factor
    P[0,4] = P_nd[0,4] * V_nd_factor * rho_nd_factor
    # Row 2 - gamma
    P[1,0] = P_nd[1,0] * V_nd_factor
    P[1,1] = P_nd[1,1]
    P[1,2] = P_nd[1,2] * z_nd_factor
    P[1,3] = P_nd[1,3]
    P[1,4] = P_nd[1,4] * rho_nd_factor
    # Row 3 - z
    P[2,0] = P_nd[2,0] * V_nd_factor * z_nd_factor
    P[2,1] = P_nd[2,1] * z_nd_factor
    P[2,2] = P_nd[2,2] * z_nd_factor * z_nd_factor
    P[2,3] = P_nd[2,3] * z_nd_factor
    P[2,4] = P_nd[2,4] * z_nd_factor * rho_nd_factor
    # Row 4 - theta
    P[3,0] = P_nd[3,0] * V_nd_factor
    P[3,1] = P_nd[3,1]
    P[3,2] = P_nd[3,2] * z_nd_factor
    P[3,3] = P_nd[3,3]
    P[3,4] = P_nd[3,4] * rho_nd_factor
    # Row 5 - rho
    P[4,0] = P_nd[4,0] * V_nd_factor * rho_nd_factor
    P[4,1] = P_nd[4,1] * rho_nd_factor
    P[4,2] = P_nd[4,2] * z_nd_factor * rho_nd_factor
    P[4,3] = P_nd[4,3] * rho_nd_factor
    P[4,4] = P_nd[4,4] * rho_nd_factor * rho_nd_factor

    return P

def convertToNonDim2DDensCov(P, params):
    # state is V, gamma, z, theta, rho
    # cov is
    # [V,V V,gamma V,z V,theta V,rho]
    # [gamma,V gamma,gamma gamma,z gamma,theta gamma,rho]
    # [z,V z,gamma z,z z,theta z,rho]
    # [theta,V theta,gamma theta,z theta,theta theta,rho]
    # [rho,V rho,gamma rho,z rho,theta rho,rho]

    P_nd = P.copy()
    z_nd_factor = gravParams[params.get('planetName')].get('R_eq')
    V_nd_factor = np.sqrt(gravParams[params.get('planetName')].get('g_surf') * gravParams[params.get('planetName')].get('R_eq'))
    rho_nd_factor = params.get('rho_ref')

    # Row 1 - V
    P_nd[0,0] = P[0,0] / V_nd_factor / V_nd_factor
    P_nd[0,1] = P[0,1] / V_nd_factor
    P_nd[0,2] = P[0,2] / V_nd_factor / z_nd_factor
    P_nd[0,3] = P[0,3] / V_nd_factor
    P_nd[0,4] = P[0,4] / V_nd_factor / rho_nd_factor
    # Row 2 - gamma
    P_nd[1,0] = P[1,0] / V_nd_factor
    P_nd[1,1] = P[1,1]
    P_nd[1,2] = P[1,2] / z_nd_factor
    P_nd[1,3] = P[1,3]
    P_nd[1,4] = P[1,4] / rho_nd_factor
    # Row 3 - z
    P_nd[2,0] = P[2,0] / V_nd_factor / z_nd_factor
    P_nd[2,1] = P[2,1] / z_nd_factor
    P_nd[2,2] = P[2,2] / z_nd_factor / z_nd_factor
    P_nd[2,3] = P[2,3] / z_nd_factor
    P_nd[2,4] = P[2,4] / z_nd_factor / rho_nd_factor
    # Row 4 - theta
    P_nd[3,0] = P[3,0] / V_nd_factor
    P_nd[3,1] = P[3,1]
    P_nd[3,2] = P[3,2] / z_nd_factor
    P_nd[3,3] = P[3,3]
    P_nd[3,4] = P[3,4] / rho_nd_factor
    # Row 5 - rho
    P_nd[4,0] = P[4,0] / V_nd_factor / rho_nd_factor
    P_nd[4,1] = P[4,1] / rho_nd_factor
    P_nd[4,2] = P[4,2] / z_nd_factor / rho_nd_factor
    P_nd[4,3] = P[4,3] / rho_nd_factor
    P_nd[4,4] = P[4,4] / rho_nd_factor / rho_nd_factor

    return P_nd


