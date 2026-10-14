# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for atmosphere models

import numpy as np
import math as m
import matplotlib.pyplot as plt
from .params_planet import * # type: ignore
from pyaudi import gdual_double as gdual # type: ignore

def exponentialAtmosphere(planetName, alt):
    """
    alt must be in m
    """
    if 'h_ranges' in atmoParams[planetName]: 
        h_ranges = atmoParams[planetName].get('h_ranges')
        for ii in range(len(h_ranges)):
            if alt < h_ranges[ii]:
                # print(ii)
                rho_ref = atmoParams[planetName].get('rho_surf')[ii]
                beta = atmoParams[planetName].get('beta')[ii]
                H = 1/beta
                break
    else:
        rho_ref = atmoParams[planetName].get('rho_surf')
        H = atmoParams[planetName].get('H')

    h_ref = 0
    rho = rho_ref * np.exp((h_ref - alt) / H)

    return rho

def exponentialAtmosphereGdual(planetName, alt):
    """
    alt must be in m
    """
    if 'h_ranges' in atmoParams[planetName]:
        h_ranges = atmoParams[planetName].get('h_ranges')
        for ii in range(len(h_ranges)):
            if alt.constant_cf < h_ranges[ii]:
                # print(ii)
                rho_ref = atmoParams[planetName].get('rho_surf')[ii]
                beta = atmoParams[planetName].get('beta')[ii]
                H = 1/beta
                break
    else:
        rho_ref = atmoParams[planetName].get('rho_surf')
        H = atmoParams[planetName].get('H')

    h_ref = 0
    rho = rho_ref * np.exp((h_ref - alt) / H)

    return rho

def polynomialAtmosphereUranus(h):
    """
    alt must be in km
    """
    c1 = atmoParams['Uranus'].get('c1')
    c2 = atmoParams['Uranus'].get('c2')
    c3 = atmoParams['Uranus'].get('c3')
    c4 = atmoParams['Uranus'].get('c4')
    c5 = atmoParams['Uranus'].get('c5')
    c6 = atmoParams['Uranus'].get('c6')
    c7 = atmoParams['Uranus'].get('c7')
    c8 = atmoParams['Uranus'].get('c8')
    c9 = atmoParams['Uranus'].get('c9')

    numerator = c1 + c3 * h + c5 * h ** 2 + c7 * h ** 3 + c9 * h ** 4
    denominator = 1 + c2 * h + c4 * h ** 2 + c6 * h ** 3 + c8 * h ** 4
    ln_rho = numerator / denominator
    rho = np.exp(ln_rho)

    return rho

def exponentialAtmosphereUranusSingle(alt):
    rho0 = atmoParams['Uranus'].get('rho_surf')
    H = atmoParams['Uranus'].get('H')

    h_ref = 0
    rho = rho0 * np.exp((h_ref - alt) / H)

    return rho

# def interpolateAtmosphere(alt, hs, rhos):
#     rho = np.interp(alt, hs, rhos)
#     return np.where(rho < 0, 0, rho)

def interpolateAtmosphere(alt, hs, rhos):
    if isinstance(alt, gdual):
        return _interpolateAtmosphereGdual(alt, hs, rhos)
    else:
        rho = np.interp(alt, hs, rhos)
        return np.where(rho < 0, 0, rho)

def _interpolateAtmosphereGdual(alt, hs, rhos):
    alt_val = alt.constant_cf
    n = len(hs)

    idx = np.searchsorted(hs, alt_val)
    idx = min(max(idx, 1), n - 1)  # clamp so idx-1 and idx are always valid indices

    h0, h1 = hs[idx - 1], hs[idx]
    rho0, rho1 = rhos[idx - 1], rhos[idx]

    if alt_val <= h0:
        rho = gdual(float(rho0))
    elif alt_val >= h1:
        rho = gdual(float(rho1))
    else:
        slope = (rho1 - rho0) / (h1 - h0)
        rho = rho0 + slope * (alt - h0)

    if rho.constant_cf < 0:
        return gdual(0.0)
    return rho

def generateGRAMAtmoFunction(h_GRAM, rho_GRAM):
    def GRAMAtmosphere(alt):
        '''
        interpolator for GRAM density
        '''
        rho = np.interp(alt, h_GRAM, rho_GRAM)
        return np.where(rho < 0, 0, rho)
        # return rho  # WARNING: only use this option when comparing density charts, not to simulate

    return GRAMAtmosphere

def evalKLEDensity(alt_d, params):
    # From Claude
    # Should work with gduals for autodifferentiation
    rho_bar = interpolateAtmosphere(alt_d, params['hs_interp'], params['rho_interp'])
    d_KLE = params.get('d_KLE', 0)
    drho = 0.0
    for i in range(d_KLE):
        lam_i = params['KLE_eigvals'][i]
        phi_i_d = interpolateKLEEigfunction(alt_d, params['hs_interp'], params['KLE_eigfuncs_interp'][:,i])
        Y_i = params['_Y_current'][i]   # should be zero!
        drho = drho + np.sqrt(lam_i) * phi_i_d * Y_i
    return rho_bar + drho

def interpolateKLEEigfunction(alt, hs, phi_i):
    if isinstance(alt, gdual):
        return _interpolateKLEEigfunctionGdual(alt, hs, phi_i)
    else:
        return np.interp(alt, hs, phi_i)


def _interpolateKLEEigfunctionGdual(alt, hs, phi_i):
    alt_val = alt.constant_cf
    n = len(hs)

    idx = np.searchsorted(hs, alt_val)
    idx = min(max(idx, 1), n - 1)  # clamp so idx-1 and idx are always valid indices

    h0, h1 = hs[idx - 1], hs[idx]
    p0, p1 = phi_i[idx - 1], phi_i[idx]

    if alt_val <= h0:
        return gdual(float(p0))
    elif alt_val >= h1:
        return gdual(float(p1))
    else:
        slope = (p1 - p0) / (h1 - h0)
        return p0 + slope * (alt - h0)