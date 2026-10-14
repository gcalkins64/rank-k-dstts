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

