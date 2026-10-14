# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for gravity models

from .params_planet import *
import numpy as np

## CONSTANTS
# G = 6.67430e-11 # m^3/(kg - s^2)

def invSquareGravity(planetName, alt):
    '''
    takes in geopotential alt and converts to geometric alt
    :param planetName:
    :param alt:
    :return:
    '''
    R_eq = gravParams[planetName].get('R_eq')
    mu = gravParams[planetName].get('mu')
    alt = alt * R_eq / (R_eq + alt)
    g = mu/(R_eq+alt)**2
    return g

def J2Gravity(planetName, alt):
    '''
    takes in geopotential alt and converts to geometric alt
    :param planetName:
    :param alt:
    :return:
    '''
    R_eq = gravParams[planetName].get('R_eq')
    J2 = gravParams[planetName].get('J2')
    mu = gravParams[planetName].get('mu')
    alt = alt*R_eq/(R_eq + alt)
    g_J2 = mu/((alt+R_eq)**2)*(1+J2*(R_eq/(alt+R_eq))**2*1.5)
    g = g_J2
    return g


def zonalJ2Gravity(planetName, r, phi):
    '''
    :param r: distance from the center of the planet to the vehicle
    :param phi: geo-centric latitude
    '''
    R_eq = gravParams[planetName].get('R_eq')
    # m = gravParams[planetName].get('mass')
    J2 = gravParams[planetName].get('J2')
    mu = gravParams[planetName].get('mu')
    gr = mu/r**2 * (1 + J2*(R_eq/r)**2 * (1.5 - 4.5*(np.sin(phi))**2))
    gphi = mu/r**2 * (J2*(R_eq/r)**2 * (3*np.sin(phi)*np.cos(phi)))
    return gr, gphi
