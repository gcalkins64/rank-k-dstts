# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for aerodynamics models

import numpy as np
from .params_planet import gravParams

def aerodynamicAccelsBeta(params, rho, V, t=None):
    if 'NonDim' in params['eomTag']:
        # Redimensionalized time
        t_check = t * np.sqrt(gravParams[params['planetName']]['R_eq'] / gravParams[params['planetName']]['g_surf'])
    else:
        t_check = t

    if 'beta' in params:  # Single ballistic coefficient vehicle
        L = 0.5 * rho * V**2 / params['beta'] * params['LD']
        D = 0.5 * rho * V**2 / params['beta']
    elif 'betas' in params:  # Drag modulation, drag skirt deployment
        if t_check < params['t_deploy']:
            beta = params['betas'][0]
        else:
            beta = params['betas'][1]
        # print(beta)
        L = 0.5 * rho * V**2 / beta * params['LD']
        D = 0.5 * rho * V**2 / beta
    return L, D