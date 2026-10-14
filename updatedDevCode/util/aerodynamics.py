# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for aerodynamics models

import numpy as np

def aerodynamicAccelsBeta(params, rho, V):
    L = 0.5 * rho * V**2 / params['beta'] * params['LD']
    D = 0.5 * rho * V**2 / params['beta']
    return L, D