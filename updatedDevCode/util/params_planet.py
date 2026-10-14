# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses parameters for planets

import numpy as np

G = 6.67408e-11 # m3 kg-1 s-2

# Must set atmosphere parameters for each planet in input function
atmoParams = {
    'Earth': {},
    'Mars': {},
    'Neptune': {},
    'Uranus': {
        # 'rho_surf': 6.40934589e-03,
        # 'H': 5.47204083e+04,
    },
}

# src: https://en.wikipedia.org/wiki/Planetary_mass
gravParams = {
    'Venus': {
        'g_surf': 8.87, # m/s^2
        'R_eq': 6052e3, # m
        'J2': 4.46E-6, # []
        'mass': 4.867E24 # kg
    },
    'Earth': {
        'g_surf': 9.81,
        'R_eq': 6378e3,
        'J2': 1.038E-3,
        'mass': 5.972E24
    },
    'Mars': {
        'g_surf': 3.71,
        'R_eq': 3396e3,
        'J2': 1.960E-3,
        'mass': 6.417E23
    },
    'Neptune': {
        'g_surf': 11.15,
        'R_eq': 24764000,
        'J2': 3.411E-3,
        'mass': 6835099500000000.0 / G,
    },
    'Uranus': {  #  https://nssdc.gsfc.nasa.gov/planetary/factsheet/uranusfact.html, verified with Curtis
        'g_surf': 9.01,
        'R_eq': 25559e3,
        'J2': 3.334343E-3,
        'mass': 86.811*10**24,  # kg
        'mu': 5.7940*10**6*(10**3)**3  # m^3/s^2
    }
}


rotParams = {
    'Venus': {  # https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html
        'Omega': -5832.6**(-1)*(2*np.pi)/3600
    },
    'Earth': {  # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
        'Omega':  23.9345**(-1)*(2*np.pi)/3600
    },
    'Mars': {  # https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
        'Omega': 24.6229**(-1)*(2*np.pi)/3600
    },
    'Neptune': {  # https://nssdc.gsfc.nasa.gov/planetary/factsheet/neptunefact.html
        'Omega': 16.11**(-1)*(2*np.pi)/3600
    },
    'Uranus': {  # https://ntrs.nasa.gov/api/citations/20230017026/downloads/SciTech24_ECI_Trajectory_Paper_STRIVES.pdf
        'Omega': -1.012371e-4
    }
}






