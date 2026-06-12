"""Lunar surface roughness model utils

Author: Fei Zhao
Create: 2024-05-04
"""
import numpy as np


def rms_height_from_slope(s0_deg, l_F, wavelength, Hurst=0.73, delta_x0=1.0):
    """
    Calculate RMS height from slope s0_deg at referenced scale delta_x0

    Args:
        s0_deg: slopes (deg) at referenced scale of delta_x0
        l_F: correlation length factor: corr_len = l_F * wavelength
        wavelength: wavelength (m)
        Hurst: Hurst exponenial, [0.5, 1.0], default is 0.73
        delta_x0: the reference scale of s0_deg, default is 1.0 m
    
    Returns:
        rms_height: RMS height (m)
    """
    s0 = np.tan(np.deg2rad(s0_deg))
    L = l_F * wavelength    # correlation length

    # Campbell, 2009, "Scale-Dependent Surface Roughness Behavior and Its
    #            Impact on Empirical Models for Radar Backscatter"
    # rms_height = np.exp(-2*Hurst)/np.sqrt(2) * s0 * (L/delta_x0)**(Hurst-1) * L

    # Iodice, A.; Di Martino, G.; Di Simone, A.; Riccio, D.; Ruello, G. Electromagnetic Scattering from Fractional Brownian Motion Surfaces via the Small Slope Approximation. Fractal and Fractional 2023, 7, doi:10.3390/fractalfract7050387.
    rms_height = 1/np.sqrt(2) * s0 * (L/delta_x0)**(Hurst-1) * L 

    return rms_height

