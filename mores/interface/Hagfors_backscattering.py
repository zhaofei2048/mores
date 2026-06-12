"""Hagfors backscatternig model

Ref: Eq. (57) in T. Hagfors, "Remote Probing of the Moon by Infrared and Microwave Emissions and by Radar," Radio Science, vol. 5, no. 2, pp. 189-227, 1970/02/01 1970, doi: https://doi.org/10.1029/RS005i002p00189.

Author: Fei Zhao
Create: 2024-04-28
"""
import numpy as np
from .fresnel import fresnel_coefficients


def sigma0_pp_Hagfors(theta_i, epsr, C):
    """Hagfors backscattering model
    
    Args:
        theta_i: incidence angle (deg) in surface scattering coordinates; can be array
        epsr: complex relative dielectric constant (er - 1j*eri)
        C: Hagfors roughness parameters, should >> 6.25 for the requirement of geometrical optics (GO) approximation;
            C increases as the roughness of the surface decreases.
    
    Returns:
        sigma0_pp: backscattering coefficients
    """
    rho0, _, _, _ = fresnel_coefficients(0, epsr)
    cs = np.cos(np.deg2rad(theta_i))
    s = np.sin(np.deg2rad(theta_i))
    sigma0_pp = C * 0.5 * np.abs(rho0)**2 * (cs**4 + C * s**2)**(-1.5)

    return sigma0_pp