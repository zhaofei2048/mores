"""
Author: Fei Zhao
Create: 2023-08-19
"""

import numpy as np

def fresnel_coefficients(theta_i, epsilon_r):
    """
    Calculates Fresnel reflection & transmission coefficients
    INPUT:
        theta_i (scalar): incident angle (deg)
        epsilon_r (scalar): relative permittivity of lower medium 2 to upper medium 1
    OUTPUT:
        rv: =Erv/Eiv, Fresnel v-polarized reflection coefficient
        rh: =Erh/Eih, Fresnel h-polarized reflection coefficient
        tv: =Etv/Eiv, Fresnel v-polarized transmission coefficient
        th: =Eth/Eih, Fresnel h-polarized transmission coefficient
    """
    c1 = np.cos(np.deg2rad(theta_i))
    s1 = np.sin(np.deg2rad(theta_i))
    rv = (epsilon_r * c1 - np.sqrt(epsilon_r - s1**2)) / (epsilon_r * c1 + np.sqrt(epsilon_r - s1**2))
    rh = (c1 - np.sqrt(epsilon_r - s1**2)) / (c1 + np.sqrt(epsilon_r - s1**2))
    tv = (1+rv) / np.sqrt(epsilon_r)
    th = 1+rh
    
    return rv, rh, tv, th


def refraction_angle(theta_i, epsilon_r):
    """
    This equivalent refraction angle reduces to general refraction angle when medium 2 is lossless.
    INPUT:
        theta_i: incident angle (deg)
        epsilon_r: relative permittivity of lower medium 2 to upper medium 1
    OUTPUT:
        theta_t: the refraction angle (deg)
    """
    thi = np.deg2rad(theta_i)
    p = np.sqrt(epsilon_r - np.sin(thi)**2).real
    cosphi = p / np.sqrt(p**2 + np.sin(thi)**2)
    theta_t = np.rad2deg(np.arccos(cosphi))
    
    return theta_t


def TransmissionAngle(theta_i, epsilon_r):
        """
        Ganesh
        """
        theta_i = np.deg2rad(theta_i)
        mu1 = np.cos(theta_i)
        n = np.sqrt(epsilon_r)
        
        mu2 = np.sqrt(1.0 - ((1.0 - mu1**2) / n**2)).real
        theta_t = np.arccos(mu2)
        theta_t = np.rad2deg(theta_t)
#        theta_t = np.arcsin(l1.ri.real * np.sin(l2.theta_i) / l2.ri.real)
        return theta_t


def critical_angle(epsilon_r):
    """get the critical angle (total reflection)

    Args:
        epsilon_r: relative dielectric constant of medium 2 to medium 1 
            (only when epsilon_r < 1, the critical angle has meaning).
    
    Returns:
        theta_c: critical angle (deg)
    """
    nr = np.real(np.sqrt(epsilon_r))
    if nr >= 1.0:
        theta_c = 90
    else:
        theta_c = np.rad2deg(np.arcsin(nr))
    
    return theta_c


def transmissivity(theta_i, epsilon_r):
     """transmissivity is defined as the ratio of the transmitted power to incident power

    Args:
        theta_i (scalar): incident angle (deg)
        epsilon_r (scalar): relative permittivity of lower medium 2 to upper medium 1

    Returns:
        Tv: v-polarized transmissivity
        Th: h-polarized transmissivity
     """
     rv, rh, _, _ = fresnel_coefficients(theta_i, epsilon_r)
     Tv = 1 - np.abs(rv)**2
     Th = 1 - np.abs(rh)**2

     return Tv, Th
