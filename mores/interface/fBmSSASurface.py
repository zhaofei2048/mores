"""
Author: Fei Zhao
Create: 2025-02-02

Description:
    Fractional Brownian Motion rough surface.
    The non-coherent/diffuse scattering is calculated by fBm-SSA model (Iodice et al., 2023) and 
    coherent scattering is evaluated at the rms height at wavelength scale.
"""

import numpy as np
import scipy.constants as sci_const
from .QuasiSmoothSurface import QuasiSmoothSurface
from .fBmSSA_upward import Mue_upward_fBmSSA
from ..misc.lunar_surface_roughness_model import rms_height_from_slope


class fBmSSASurface(QuasiSmoothSurface):
    """
    Fractional Brownian Motion rough surface.
    The non-coherent/diffuse scattering is calculated by fBm-SSA model (Iodice et al., 2023) and 
    coherent scattering is evaluated at the rms height at wavelength scale (Campbell et al., 2009).
    """

    def __init__(self, epsilon_r, f, s0, H, shadow_flag=True):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: relative dielectric constant
            f: frequency (Hz)
            s0: RMS slope of the rough surface at the reference scale 1 m
            H: Hurst exponent of the rough surface
            shadow_flag: True (default) or False for including shadow effect
        OUTPUT:
            a rough surface instance
        """
        Lambda = sci_const.speed_of_light / f
        k = 2.0 * np.pi / Lambda
        s0_deg = np.rad2deg(np.arctan(s0))
        delta = rms_height_from_slope(s0_deg=s0_deg, l_F=1.0, wavelength=Lambda, Hurst=H)
        kdel = k * delta
        super(fBmSSASurface, self).__init__(epsilon_r, kdel)

        self.f = f
        self.s0 = s0
        self.H = H
        self.shadow_flag = shadow_flag

    
    def Mue_noncoh_R(self, geom, isdown=True):
        """
        Mueller matrix for upward diffuse scattering from rough surface in forward scattering alignment (FSA) convention.
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_s and theta_i belong to [0, 90] defined in surface scattering coordinate
                            theta_s is the angle between z and ks, while theta_i is the angle between z and -ki
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            R: 4x4 real Mueller matrix
        """
        theta_s, phi_s, theta_i, phi_i = geom
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r

        R = Mue_upward_fBmSSA(self.f, (theta_s, phi_s), (theta_i, phi_i), epsr, 
                            self.s0, self.H, self.shadow_flag)
            
        return R
        
