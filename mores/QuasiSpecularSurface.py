"""
Author: Fei Zhao
Create: 2023-08-19

Description:
    Rough surface class (with zero diffuse/non-coherent scattering)
"""
import numpy as np
from fresnel import fresnel_coefficients, refraction_angle
from SmoothSurface import SmoothSurface

class QuasiSpecularSurface(SmoothSurface):
    """
    Rough surface but with zero diffuse/non-coherent scattering
    """
    def __init__(self, epsilon_r, kdel):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: relative dielectric constant
            kdel: k * delta, normalized RMS height
            kcor: k * corr_len, normalized correlation length
        OUTPUT:
            a rough surface instance
        """
        super(QuasiSpecularSurface, self).__init__(epsilon_r)
        self.kdel = kdel

    def M_coh_R(self, theta_i, isdown=True):
        """
        Coherent reflection matrix for rough surface (Ulaby 2014, eq. 5.110; Fung 1994, p54)
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Rc: 4x4 Stokes matrix for reflection coherent scattering
        """
        R = super(QuasiSpecularSurface, self).M_coh_R(theta_i, isdown)
        cs = np.cos(np.deg2rad(theta_i))
        Loss = np.exp(-4 * self.kdel**2 * cs**2)
        Rc = Loss * R

        return Rc

    def M_coh_T(self, theta_i, isdown=True):
        """
        Coherent transmission matrix for rough surface (Fung 1994, p54)
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Tc: 4x4 Stokes matrix for transmission coherent scattering
        """
        T = super(QuasiSpecularSurface, self).M_coh_T(theta_i, isdown)
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r

        theta_t = refraction_angle(theta_i, epsr)
        cs = np.cos(np.deg2rad(theta_i))
        cst = np.cos(np.deg2rad(theta_t))
        Loss = np.exp(-self.kdel**2 * (np.sqrt(epsr).real * cst - cs)**2)
        # Loss = 1    # just for test
        Tc = Loss * T

        return Tc