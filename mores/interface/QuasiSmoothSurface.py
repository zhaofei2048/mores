"""
Author: Fei Zhao
Create: 2023-08-19

Description:
    quasi-smooth surface (with some roughness), only coherent fresnel scattering is accounted for, with zero diffuse/non-coherent scattering. The loss of coherent scattering due to roughness is accounted for.
"""

import numpy as np
from .fresnel import refraction_angle
from .SmoothSurface import SmoothSurface


class QuasiSmoothSurface(SmoothSurface):
    """
    Quasi-smooth surface (with some roughness), but with zero diffuse/non-coherent scattering
    """

    def __init__(self, epsilon_r, kdel):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: relative dielectric constant
            kdel: k * delta, normalized RMS height
                It should be noted that the roughness parameter kdel here is defined in free space (k means k0).
        OUTPUT:
            a rough surface instance
        """
        super(QuasiSmoothSurface, self).__init__(epsilon_r)
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
        R = super(QuasiSmoothSurface, self).M_coh_R(theta_i, isdown)
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
        T = super(QuasiSmoothSurface, self).M_coh_T(theta_i, isdown)
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