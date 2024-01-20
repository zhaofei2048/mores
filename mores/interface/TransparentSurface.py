"""
Author: Fei Zhao
Create: 2023-08-30

Description:
    Ideallized transparent surface.
"""

import numpy as np
from .fresnel import refraction_angle
from ..core.Surface import Surface


class TransparentSurface(Surface):
    """
    Transparent surface with totally transmission and zero reflection.
    *Though transparent, the change of propagation direction is accounted.
    """
    def __init__(self, epsilon_r=1.0):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: dielectric constant of medium below (relative to above)
                default 1.0 means totally transparent, even no refraction at the surface
        OUTPUT:
            a smooth surface instance
        """
        self.epsilon_r = epsilon_r


    def M_coh_R(self, theta_i, isdown=True):
        """
        Coherent reflection at the surface
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Rc: 4x4 Stokes matrix for reflection coherent scattering
        """
        return np.zeros((4, 4))


    def M_coh_T(self, theta_i, isdown=True):
        """
        Coherent transmission at the surface
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Tc: 4x4 Stokes matrix for transmission coherent scattering
        """
        return np.diag([1.0]*4)


    def refraction_angle(self, theta_i, isdown=True):
        """
        Calculate refraction angle at this interface (surface).
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            theta_t: the refraction angle (deg)
        
        Note:
            Though transparent, the change of propagation direction should keep accounted.
        """
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r
        theta_t = refraction_angle(theta_i, epsr)

        return theta_t