"""RoughSurface interface based on Hagfors backscattering model

Author: Fei Zhao
Create: 2023-08-30
"""

import numpy as np
from .SmoothSurface import SmoothSurface
from .Hagfors_backscattering import sigma0_pp_Hagfors



class HagforsSurface(SmoothSurface):
    """Note this interface only support backscattering currently!!!

    1. No cohernet reflection and transmission are included, i.e., they are zero.
    """

    def __init__(self, epsilon_r, C):
        """Set parameters for characterizing the rough surface

        Args:
            epsilon_r: relative dielectric constant
            C: Hagfors roughness parameters, should >> 6.25 for the requirement of geometrical optics (GO) approximation;
                C increases as the roughness of the surface decreases.
        Returns:
            a rough surface instance
        """
        super(SmoothSurface, self).__init__(epsilon_r)
        self.C = C
    

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

        if (theta_i == theta_s and (phi_s - phi_i) == 180) or (theta_i == -theta_s and phi_i == phi_s):
            # backscattering
            sigma0_pp = sigma0_pp_Hagfors(theta_i, epsr, self.C)
            R = np.zeros((4, 4))
            R[0, 0] = sigma0_pp / (4 * np.pi)
            R[1, 1] = sigma0_pp / (4 * np.pi)
            R[2, 2] = sigma0_pp / (4 * np.pi)
            R[3, 3] = sigma0_pp / (4 * np.pi)
        else:
            raise ValueError('The HagforsSurface currently only support backscattering configuration.')

        # BSA to FSA
        U = np.diag([1, 1, -1, -1])
        R = U @ R

        return R
    

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
        return np.zeros((4, 4))
