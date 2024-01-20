"""
Author: Fei Zhao
Create: 2024-01-17

Description:
    Random rough surface.
    The non-coherent/diffuse scattering is calculated by IEM (backscattering) and AIEM (bistatic scattering).
"""

import numpy as np
import scipy.constants as sci_const
from .QuasiSmoothSurface import QuasiSmoothSurface
from .IEM_upward import Mue_upward_IEMs
from .AIEM_upward import Mue_upward_AIEMs
from .AIEM_downward import Mue_downward_AIEMs


class RoughSurface(QuasiSmoothSurface):
    """
    Random rough surface, the non-coherent/diffuse scattering is calculated by IEM (backscattering) and AIEM (bistatic scattering).
    """

    def __init__(self, epsilon_r, kdel, kcor, corr_fun='exp', x=1.5, shadow_flag=True):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: relative dielectric constant
            kdel: k * delta, normalized RMS height
            kcor: k * corr_len, normalized correlation length
                It should be noted that the roughness parameters kdel, kcor here is defined in free space (k means k0).
            corr_fun: correlation function: 'gauss', 'exp', 'x-power', 'x-exp' (default 'exp')
            x: coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correl. fnc. (default 1.5)
            shadow_flag: True (default) or False for including shadow effect
        OUTPUT:
            a rough surface instance
        """
        super(RoughSurface, self).__init__(epsilon_r, kdel)
        self.f = 1.0e9
        self.kcor = kcor

        Lambda = sci_const.speed_of_light / self.f
        k = 2.0 * np.pi / Lambda
        delta = self.kdel / k
        corr_len = self.kcor / k
        self.delta = delta
        self.corr_len = corr_len
        self.corr_fun = corr_fun
        self.x = x
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

        if (theta_i == theta_s and (phi_s - phi_i) == 180) or (theta_i == -theta_s and phi_i == phi_s):
            # backscattering
            R = Mue_upward_IEMs(self.f, (theta_s, phi_s), (theta_i, phi_i), epsr, 
                            self.delta, self.corr_len, self.corr_fun , self.x, self.shadow_flag)
        else:
            # bistatic scattering
            R = Mue_upward_AIEMs(self.f, (theta_s, phi_s), (theta_i, phi_i), epsr, 
                            self.delta, self.corr_len, self.corr_fun , self.x, self.shadow_flag)
            
        return R
        

    # def Mue_noncoh_T(self, geom, isdown=True):
    #     """
    #     Mueller matrix for downward diffuse scattering from rough surface in forward scattering alignment (FSA) convention.
    #     INPUT:
    #         geom (tuple): observation angles (theta_t, phi_t, theta_i, phi_i) in degree
    #                         theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
    #                         Note that theta_t and theta_i belong to [0, 90] defined in surface scattering coordinate
    #                         theta_t is the angle between -z and ks, while theta_i is the angle between z and -ki
    #         isdown: True (default) for downward incident and False for upward incident
    #     OUTPUT:
    #         T: 4x4 real Mueller matrix
    #     """
    #     theta_t, phi_t, theta_i, phi_i = geom
    #     if isdown is True:
    #         epsr = self.epsilon_r
    #     else:
    #         epsr = 1.0 / self.epsilon_r
        
    #     R = Mue_downward_AIEMs(self.f, (theta_t, phi_t), (theta_i, phi_i), epsr, 
    #                     self.delta, self.corr_len, self.corr_fun , self.x, self.shadow_flag)
            
    #     return R