"""
Author: Fei Zhao
Create: 2023-08-30

Description:
    Abstract class of surface

Update:
    2024-01-17: Change the non-coherent abstract method to real method.
"""
from abc import ABC, abstractmethod
import numpy as np


class Surface(ABC):
    """
    Abstract class for surface.
    Interfaces to be implemented for VRT solver:
        M_coh_R(theta_i, isdown)
        M_coh_T(theta_i, isdown)
        Mue_noncoh_R(geom, isdown)
        Mue_noncoh_T(geom, isdown)
        refraction_angle(theta_i, isdown)
    """
    @abstractmethod
    def M_coh_R(self, theta_i, isdown=True):
        """
        Coherent reflection at the surface
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Rc: 4x4 Stokes matrix for reflection coherent scattering
        """
        pass


    @abstractmethod
    def M_coh_T(self, theta_i, isdown=True):
        """
        Coherent transmission at the surface
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Tc: 4x4 Stokes matrix for transmission coherent scattering
        """
        pass


    def Mue_noncoh_R(self, geom, isdown=True):
        """
        Non-coherent reflection at the surface
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                Note that theta_s and theta_i belong to [0, 90] defined in surface scattering coordinate
                theta_s is the angle between z and ks, while theta_i is the angle between z and -ki
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            R: 4x4 real Mueller matrix
        """
        return np.zeros((4, 4))


    def Mue_noncoh_T(self, geom, isdown=True):
        """
        Non-coherent transmission at the surface
        INPUT:
            geom (tuple): observation angles (theta_t, phi_t, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_t and theta_i belong to [0, 90] defined in surface scattering coordinate
                            theta_t is the angle between -z and ks, while theta_i is the angle between z and -ki
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            T: 4x4 real Mueller matrix
        """
        return np.zeros((4, 4))


    def refraction_angle(self, theta_i, isdown=True):
        """
        Calculate refraction angle at this interface(surface).
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            theta_t: the refraction angle (deg)
        """
        return theta_i