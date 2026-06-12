"""
Author: Fei Zhao
Create: 2023-08-29

Description:
    A layer with prescribed/predifined volume scattering coefficient ks & volume absorption coefficient ka and 
    equivalent complex dielectric constant of background medium.
"""

import numpy as np
from ..core.Layer import Layer


class Isotropickska(Layer):
    """
    A isotropic! layer defined by ks, ka, and epsr_background (Ulaby2014, p474, eq. 11.49).
    """
    def __init__(self, f, thickness=None, epsr_background=1.0, ks=0, ka=0, vol_frac=0):
        """
        INPUT:
            f: frequency (Hz) of the incident waves
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
            epsr_background: relative complex dielectric constant of the background medium
            ks: volume scattering coefficient of the scatterer
            ka: absorption coefficients of the scatterer
            vol_frac: volume fraction of the particles, have no impact on ks and ka
        """
        super(Isotropickska, self).__init__(f=f, thickness=thickness, epsr_background=epsr_background)
        self.ks = ks
        # self.ka = ka + self.Kab # background medium absorption have already been accounted for in VRT solver
        self.ka = ka
        self.ke = ks + ka
        self.Kab = -2 * self.k0 * np.sqrt(self.epsr_background).imag * (1 - vol_frac)
        self.ssa = self.ks / (self.ke + self.Kab)


    def print_params(self):
        """print characterized parameters:ks, ka, ke, ssa
        """
        print("ks:{}, ka:{}, ke:{}, ssa:{}".format(self.ks, self.ka, self.ke, self.ssa))
        

    def phase_matrix(self, geom=None):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        *Note that the isotropic point scatterers are assumed.
        INPUT:
            geom (tuple): (not needed) observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix has already been multiplied by n0
        """
        return np.diag([self.ks/(4*np.pi)]*4)
    

    def extinction_matrix(self, direction=None):
        """
        Compute extinction matrix Ke.
        *Note that the isotropic point scatterers are assumed.
        INPUT:
            direction (tuple): (not needed) direction of propagation (theta, phi) in degree
                            Note that theta belongs to [0, 180] defined in volume scattering coordinate.
                            theta is the angle between z and k.
        OUTPUT:
            Ke: 4x4 extinction matrix of the discrete scatterers
        """
        return np.diag([self.ke]*4)