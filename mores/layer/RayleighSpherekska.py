"""
Author: Fei Zhao
Create: 2023-08-29

Description:
    Rayleigh sphere layer with prescribed ks, ka
"""
import numpy as np
from .Isotropickska import Isotropickska
# from postprocessing import scattering_amplitudes_to_Mueller

class RayleighSpherekska(Isotropickska):
    """
    Rayleigh sphere layer with prescribed ks, ka
    """
    def __init__(self, f, thickness=None, epsr_background=1.0, ks=0, ka=0, vol_frac=0):
        """
        INPUT:
            f: frequency (Hz) of the incident waves
            epsr_background: relative complex dielectric constant of the background medium
            ks: volume scattering coefficient
            ka: volume adsorption coefficient
            vol_frac: volume fraction of the particles, have no impact on ks and ka
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
        """
        super(RayleighSpherekska, self).__init__(f=f, thickness=thickness, epsr_background=epsr_background, ks=ks, ka=ka, vol_frac=vol_frac)

   
    def phase_matrix(self, geom):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        *Note that the Rayleigh spherical scatterers are assumed.
        INPUT:
            geom (tuple): (not needed) observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix has already been multiplied by n0
        """
        ths, phis, thi, phi = np.deg2rad(geom)
        s = np.sin(thi)
        cs = np.cos(thi)
        ss = np.sin(ths)
        css = np.cos(ths)
        sf = np.sin(phis - phi)
        csf = np.cos(phis - phi)
        b = 3 * self.ks / (8*np.pi)

        # Ulaby 2014, eq. 11.51, p474
        p11 = ss**2 * s**2 + css**2 * cs**2 * csf**2 + 2 * ss *s *css *cs * csf
        p12 = css**2 * sf**2
        p13 = css * ss * s * sf + css**2 * cs * sf * csf
        p21 = cs**2 * sf**2
        p22 = csf**2
        p23 = -css * sf * csf
        p31 = -2 * ss *s * cs * sf - 2 * css * cs**2 * csf * sf
        p32 = 2 * css * sf * csf
        p33 = ss * s * csf + css * cs * (csf**2 - sf**2)
        p44 = ss * s * csf + css * cs

        # P = 3 * self.ks / (8*np.pi) * np.array([[p11, p12, p13, 0],
        #                                         [p21, p22, p23, 0],
        #                                         [p31, p32, p33, 0],
        #                                         [0, 0, 0, p44]])
        # p11 = 1 # just for test
        P = b * np.array([[p11, p12, p13, 0],
                        [p21, p22, p23, 0],
                        [p31, p32, p33, 0],
                        [0, 0, 0, p44]])


        # Rayleigh sphere scattering amplitudes (adapted from SMRT), and this is equivalent to Ulaby
        # S = np.array([[csf*css*cs+ss*s, sf*css],
        #               [-sf*cs, csf]])
        # P = b * scattering_amplitudes_to_Mueller(S)

        return P