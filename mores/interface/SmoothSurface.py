"""
Author: Fei Zhao
Create: 2023-08-30

Description:
    Smooth surface, only coherent fresnel scattering is accounted for, with zero diffuse/non-coherent scattering.
"""

import numpy as np
from .TransparentSurface import TransparentSurface
from .fresnel import fresnel_coefficients, refraction_angle, critical_angle


class SmoothSurface(TransparentSurface):
    """
    Smooth surface class
    """

    def M_coh_R(self, theta_i, isdown=True):
        """
        Coherent reflection at the surface
        aka Reflectivity matrix (Ulaby 2014, eq. 11.61a)
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Rc: 4x4 Stokes matrix for reflection coherent scattering
        """
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r
        rv, rh, _, _ = fresnel_coefficients(theta_i, epsr)
        # Ulaby 2014, eq. 11.61a
        R = np.array([[np.abs(rv)**2, 0, 0, 0],
                    [0, np.abs(rh)**2, 0, 0],
                    [0, 0, np.real(rv * np.conj(rh)), -np.imag(rv * np.conj(rh))],
                    [0, 0, np.imag(rv * np.conj(rh)), np.real(rv * np.conj(rh))]])
        
        return R


    def M_coh_T(self, theta_i, isdown=True):
        """
        Coherent transmission at the surface
        aka. Transmissivity matrix (Ulaby 2014, eq. 11.61b)
        INPUT:
            theta_i: incident angle (deg)
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            Tc: 4x4 Stokes matrix for transmission coherent scattering
        """
        assert(np.isscalar(theta_i))
        assert(np.isscalar(self.epsilon_r))
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r

        _, _, tv, th = fresnel_coefficients(theta_i, epsr)
        er_r = np.real(epsr)
        if er_r < 1:
            # possibility for total reflection under critical incident angle
            theta_c = self.critical_angle(isdown)
            # print("theta_c:{}".format(theta_c))
            if theta_i >= theta_c:
                return np.zeros((4, 4))
        
        theta_t = refraction_angle(theta_i, epsr)
        cs = np.cos(np.deg2rad(theta_i))
        cst = np.cos(np.deg2rad(theta_t))
        # Ulaby 2014, eq. 11.61b
        T = er_r**1.5 * cst / cs * np.array([[np.abs(tv)**2, 0, 0, 0],
                    [0, np.abs(th)**2, 0, 0],
                    [0, 0, np.real(tv * np.conj(th)), -np.imag(tv * np.conj(th))],
                    [0, 0, np.imag(tv * np.conj(th)), np.real(tv * np.conj(th))]])
        
        # T[0,0] = (1-np.abs(rv)**2)  # just for test
        # T[1,1] = (1-np.abs(rh)**2)  # just for test
        # T = np.diag([0]*4) # just for test
        # T[0,0] = 1  # just for test
        # T[1,1] = 1  # just for test
        # T = er_r**0.5 * cst / cs * np.array([[np.abs(tv)**2, 0, 0, 0],
        #             [0, np.abs(th)**2, 0, 0],
        #             [0, 0, np.real(tv * np.conj(th)), -np.imag(tv * np.conj(th))],
        #             [0, 0, np.imag(tv * np.conj(th)), np.real(tv * np.conj(th))]])
        # T[0, 0] = 1-np.abs(rv)**2
        # adapted from SMRT
        # rv, rh, _, _ = fresnel_coefficients(theta_i, epsr)
        # T1 = np.array([[1-np.abs(rv)**2, 0, 0, 0],
        #     [0, 1-np.abs(rh)**2, 0, 0],
        #     [0, 0, cst/cs*np.real((1+rv) * np.conj(1+rh)), -cst/cs*np.imag((1+rv) * np.conj(1+rh))],
        #     [0, 0, cst/cs*np.imag((1+rv) * np.conj(1+rh)), cst/cs*np.real((1+rv) * np.conj(1+rh))]])
        # T = np.zeros((4,4))
        # print(T1)
        # print('---------')
        # print(T)

        return T

    
    def critical_angle(self, isdown=True):
        """
        Calculate critical angle (total reflection) of the surface
        INPUT:
        OUTPUT:
            theta_c: critical angle (deg)
        """
        if isdown is True:
            epsr = self.epsilon_r
        else:
            epsr = 1.0 / self.epsilon_r

        theta_c = critical_angle(epsr)
        
        return theta_c