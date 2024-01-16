"""
Author: Fei Zhao
Create: 2023-08-19

Description:
    Vector Radiative Transfer (VRT) equation first-order iterative solver

Updated:
    2024-01-15: Corrected the subsurface term, the non-coherent Mueller matrix is not needed to be divided by (4*np.pi)
"""

import numpy as np
# import scipy.constants as C
from scipy.integrate import quad_vec
# from AIEM import AIEM
# from TmatrixScatterer import TmatrixScatterer
# from Prescribedkskaeps import Prescribedkskaeps
# from RayleighSphereKsKa import RayleighSphereKsKa
from fresnel import refraction_angle
import warnings

class VRT:
    """
    VRT first-order iterative solver class
    """
    def __init__(self, surface, layer, subsurface=None):
        """
        Init the VRT solver with MORES model parameters
        INPUT:
            surface: an initialized surface object
            layer: an initialized layer object
            subsurface: an initialized surface object
        """
        self.surface = surface
        self.layer = layer
        self.subsurface = subsurface
        self.__check()

    def __check(self):
        """ Check the model
        
        Raises:
            Model error
        """

        if self.surface.epsilon_r != self.layer.epsr_background:
            # TODO:
            # this is the bad implementation, surface should not have independent dielectric constant
            # Another problem: The frequency for different components may be different
            raise ValueError("Non-consistent dielectric constant between surface and the layer.")
        
        ssa = self.layer.single_scattering_albedo()
        if ssa > 0.3:
            warnings.warn("The ssa ({}) is too high, the first-order iterative VRT solver may obtain inaccurate results.".format(ssa))


    # Please preserve this function for reference
    # def __extinction_eig_ana(self, direction):
    #     """
    #     Compute eigenvectors and corresponding eigenvalues of the extinction matrix analytically with methods of Tsang 1985.
    #     Note this is equivalent to np.linalg(Ke), where Ke is the extinction matrix.
    #     INPUT:
    #         direction (tuple): direction of propagation (theta, phi) in degree
    #                         Note that theta belongs to [0, 180] defined in volume scattering coordinate.
    #                         theta is the angle between z and k.
    #     OUTPUT:
    #         EDt: (E, beta, theta) where E is the 4x4 eigenvectors matrix of Ke,
    #                    and beta is the corresponding 1x4 eigenvalus array,
    #                    and theta is the polar angle of propagation (see extinction_eig()).
    #     """
    #     S = self.layer.forward_scattering_amplitudes(direction) # 2x2 scattering amplitudes matrix in the forward scattering direction
    #     # print('Scattering amplitudes')
    #     # print(S)
    #     r = np.sqrt((S[0, 0] - S[1, 1])**2 + 4 * S[0, 1] * S[1, 0])
    #     if np.abs(4 * S[0, 1] * S[1, 0]) < (1e-6 * (S[0, 0] - S[1, 1])**2):
    #         r = np.sign(S[0, 0] - S[1, 1]) * r
    #         pass
    #     denom = (S[0, 0] - S[1, 1] + r)
    #     if denom < 1e-20:
    #         b1 = 0.
    #         b2 = 0.
    #     else:
    #         b1 = 2*S[1, 0] / denom
    #         b2 = -2*S[0, 1] / denom

    #     beta = np.array([r.imag, -1j*r.real, 1j*r.real, -r.imag]) + (S[0, 0] + S[1, 1]).imag
    #     beta = beta * 2 * np.pi * self.n0 / self.k
    #     E = np.array([[1, np.conj(b2), b2, np.abs(b2)**2],
    #                     [np.abs(b1)**2, b1, np.conj(b1), 1],
    #                     [2*b1.real, 1+b1*np.conj(b2), 1+np.conj(b1)*b2, 2*np.real(b2)],
    #                     [-2*b1.imag, -1j*(1-b1*np.conj(b2)), 1j*(1-np.conj(b1)*b2), 2*b2.imag]])
        
    #     return (E, beta, direction[0])


    def __extinction_eig(self, direction):
        """
        Compute eigenvectors and corresponding eigenvalues of the extinction matrix numerically.
        INPUT:
            direction (tuple): direction of propagation (theta, phi) in degree
                            Note that theta belongs to [0, 180] defined in volume scattering coordinate.
                            theta is the angle between z and k.
        OUTPUT:
            EDt: (E, beta, theta) where E is the 4x4 eigenvectors matrix of Ke,
                       and beta is the corresponding 1x4 eigenvalus array,
                       and theta is the polar angle of propagation (see extinction_eig()).
        """
        # extinction matrix
        Ke = self.layer.extinction_matrix(direction)
        beta, E = np.linalg.eig(Ke)

        return (E, beta, direction[0])
    

    def __extinction_propagate(self, EDt, z, bg_absorb=True):
        """
        Extinction propagation in the layer: E(theta, phi)D(z*beta(theta,phi)*abs(sec(theta)))E^-1(theta, phi)
        INPUT:
            EDt: (E, beta, theta) where E is the 4x4 eigenvectors matrix of Ke
                       and beta is the corresponding 1x4 eigenvalus array 
                       and theta is the polar angle of propagation (see extinction_eig())
            z: position of current propagation, should belongs to (-d, 0) (must < 0)
            bg_absorb: True (default) or False to include or not the effect of absorption by background medium
        OUTPUT:
            EDinvE: 4x4 extinction propation matrix
        """
        E, beta, theta = EDt
        # below should be uncommented, just for test
        if bg_absorb is True:
            beta = beta + self.layer.Kab

        c = beta * z * np.abs(1/np.cos(np.deg2rad(theta)))
        D = np.diag(np.exp(c))
        EDinvE = E @ D @ np.linalg.inv(E)

        return EDinvE


    def Mue_surface(self, geom):
        """
        Mueller matrix for surface scattering in forward scattering alignment (FSA) convention.
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
        OUTPUT:
            Mue: 4x4 real Mueller matrix
        """
        Mue = self.surface.Mue_noncoh_R(geom)

        return Mue


    def Mue_volume(self, geom):
        """
        Mueller matrix for volume scattering in forward scattering alignment (FSA) convention.
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
        OUTPUT:
            Mue: 4x4 real Mueller matrix
        """
        theta_s, phi_s, theta_i, phi_i = geom
        theta_it = self.surface.refraction_angle(theta_i)
        theta_st = self.surface.refraction_angle(theta_s)
        d = self.layer.thickness

        Tc01 = self.surface.M_coh_T(theta_i)

        EDt1 = self.__extinction_eig((180-theta_it, phi_i))
        P = self.layer.phase_matrix((theta_st, phi_s, 180-theta_it, phi_i))
        EDt2 = self.__extinction_eig((theta_st, phi_s))
        fz = lambda z: (self.__extinction_propagate(EDt2, z) 
                @ P
                @ self.__extinction_propagate(EDt1, z))
        Fz,err = quad_vec(fz, -d, 0)

        Tc10 = self.surface.M_coh_T(theta_st, False)
        # Tc01 = np.diag([1]*4) # just for test
        # Tc10 = np.diag([1]*4)  # just for test
        M =  (1/np.cos(np.deg2rad(theta_st))) * (Tc10 @ Fz @ Tc01)
        Mue =  np.cos(np.deg2rad(theta_s)) * np.real(M)

        # Mue = Tc10@Tc01 # just for test

        return Mue
    

    def Mue_subsurface(self, geom, path=0):
        """Mueller matrix for volume scattering in forward scattering alignment (FSA) convention from subsurface.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
            path: scattering path 1 for Tc01->Rn12->Tc10, path 2 for Tc01->Rc12->Tn10, path 3 for Tn01->Rc12->Tc10
                and 0 for total contribution=path_1+path_2+path_3

        Returns:
            Mue: 4x4 real Mueller matrix
        """
        if self.subsurface is None:
            return np.zeros((4, 4))
        # prepare
        theta_s, phi_s, theta_i, phi_i = geom
        theta_it = self.surface.refraction_angle(theta_i)
        theta_st = self.surface.refraction_angle(theta_s)
        d = self.layer.thickness

        Tc01 = self.surface.M_coh_T(theta_i)
        Tc10 = self.surface.M_coh_T(theta_st, False)
        Tn01 = self.surface.Mue_noncoh_T((theta_st, phi_s, theta_i, phi_i)) / (np.cos(np.deg2rad(theta_st)))
        Tn10 = self.surface.Mue_noncoh_T((theta_s, phi_s, theta_it, phi_i)) / (np.cos(np.deg2rad(theta_s)))

        Rc12 = self.subsurface.M_coh_R(theta_it)
        Rn12 = self.subsurface.Mue_noncoh_R((theta_st, phi_s, theta_it, phi_i)) / (np.cos(np.deg2rad(theta_st)))

        # 0: theta_it down, 1: theta_it up, 2: theta_st down, 3: theta_st up
        directions = [(180-theta_it, phi_i), (theta_it, phi_i), (180-theta_st, phi_s), (theta_st, phi_s)]
        EDts = [self.__extinction_eig(direction) for direction in directions]
        EDEinvs = [self.__extinction_propagate(EDt, -d) for EDt in EDts]

        # path 1
        M1 = Tc10 @ EDEinvs[3] @ Rn12 @ EDEinvs[0] @ Tc01
        # path 2
        M2 = Tn10 @ EDEinvs[1] @ Rc12 @ EDEinvs[0] @ Tc01
        # path 3
        M3 = Tc10 @ EDEinvs[3] @ Rc12 @ EDEinvs[2] @ Tn01
        Ms = [M1, M2, M3]

        # path = 1
        if path == 0:
            M = M1 + M2 + M3
        else:
            M = Ms[path-1]
        
        Mue = np.cos(np.deg2rad(theta_s)) * np.real(M)
        # Mue = np.cos(np.deg2rad(theta_s)) * Rn12

        return Mue


    def Mue_subsurface_volume(self, geom, path=0):
        """Mueller matrix for volume scattering in forward scattering alignment (FSA) convention from subsurface-volume.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
            path: scattering path 1 for subsurface->volume interaction and 2 for volume->subsurface interaction
                and 0 for total contribution=path_1+path_2

        Returns:
            Mue: 4x4 real Mueller matrix
        """
        if self.subsurface is None:
            return np.zeros((4, 4))
        # prepare
        theta_s, phi_s, theta_i, phi_i = geom
        theta_it = self.surface.refraction_angle(theta_i)
        theta_st = self.surface.refraction_angle(theta_s)
        d = self.layer.thickness

        Tc01 = self.surface.M_coh_T(theta_i)
        Tc10 = self.surface.M_coh_T(theta_st, False)
        Rc12i = self.subsurface.M_coh_R(theta_it)
        Rc12s = self.subsurface.M_coh_R(theta_st)

        # path 1: sub-vol
        if path == 1 or path == 0:
            EDt_id = self.__extinction_eig((180-theta_it, phi_i))   # i, d for subscripts incident and d (layer thickness)
            EDEinv_id = self.__extinction_propagate(EDt_id, -d) # inv for inversion
            EDt11 = self.__extinction_eig((theta_it, phi_i))
            EDt12 = self.__extinction_eig((theta_st, phi_s))
            P1 = self.layer.phase_matrix((theta_st, phi_s, theta_it, phi_i))
            # P1 = np.diag([1]*4) # just for test
            # P1[0, 0] = 1 # just for test
            fz1 = lambda z: (self.__extinction_propagate(EDt12, z)
                            @ P1
                            @ self.__extinction_propagate(EDt11, -(d+z)))
            # Fz1, err = quad_vec(fz1, -d, 0.5*d) # just for test
            Fz1, err = quad_vec(fz1, -d, 0)
            # !!! Please check! 2024-01-13: checked
            # M1 = 1/np.cos(np.deg2rad(theta_st)) * (Tc10 @ Fz1 @ Rc12i @ EDt_id @ Tc01)
            # Fz1 = np.diag([1]*4) # just for test
            M1 = 1/np.cos(np.deg2rad(theta_st)) * (Tc10 @ Fz1 @ Rc12i @ EDEinv_id @ Tc01)
        else:
            M1 = np.zeros((4, 4))

        # path 2: vol-sub
        if path == 2 or path == 0:
            EDt_su = self.__extinction_eig((theta_st, phi_s))   # s, u for subscripts scattering and sub?
            EDEinv_su = self.__extinction_propagate(EDt_su, -d)
            EDt21 = self.__extinction_eig((180-theta_it, phi_i))
            EDt22 = self.__extinction_eig((180-theta_st, phi_s))
            P2 = self.layer.phase_matrix((180-theta_st, phi_s, 180-theta_it, phi_i))
            fz2 = lambda z: (self.__extinction_propagate(EDt22, -(d+z))
                            @ P2
                            @ self.__extinction_propagate(EDt21, z))
            Fz2, err = quad_vec(fz2, -d, 0)
            # M2 = 1/np.cos(np.deg2rad(theta_st)) * (Tc10 @ EDEinv_su @ Rc12s @ Fz1 @ Tc01)
            # !!! Please check! 2024-01-13: checked
            M2 = 1/np.cos(np.deg2rad(theta_st)) * (Tc10 @ EDEinv_su @ Rc12s @ Fz2 @ Tc01)
        else:
            M2 = np.zeros((4, 4))

        Ms = [M1, M2]
        
        if path == 0:
            M = M1 + M2
        else:
            M = Ms[path-1]
        
        Mue = np.cos(np.deg2rad(theta_s)) * np.real(M)
        # Mue = Tc01

        return Mue


    def Mue_volume_nosurface(self, geom):
        """Mueller matrix for volume scattering in forward scattering alignment (FSA) convention without surface (or is transparent).

            For validation with Tsang and SMRT.
            But the refraction effect at the surface is still accounted for, if the medium of layer is not air.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles

        Returns:
            Mue: 4x4 real Mueller matrix
        
        Note:
            The utility of this function can be replaced with a combination use of transparent surface and `self.Mue_volume`.
        """
        theta_s, phi_s, theta_i, phi_i = geom
        # refraction at the interface though transparent
        theta_it = refraction_angle(theta_i, self.layer.epsr_background)
        theta_st = refraction_angle(theta_s, self.layer.epsr_background)

        d = self.layer.thickness
        EDt1 = self.__extinction_eig((180-theta_it, phi_i))
        P = self.layer.phase_matrix((theta_st, phi_s, 180-theta_it, phi_i))
        EDt2 = self.__extinction_eig((theta_st, phi_s))

        fz = lambda z: (self.__extinction_propagate(EDt2, z) 
                        @ P
                        @ self.__extinction_propagate(EDt1, z))
        Fz,err = quad_vec(fz, -d, 0)
        # Fz = np.ones((4, 4))
        # print('Fz')
        # print(Fz)
        # Mue volume
        # Mue = np.cos(np.deg2rad(theta_s)) / np.cos(np.deg2rad(theta_s)) * (Fz)
        M = Fz

        Mue = np.cos(np.deg2rad(theta_s)) * np.real(M)

        return Mue


    def test_Mue_volume_nosurface(self, geom):
        """Mueller matrix for volume scattering in forward scattering alignment (FSA) convention without surface (or is transparent).

            For validation with Tsang and SMRT.
            But the refraction effect at the surface is still accounted for, if the medium of layer is not air.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles

        Returns:
            Mue: 4x4 real Mueller matrix
        
        Note:
            The utility of this function can be replaced with a combination use of transparent surface and `self.Mue_volume`.
        """
        print("test_Mue_volume_nosurface")
        theta_s, phi_s, theta_i, phi_i = geom
        # refraction at the interface though transparent
        theta_it = theta_i # 入射时假设不变化，出射时才折射
        theta_st = refraction_angle(theta_s, self.layer.epsr_background)

        d = self.layer.thickness
        EDt1 = self.__extinction_eig((180-theta_it, phi_i))
        P = self.layer.phase_matrix((theta_st, phi_s, 180-theta_it, phi_i))
        EDt2 = self.__extinction_eig((theta_st, phi_s))

        fz = lambda z: (self.__extinction_propagate(EDt2, z) 
                        @ P
                        @ self.__extinction_propagate(EDt1, z))
        Fz,err = quad_vec(fz, -d, 0)
        # Fz = np.ones((4, 4))
        # print('Fz')
        # print(Fz)
        # Mue volume
        # Mue = np.cos(np.deg2rad(theta_s)) / np.cos(np.deg2rad(theta_s)) * (Fz)
        M = Fz

        Mue = np.cos(np.deg2rad(theta_s)) * np.real(M)

        return Mue
    
    # def Mue_smooth_subsurface_nosurface(self, geom):
    #     """
    #     Mueller matrix for volume scattering in forward scattering alignment (FSA) convention with smooth subsurface, but without surface.
    #     for validation with Tsang
    #     INPUT:
    #         geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
    #                         theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
    #     OUTPUT:
    #         Mue: 4x4 real Mueller matrix
    #     """
    #     theta_s, phi_s, theta_i, phi_i = geom
    #     d = self.layer.thickness
    #     EDt1 = self.layer.extinction_eig((theta_s, phi_s))
    #     P = self.layer.phase_matrix((theta_s, phi_s, 180-theta_i, phi_i))
    #     EDt2 = self.layer.extinction_eig((180-theta_i, phi_i))
    #     EDinvE2 = self.layer.extinction_propagate(EDt2, -d)