"""
Author: Fei Zhao
Create: 2023-08-29

Description:
    Abstract class of a layer

Update:
    2024-01-13: Add comments on interface to the Layer class
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as sci_const


class Layer(ABC):
    """
    Providing interfaces of a layer needed by the VRT solver

    Interfaces to be implemented for VRT solver:
        thickness: attribute, thickness of the layer
        Kab: attribute, background absorption coefficient
        phase_matrix(geom):
        extinction_matrix(direction): direction is the propagation direction
        single_scattering_albedo(): return the ssa of the layer
    """

    def __init__(self, f, thickness=None, epsr_background=1.0):
        """
        Construct a layer
        INPUT:
            f: frequency (Hz) of the incident waves
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
            epsr_background: relative complex dielectric constant of the background medium
        """
        self.freq = f
        self.Lambda0 = sci_const.speed_of_light / f  # wavelength in the free space
        self.k0 = 2 * np.pi / self.Lambda0   # wavenumber in the free space
        self.epsr_background = epsr_background
        self.Lambda = self.Lambda0 / np.sqrt(np.real(self.epsr_background)) # wavelength in the background medium
        self.k = 2 * np.pi / self.Lambda                                    # wavenumber in the background medium        

        # thickness
        if thickness is None:
            # calculate thickness by penetration depth
            self.thickness = self.__penetration_depth_scatterer_free()
        else:
            self.thickness = thickness       


    @abstractmethod
    def phase_matrix(self, geom):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix has already been multiplied by n0
        """
        pass


    @abstractmethod
    def extinction_matrix(self, direction):
        """
        Compute extinction matrix Ke.
        INPUT:
            direction (tuple): direction of propagation (theta, phi) in degree
                            Note that theta belongs to [0, 180] defined in volume scattering coordinate.
                            theta is the angle between z and k.
        OUTPUT:
            Ke: 4x4 extinction matrix of the discrete scatterers
        """
        pass


    def single_scattering_albedo(self):
        """The single-scattering albdeo of the particles in the layer.
        
        Returns:
            ssa: single-scattering albedo in [0, 1]
        """
        return self.ssa


    def __penetration_depth_scatterer_free(self):
        """
        Penetration depth of the medium free of scatterers
        INPUT:
        OUTPUT:
            d: penetration depth (meters)
        """
        d = 1 / self.Kab  if self.Kab != 0 else None  # Ulaby 2014, p469

        return d