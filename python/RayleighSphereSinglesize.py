"""
Author: Fei Zhao
Create: 2023-09-01

Description:
    A layer consists of Rayleigh spherical scatterers
"""
import numpy as np
import scipy.constants as C

from RayleighSpherekskaeps import RayleighSpherekskaeps 


class RayleighSphereSinglesize(RayleighSpherekskaeps):
    """
    Rayleigh sphere layers, see Ulaby 2014, p474
    """
    def __init__(self, f, epsr_background, epsr_particle, radius, fs, thickness=None):
        """
        INPUT:
            f: frequency (Hz) of the incident waves
            epsr_background: relative complex dielectric constant of the background medium
            epsr_particle: relative complex dielectric constant of the particle
            radius: radius of equivalent (volume) a single sphere
            fs: volume fraction of the particles
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
        """
        # particle_orientation = (0, 0)
        # particle_shape=(1, 'SPHEROID')
        # super(RayleighSphere, self).__init__(f, epsr_background, epsr_particle, particle_size, particle_orientation,
        #                                      particle_shape, thickness)
        epsr = epsr_particle / epsr_background
        Lambda0 = C.speed_of_light / f  # wavelength in the free space
        Lambda = Lambda0 / np.sqrt(np.real(epsr_background)) # wavelength in the background medium
        k = 2 * np.pi / Lambda  

        # volume scattering coefficient
        K = (epsr - 1) / (epsr + 2)
        ks = 2 * fs * k**4 * radius**3 * np.abs(K)**2
        # volume absorption coefficient
        ka = 3 * fs * k * np.imag(-K)

        super(RayleighSphereSinglesize, self).__init__(f, epsr_background, ks, ka, fs, thickness)

        # self.epsr_particle = epsr_particle
        # self.epsr = epsr # relative complex diel. of particles to that of background medium
        # self.refractive_index = np.sqrt(self.epsr)
        # self.radius = radius
        # self.fs = fs 
