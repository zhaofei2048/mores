"""
Author: Fei Zhao
Create: 2023-08-20

Description:
    A layer consists of discrete scatterers
"""

from abc import abstractmethod
import numpy as np
# import scipy.integrate
from .RayleighSphere import RayleighSphere 


class ScattererLayer(RayleighSphere):
    """
    Base class for a layer consists of discrete scatterers
    """
    def __init__(self, f, thickness=None, epsr_background=1.0, 
                 epsr_particle=1.0, particle_size=(1.0, 1.0), 
                 particle_orientation=0, particle_shape=(1, 'SPHEROID'), num_points=1024, **kwargs):
        """
        Construct a discrete scatterer layer
        INPUT:
            f: frequency (Hz) of the incident waves
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
            epsr_background: relative complex dielectric constant of the background medium
            epsr_particle: relative complex dielectric constant of the particle
            particle_size: can be a 1)1x2 tuple (radius, fs) = radius of equivalent (volume) sphere, volume fraction of the particles
                              or 2)1x3 tuple (0, Dmax, size_distribution_func) = Dmax is the maximum diameter of scatterers, size_distribution_func is
                              the size distribution function (callable) of particles defined in (0, Dmax), the integration of size_distribution_func between 0 to
                              Dmax should equal to n0, i.e., the number concentration of particles in a unit volume. Note the unit should be meters.
                              e.g. size_distribution_func(D) -> N(D)
            particle_orientation: can be a 1)1x2 tuple (alpha, beta) = The Euler angle alpha and beta of the scatterer orientation
                                or 2) scalar 0=uniform distribution, a value std_orien>0 = specifies the standard deviation of the angle (deg) with respect to
                                vertical orientation (canting angle)
            particle_shape: (axis_ratio, shape_type) = the horizontal-to-rotational axis ratio, shape_type can be 'SPHEROID' or 'CYLINDER'
            num_points: num of discrete points to calculate the size averaged parameters
        """
        super(ScattererLayer, self).__init__(f=f, thickness=thickness, epsr_background=epsr_background,
                                             epsr_particle=epsr_particle, particle_size=particle_size, num_points=num_points)

        # particle shape
        self.axis_ratio = particle_shape[0]
        self.shape_type = particle_shape[1]

        # particle orientation
        if type(particle_orientation) == tuple:
            self.is_multi_orientations = False
            self.alpha = particle_orientation[0]
            self.beta = particle_orientation[1]
        else:
            self.is_multi_orientations = True
            self.alpha = 0
            self.beta = 0
            self.std_orien = particle_orientation   # if std_orien == 0, then uniform distribution

    # @abstractmethod
    # def _phase_matrix(self, geom):
    #     """
    #     Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
    #     INPUT:
    #         geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
    #                         theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
    #                         Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
    #                         theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
    #     OUTPUT:
    #         P: 4x4 phase matrix
    #     """
    #     pass


    @abstractmethod
    def forward_scattering_amplitudes(self, direction):
        """
        Compute scattering amplitudes matrix in forward direction.
        INPUT:
            direction (tuple): direction of incidence (theta, phi) in degree
                            Note that theta belongs to [0, 180] defined in volume scattering coordinate.
                            theta is the angle between z and k.
        OUTPUT:
            SAf: 2x2 scattering amplitudes matrix in the forward scattering direction
        """
        pass


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
        S = self.forward_scattering_amplitudes(direction) # 2x2 scattering amplitudes matrix in the forward scattering direction
        Ke = np.array([[2*np.imag(S[0, 0]), 0, np.imag(S[0, 1]), -np.real(S[0, 1])],
                [0, 2*np.imag(S[1, 1]), np.imag(S[1, 0]), np.real(S[1, 0])],
                [2*np.imag(S[1, 0]), 2*np.imag(S[0, 1]), np.imag(S[0, 0]+S[1, 1]), np.real(S[0, 0]-S[1, 1])],
                [2*np.real(S[1, 0]), -2*np.real(S[0, 1]), -np.real(S[0, 0]-S[1, 1]), np.imag(S[0, 0]+S[1, 1])]])
        Ke = Ke * 2 * np.pi / self.k

        if self.is_multi_sizes is False:
            Ke = self.n0 * Ke
        # for multi_sizes, the n0 has been already multiplied equivalently

        return Ke 


