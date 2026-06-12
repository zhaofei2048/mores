"""
Author: Fei Zhao
Create: 2023-08-20

Description:
    A layer consists of discrete scatterers (solved by Tmatrix numeric method)
"""

from .ScattererLayer_old import ScattererLayer
import numpy as np
from pytmatrix import tmatrix, orientation
from pytmatrix.psd import PSDIntegrator
from pytmatrix import scatter
from ..utils.postprocessing import Mueller_matrix_L2M


class SurfaceTmatrixRocks():
    """
    A layer consists of discrete scatterers (solved by Tmatrix numeric method)
    """

    def __init__(self, f,
                 epsr_particle=1.0, particle_size=(1.0, 1.0), 
                 particle_orientation=0, particle_shape=(1, 'SPHEROID'), num_points=1024):
        """
        Construct a discrete scatterer layer driven by pytmatrix
        INPUT:
            f: frequency (Hz) of the incident waves
            epsr_particle: relative complex dielectric constant of the particle
            particle_size: can be a 1)1x2 tuple (radius, Fs) = radius of equivalent (volume) sphere, area fraction of the particles
                              or 2)1x3 tuple (0, Dmax, size_distribution_func) = Dmax is the maximum diameter of scatterers, size_distribution_func is
                              the size distribution function of particles defined in (0, Dmax), the integration of size_distribution_func between 0 to
                              Dmax should equal to n0, i.e., the number concentration of particles in a unit volume. Note the unit should be meters
            particle_orientation: can be a 1)1x2 tuple (alpha, beta) = The Euler angle alpha and beta of the scatterer orientation
                                or 2) scalar 0=uniform distribution, a value std_orien>0 = specifies the standard deviation of the angle (deg) with respect to
                                vertical orientation (canting angle)
            particle_shape: (axis_ratio, shape_type) = the horizontal-to-rotational axis ratio, shape_type can be 'SPHEROID' or 'CYLINDER'
            num_points: num of discrete points to calculate the size averaged parameters
        """
        if self.shape_type == 'SPHEROID':
            shape_type = tmatrix.Scatterer.SHAPE_SPHEROID
        elif self.shape_type == 'CYLINDER':
            shape_type = tmatrix.Scatterer.SHAPE_CYLINDER
        else:
            raise ValueError("Wrong shape type of particles")
        m = np.conj(self.refractive_index)  # pytmatrix convention: m=m'+i*m"
        # Construct and init pytmatrix scatterer
        self.scatterer = tmatrix.Scatterer(radius_type=tmatrix.Scatterer.RADIUS_EQUAL_VOLUME, wavelength=self.Lambda, m=m, axis_ratio=self.axis_ratio, 
                                           shape=shape_type,
                                           ddelt=1e-3,  # ddelt = The accuracy of the computations (see doc of pytmatrix)
                                            ndgs=2,     # ndgs = Number of division points used to integrate over the particle surface. Try increasing this if the computation do not converge (see doc of pytmatrix)
                                            n_alpha=5, n_beta=10) # Number of integration points for averaging over the alpha and beta angle when fixed-point orientation averaging is used (see doc of pytmatrix)
        # orientation specific
        if self.is_multi_orientations is False:
            self.scatterer.orient = orientation.orient_single
            self.scatterer.alpha = self.alpha
            self.scatterer.beta = self.beta
        else:
            self.scatterer.orient = orientation.orient_averaged_fixed
            if self.std_orien == 0:
                self.scatterer.or_pdf = orientation.uniform_pdf()
            else:
                self.scatterer.or_pdf = orientation.gaussian_pdf(self.std_orien)
        # size specific
        if self.is_multi_sizes is False:
            self.scatterer.radius = self.radius
        else:
            # see doc of pytmatrix PSDIntegrator class for more information (PSD = Particle Size Distribution)
            self.scatterer.psd_integrator = PSDIntegrator(num_points=num_points,  # The number of different (equally spaced) particle diameters at which to store the amplitude and phase matrices (default is 1024)
                                                          D_max=self.Dmax,  # The maximum diameter for which to store the amplitude and phase matrices
                                                          m_func=None,      # The fractive index as a function of size (None for constant fractive index)
                                                          axis_ratio_func=None) # The horizontal-to-rotational axis ratio as a function of size (None for constant axis ratio)
            self.scatterer.psd = self.size_psd
        # print_scatterer(self.scatterer)
    

    def __Cgeom2PYTMgeom(self, geom):
        """
        Convert Conventional observation geometry tuple to pytmatrix specified observation geometry tuple
        INPUT:
            geom: (theta_s, phi_s, theta_i, phi_i) is the conventional observation geometry tuple
        OUTPUT:
            PYTMgeom: (theta_i, theta_s, phi_i, phi_s, alpha, beta) is the pytmatrix specified observation geometry tuple
        """
        PYTMgeom = (geom[2], geom[0], geom[3], geom[1], self.alpha, self.beta)

        return PYTMgeom


    def phase_matrix(self, geom):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix is not multiplied by n0 (i.e., only for single scatterer), see self.phase_matrix()
        """
        geom = self.__Cgeom2PYTMgeom(geom)
        self.scatterer.set_geometry(geom=geom)
        if self.scatterer.psd_integrator != None:   # multi-sized particles
            self.scatterer.psd_integrator.geometries = (geom, ) # can be tuple of geometries: (geom1, geom2, geom3, ...)
            self.scatterer.psd_integrator.init_scatter_table(self.scatterer)
        # print_scatterer(self.scatterer)
        Z = self.scatterer.get_Z()  # phase matrix
        # Z = np.zeros((4, 4))
        
        # convert phase matrix Z for stokes vector g to phase matrix P for stokes vector I
        P = Mueller_matrix_L2M(Z)
        # Q = np.array([[1., 1, 0, 0],
        #               [1, -1, 0, 0],
        #               [0, 0, 1, 0],
        #               [0, 0, 0, -1]])
        # invQ = np.array([[0.5,  0.5,  0.,  0.],
        #                     [0.5, -0.5, -0., -0.],
        #                     [0.,  0.,  1.,  0.],
        #                     [-0., -0., -0., -1.]])
        # P = invQ @ Z @ Q

        if self.is_multi_sizes is False:
            P = self.n0 * P
        # for multi_sizes, the n0 has been already multiplied equivalently

        return P
