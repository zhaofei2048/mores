"""Surface Mie spherical Rocks scattering calculation

Author: Fei Zhao
Create: 2024-05-01
"""
import numpy as np
import scipy.constants as sci_const
from .miepython_utils import mie_cross_sections_psd, mie_cross_sections, mie_phase_matrix_elements_psd
from ..utils.postprocessing import Mueller_matrix_L2M


class SurfaceMieRocks:
    """Spherical rocks on surface"""
    def __init__(self, f, epsr_particle, area_psd, Dmax):
        """
        
        Args:
            f: frequency, Hz
            epsr_particle:
            area_psd:
            Dmax:
        """
        self.Lambda = sci_const.speed_of_light / f
        self.refractive_index = np.sqrt(epsr_particle)
        self.size_psd = area_psd
        if Dmax > 10 * self.Lambda:
            self.Dmax = 10 * self.Lambda
        else:
            self.Dmax = Dmax

        # Calculate the backscattering phase matrix
        Cext, Csca, Cback = mie_cross_sections_psd(m=self.refractive_index, 
                                                       wavelength=self.Lambda, 
                                                       psd=self.size_psd, 
                                                       Dmin=None, 
                                                       Dmax=self.Dmax)
        
        Z = np.zeros((4, 4))
        Z[0, 0] = Cback / (4 * np.pi)
        Z[1, 1] = Cback / (4 * np.pi)
        Z[2, 2] = -Cback / (4 * np.pi)
        Z[3, 3] = -Cback / (4 * np.pi)

        self.back_phase_matrix = Mueller_matrix_L2M(Z)


    def phase_matrix(self, geom=None):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        INPUT:
            geom (tuple): (not needed) observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix has already been multiplied by n0
        """
        ths, phis, thi, phi = np.deg2rad(geom)
        # the directional norm vector
        ki_ = np.array([np.sin(thi)*np.cos(phi), np.sin(thi)*np.sin(phi), np.cos(thi)])
        ks_ = np.array([np.sin(ths)*np.cos(phis), np.sin(ths)*np.sin(phis), np.cos(ths)])
        mu = np.sum(ki_ * ks_)
        if mu == -1.0:  # backscattering
            P = self.back_phase_matrix
        else:
            Z = mie_phase_matrix_elements_psd(m=self.refractive_index, 
                                              wavelength=self.Lambda, 
                                              psd=self.size_psd, 
                                              Dmin=None, 
                                              Dmax=self.Dmax, 
                                              mu=mu)
            P = Mueller_matrix_L2M(Z)
            
        return P