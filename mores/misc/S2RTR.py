"""
Author: Fei Zhao
Create: 2023-08-29

Description:
    The single-scattering radiative-transfer model with Rayleigh particles (Ulaby2014, p466)
    aka. Heuristic Single-Scattering Model for Vegetation
"""

import numpy as np


def optical_depth(d, theta_i, ke):
    """
    Optical depth / attenuation of the medium due to extinction
    INPUT:
        d: propagation distance (m)
        theta_i: zenith angle of propagation direction (deg)
        ke: volume extinction coefficient under give polarization (Np/m)
    OUTPUT:
        tau_p: optical depth (Np)
    """
    thi = np.deg2rad(theta_i)
    tau_p = ke * d / np.cos(thi)

    return tau_p


def gamma_trans(d, theta_i, ke):
    """
    One way oblique transmittivity
    INPUT:
        d: propagation distance (m)
        theta_i: zenith angle of propagation direction (deg)
        ke: volume extinction coefficient under given polarization (Np/m)
    OUTPUT:
        Gamma: one-way oblique transmittivity (atteuation rate)
    """
    tau_p = optical_depth(d, theta_i, ke)
    Gamma = np.exp(-tau_p)

    return Gamma


class S2RTR:
    """
    The single-scattering radiative-transfer model with Rayleigh sphere particles
    aka. Heuristic Single-Scattering Model for Vegetation
    """
    def __init__(self, f, d, ke, ks, epsr_ground, delta, corr_len, scatterer_type='RAYLEIGH'):
        """
        INPUT:
            f: frequency (Hz) for ground scattering
            d: depth (m) of the layer
            ke: volume extinction coefficient under given polarization (Np/m), note this is not the extinction cross section Qe for single particle
            ks: volume scattering coefficient under given polarization
            scatterer_type: 'RAYLEIGH' (default) for Rayleigh sphere and 'ISOTROPIC' for isotropic scatterers
            epsr_ground: dielectric constant of ground soil
            delta: RMS height of the ground surface
            corr_len: correlation length of the ground surface
        """
        self.freq = f
        self.thickness = d
        self.ke = ke
        self.ks = ks
        self.scatter_type = scatterer_type

        # the volume backscattering coefficient
        if self.scatter_type == 'ISOTROPIC':
            self.sigma_vback = self.ks
            self.sigma_vbist = self.ks
        else:
            self.sigma_vback = 1.5 * self.ks
            self.sigma_vbist = 1.5 * self.ks


    def sigma0_canopy(self, theta_i):
        """
        backscattering coefficient of canopy contribution
        INPUT:
            theta_i: incident angle (deg)
        OUTPUT:
            sigma0: backscattering coefficeint in linear scale
        """
        # Ulaby2014, p464, eq. 11.10
        
        thi = np.deg2rad(theta_i)
        Gamma = gamma_trans(self.thickness, theta_i, self.ke)
        sigma0 = (self.sigma_vback * np.cos(thi) / (2*self.ke)) * (1 - Gamma * Gamma)

        return sigma0


    def sigma0_ground(self, theta_i):
        # thi = np.deg2rad(theta_i)
        # Gamma = gamma_trans(self.thickness, theta_i, self.ke)
        # sigma0_IEM = 
        # sigma0 = Gamma * Gamma * sigma0_IEM
        pass