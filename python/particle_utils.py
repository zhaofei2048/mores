"""
Author: Fei Zhao
Create: 2023-08-28

Description:
    Routine functions for particles
"""

from pytmatrix import tmatrix, tmatrix_aux
from pytmatrix.scatter import sca_xsect, ext_xsect, ssa
import numpy as np

def extinction_efficiency(epsr, Chi, h_pol=True, epsr_bg=1.0):
    """
    Calculate the extinction efficiency factor of a sphere
    INPUT:
        epsr: the complex dielectric constant of the sphere
        Chi: = 2*pi*r/lambda_b
        h_pol: True for horizontal polarization incident and False for vertical polarization incident
        epsr_bg: the dielectric constant of background medium (default assumed the sphere in air)
    OUTPUT:
        ext_eff: the extinction efficiency factor

    See Ulaby 2014, p342 for definitions
    """
    epsr = epsr / epsr_bg
    m = np.conj(np.sqrt(epsr))  # pytmatrix convention
    wavelength = 1.0/np.sqrt(np.real(epsr_bg))
    radius = Chi * wavelength / (2*np.pi)
    scatterer = tmatrix.Scatterer(radius=radius, wavelength=wavelength, m=m, axis_ratio=1.0)
    if h_pol is True:
        scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
    else:
        scatterer.set_geometry(tmatrix_aux.geom_vert_forw)

    A = np.pi * radius**2   # the geometry cross section of the sphere
    ext_eff = ext_xsect(scatterer, h_pol=h_pol) / A

    return ext_eff

def scattering_efficiency(epsr, Chi, h_pol=True, epsr_bg=1.0):
    """
    Calculate the scattering efficiency factor of a sphere
    INPUT:
        epsr: the complex dielectric constant of the sphere
        Chi: = 2*pi*r/lambda_b
        h_pol: True for horizontal polarization incident and False for vertical polarization incident
        epsr_bg: the dielectric constant of background medium (default assumed the sphere in air)
    OUTPUT:
        sca_eff: the extinction efficiency factor
    
    See Ulaby 2014, p342 for definitions
    """
    epsr = epsr / epsr_bg
    m = np.conj(np.sqrt(epsr))  # pytmatrix convention
    wavelength = 1.0/np.sqrt(np.real(epsr_bg))
    radius = Chi * wavelength / (2*np.pi)
    scatterer = tmatrix.Scatterer(radius=radius, wavelength=wavelength, m=m, axis_ratio=1.0)

    A = np.pi * radius**2   # the geometry cross section of the sphere
    sca_eff = sca_xsect(scatterer, h_pol=h_pol) / A

    return sca_eff

def single_scattering_albedo(epsr, Chi, h_pol=True, epsr_bg=1.0):
    """
    Calculate the SSA(single scattering albedo) of a sphere
    INPUT:
        epsr: the complex dielectric constant of the sphere
        Chi: = 2*pi*r/lambda_b
        h_pol: True for horizontal polarization incident and False for vertical polarization incident
        epsr_bg: the dielectric constant of background medium (default assumed the sphere in air)
    OUTPUT:
        ssa: the extinction efficiency factor
    """
    epsr = epsr / epsr_bg
    m = np.conj(np.sqrt(epsr))  # pytmatrix convention
    wavelength = 1.0/np.sqrt(np.real(epsr_bg))
    radius = Chi * wavelength / (2*np.pi)
    scatterer = tmatrix.Scatterer(radius=radius, wavelength=wavelength, m=m, axis_ratio=1.0)
    if h_pol is True:
        scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
    else:
        scatterer.set_geometry(tmatrix_aux.geom_vert_forw)

    SSA = ssa(scatterer, h_pol=h_pol)

    return SSA


def scattering_cross_section_rayleigh_sphere(Lambda, epsilon_r, Chi):
    """Calculate the sca_xsec of sphere under Rayleigh Approximation
    
    Args:
        Lambda: wavelength in the background medium, not in the air
        epsilon_r: relative epsr of particle to background medium
        Chi: =2*pi*r / Lambda where r is the radius of the sphere (Can be a array)
    
    Returns:
        Qs: Scattering cross section of single sphere
    """
    # Ulaby 2014, eq. 8.45(a), p344
    K = (epsilon_r - 1) / (epsilon_r + 2)
    Qs = 2 * Lambda**2 / (3 * np.pi) * Chi**6 * np.abs(K)**2

    return Qs


def absorption_cross_section_rayleigh_sphere(Lambda, epsilon_r, Chi):
    """Calculate the a_xsec of sphere under Rayleigh Approximation
    
    Args:
        Lambda: wavelength in the background medium, not in the air
        epsilon_r: relative epsr of particle to background medium
        Chi: =2*pi*r / Lambda where r is the radius of the sphere (Can be a array)
    
    Returns:
        Qa: Scattering cross section of single sphere
    """
    # Ulaby 2014, eq. 8.45(b), p344
    K = (epsilon_r - 1) / (epsilon_r + 2)
    Qa = Lambda**2 / (np.pi) * Chi**3 * np.imag(-K)

    return Qa