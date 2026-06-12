"""Wrapper functions for MiePython"""
import numpy as np
import scipy
import scipy.constants as sci_const
from scipy.integrate import quad_vec
import miepython


def mie_cross_sections(m, wavelength, r):
    """
    Calculate the cross sections for a sphere where m or r maybe arrays

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        r: the radius of the sphere [m]
    
    Returns:
        Cext: extincation cross section
        Csca: scattering cross section
        Cback: backscattering cross section
    """
    geometric_cross_section = np.pi * r**2
    x = 2 * np.pi * r / wavelength
    qext, qsca, qback, g = miepython.efficiencies_mx(m=m, x=x)
    Cext = qext * geometric_cross_section
    Csca = qsca * geometric_cross_section
    Cback = qback * geometric_cross_section

    return Cext, Csca, Cback


def _mie_cross_sections_weighted(m, wavelength, diameters, psd):
    """
    Calculate the weighted cross sections, auxilury function for weighted integration

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        diameters: diameters of the spheres, can be array
        psd: psd(D) should return the weight for the particle with diameter of D
    Returns:
        Cext_w: weighted extinction cross section
        Csca_w: weighted scattering cross section
        Cback_w: weighted backscattering cross section
    """
    weight = psd(diameters)
    Cext, Csca, Cback = mie_cross_sections(m=m, wavelength=wavelength, r=diameters*0.5)
    Cext_w = Cext * weight
    Csca_w = Csca * weight
    Cback_w = Cback * weight

    return np.array([Cext_w, Csca_w, Cback_w])


def mie_cross_sections_psd(m, wavelength, psd, Dmin=None, Dmax=None):
    """
    Calculate cross sections of particles with particle size distribution.

    Ci_out = quad(Ci*psd, Dmin, Dmax) where Ci is the cross section (e.g., Cext, Csca, Cback)

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        psd: psd(D) should return the weight for the particle with diameter of D
        Dmin: minimum diameter for the integration
        Dmax: maximum diameter for the integration
    
    Returns:
        Cext, Csca, Cback: weighted integration of the Cross section with diameter
    """
    if Dmin is None:
        Dmin = 1e-6 * wavelength
    if Dmax is None:
        Dmax = 1e3 * wavelength

    calc_cross_sections = lambda D: _mie_cross_sections_weighted(m=m, wavelength=wavelength, diameters=D, psd=psd)
    # r, err, info = quad_vec(calc_cross_sections, Dmin, Dmax, full_output=True)
    r, err = quad_vec(calc_cross_sections, Dmin, Dmax)
    Cext = r[0]
    Csca = r[1]
    Cback = r[2]
    
    return Cext, Csca, Cback


def mie_phase_matrix_Csca(m, wavelength, r, mu=-1):
    """
    mie_phase_matrix with Csca normalization

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        r: the radius of the sphere [m]
        mu: cosin of the scattering angle, cosd(0) for forward and cosd(180) for backward scattering
            default is -1 for backward scattering
    
    Returns:
        Z: phase matrix normalized to Csca, i.e., in the backscattering direciton, 4*np.pi*Z[0,0] == Cback
    """
    geometric_cross_section = np.pi * r**2
    x = 2 * np.pi * r / wavelength
    Z = miepython.phase_matrix(m=m, x=x, mu=mu, norm='qsca') * geometric_cross_section
    # print(Z)
    return Z


def mie_phase_matrix_elements(m, wavelength, r, mu=-1):
    """
    Calculate the phase matrix elements S11, S12, S33, S34

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        r: the radius of the sphere [m]
        mu: cosin of the scattering angle, cosd(0) for forward and cosd(180) for backward scattering
            default is -1 for backward scattering

    Returns:
        S11: 0.5 * (|S2|^2 + |S1|^2 )
        S12: 0.5 * (|S2|^2 - |S1|^2 )
        S33: 0.5 * (np.conj(S2)*S1 + np.conj(S1)*S2)
        S34: 0.5j * (np.conj(S2)*S1 - np.conj(S1)*S2)
    """
    Z = mie_phase_matrix_Csca(m, wavelength, r, mu=mu)
    S11 = Z[0, 0]
    S12 = Z[0, 1]
    S33 = Z[2, 2]
    S34 = Z[2, 3]

    return S11, S12, S33, S34


def _mie_phase_matrix_elements_weighted(m, wavelength, diameters, psd, mu=-1):
    """
    Calculate the weighted phase matrix elements S11, S12, S33, S34, auxilury function for weighted integration.

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        diameters: diameter of the sphere
        psd: psd(D) should return the weight for the particle with diameter of D
        mu: cosin of the scattering angle, cosd(0) for forward and cosd(180) for backward scattering
            default is -1 for backward scattering
    Returns:
        S11_w: 0.5 * (|S2|^2 + |S1|^2 )
        S12_w: 0.5 * (|S2|^2 - |S1|^2 )
        S33_w: 0.5 * (np.conj(S2)*S1 + np.conj(S1)*S2)
        S34_w: 0.5j * (np.conj(S2)*S1 - np.conj(S1)*S2)
    """
    S11, S12, S33, S34 = mie_phase_matrix_elements(m=m, wavelength=wavelength, r=diameters*0.5, mu=mu)
    weight = psd(diameters)
    S11_w = S11 * weight
    S12_w = S12 * weight
    S33_w = S33 * weight
    S34_w = S34 * weight

    return np.array([S11_w, S12_w, S33_w, S34_w])


def mie_phase_matrix_elements_psd(m, wavelength, psd, Dmin=None, Dmax=None, mu=-1):
    """
    Calculate the phase matrix of particles with "psd" particle size distribution funciton.

    Args:
        m: the complex index of refraction of the sphere
        wavelength:  [m]
        psd: psd(D) should return the weight for the particle with diameter of D
        Dmin: minimum diameter for the integration
        Dmax: maximum diameter for the integration
        mu: cosin of the scattering angle, cosd(0) for forward and cosd(180) for backward scattering
            default is -1 for backward scattering
    Returns:
        Z: weigted phase matrix in FSA
    """
    if Dmin is None:
        Dmin = 1e-6 * wavelength
    if Dmax is None:
        Dmax = 1e3 * wavelength

    calc_phase_matrix_elements = lambda D: _mie_phase_matrix_elements_weighted(m=m, 
                                                                               wavelength=wavelength, 
                                                                               diameters=D, psd=psd, mu=mu)
    r, err = quad_vec(calc_phase_matrix_elements, Dmin, Dmax)
    S11, S12, S33, S34 = r[0], r[1], r[2], r[3]

    Z = np.zeros((4, 4))
    Z[0, 0] = S11
    Z[1, 1] = Z[0, 0]
    Z[0, 1] = S12
    Z[1, 0] = Z[0, 1]
    Z[2, 2] = S33
    Z[3, 3] = Z[2, 2]
    Z[2, 3] = S34
    Z[3, 2] = -Z[2, 3]

    return Z