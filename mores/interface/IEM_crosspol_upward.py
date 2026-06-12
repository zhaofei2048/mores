"""IEM-based cross-polarized scattering coefficients, i.e., sigma0_HV.
This code is adpated from the implementation of robbiemallett (https://github.com/robbiemallett/IIEM).
The orginal version is the matlab code published by Ulaby & Long, 2014, "Microwave radar and radiometric remote sensing".

Author: Fei Zhao
Create: 2024-01-16
"""
import math

import numpy as np
from scipy.integrate import dblquad
from scipy.special import erfc, erf
from scipy.special import gamma, kv as besselk, jn as besselj
import itertools


def sigma0_VH_IEM(f, theta_i, epsr, delta, corr_len, spectrum='exp', x=1.5, shadow_flag=True):
    """Cross-polarized scattering coefficients based on IEM. 

    The model is referenced to Fung, 1994, "Microwave scattering and emission models and their appliations". 
    Note the scattering coefficients is in linear scale, not the dB scale.
    
    Args:
        f: frequency (Hz)
        theta_i: incidence angle (deg) in surface scattering coordinates
        epsr: complex relative dielectric constant (er - 1j*eri)
        delta: RMS height (m) of the rough surface
        corr_len: correlation length (m) of the rough surface
        spectrum: spectrum type, one of {'Gauss', 'exp', 'x-power', 'x-exp'}, default is 'exp'
        x: coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correlation function, default is 1.5
        shadow_flag: True (default) or False for shadowing. Note the shadow is currently not applied for the cross-polarized scattering.
    
    Returns:
        sigma0_HV: Cross-polarized scattering coefficients in linear scale
    """
    # customize
    fr = f / 1e9 # -> GHz
    sig = delta
    L = corr_len
    theta_d = theta_i
    er = epsr
    sp = spectrum
    xx = x

    #################################################
    sig = sig * 100 # change to cm scale
    L = L * 100 # change to cm scale

    # - fr: frequency in GHz
    # - sig: rms height of surface in cm
    # - L: correlation length of surface in cm
    # - theta_d: incidence angle in degrees
    # - er: relative permittivity
    # - sp: type of surface correlation function

    error = 1.0e8

    k = 2 * np.pi * fr / 30 # wavenumber in free space.Speed of light is in cm / sec
    theta = theta_d * np.pi / 180 # transform to radian


    ks = k * sig # roughness parameter
    kl = k * L

    ks2 = ks * ks
    kl2 = kl ** 2

    cs = np.cos(theta)
    s = np.sin(theta + 0.001)

    s2 = s ** 2

    # -- calculation of reflection coefficints
    rt = np.sqrt(er - s2)


    rv = (er * cs - rt) / (er * cs + rt)


    rh = (cs - rt) / (cs + rt)

    rvh = (rv - rh) / 2

    # print(rt, rv, rh, rvh)
    # exit()


    # -- rms slope values
    sig_l = sig / L
    if sp.lower() == 'exp': # -- exponential correl func
        rss = sig_l


    if sp.lower() == 'gauss': # -- Gaussian correl func
        rss = sig_l * np.sqrt(2)


    if sp.lower() == 'x-power': # -- 1.5 - power spectra correl func
        rss = sig_l * np.sqrt(2 * xx)

    # --- Selecting number of spectral components of the surface roughness
    # if auto == 0:
        # n_spec = 15 # numberofterms to include in the surface roughness spectra
    auto = 1
    if auto == 1:
        n_spec = 1
        while error > 1.0e-8:
            n_spec = n_spec + 1
            error = (ks2 * (2 * cs) ** 2) ** n_spec / math.factorial(n_spec)

    # -- calculating shadow consideration in single scat(Smith, 1967)

    ct = 1 / np.tan(theta + 0.001)
    farg = (ct / np.sqrt(2) / rss).real

    if shadow_flag is True:
        Gamma = 0.5 * (np.exp(-float(farg.real) ** 2) / 1.772 / float(farg.real) - erfc(float(farg.real)))

        Shdw = 1 / (1 + Gamma)

    # -- calculating multiple scattering contribution
    # ------ a double integration function

    factorials = {}
    for number in np.arange(1,n_spec+1):
        factorials[number] = math.factorial(number)

    svh = dblquad(lambda phi, r : xpol_integralfunc_vec(r, phi, sp, xx, ks2, cs, s,
                                                            kl2, L, er, rss, rvh, n_spec, factorials),
                          0.1, 1, lambda x : 0, lambda x : np.pi)[0]

    sigma0_VH = svh * 1.e-5 # # un - scale after rescalingin the integrand function.

    return sigma0_VH


def x_exponential_spectrum(z,wvnb,L,n,xx):

    tmp = np.exp(-abs(z) ** xx) * besselj(0, z*wvnb*L/(n**(1/xx)))*z

    return(tmp)


def spectrm2(sp, xx, kl2, L, rx, ry, s, np_, nr):

    wm = np.zeros((np_,nr))

    if sp.lower() == 'exp':  # exponential
        for n in np.arange(1,np_+1):
            wm[n-1,:] = n* kl2/(n**2 + kl2 *((rx+s)**2+ry**2))**1.5


    if sp.lower() == 'gauss':  #  gaussian
        for n in np.arange(1,np_+1):
            wm[n-1,:] = 0.5 * kl2/n* np.exp(-kl2*((rx+s)**2 + ry**2)/(4*n))


    if sp.lower()== 'x-power': # x-power
        for n in np.arange(1,np_+1):
            wm[n-1,:] = kl2/(2.**(xx*n-1)*gamma(xx*n))* ( ( (rx+s)**2.
              + ry**2)*L)**(xx*n-1)* besselk(-xx*n+1, L*((rx+s)**2 + ry**2))

    return wm


def spectrm1(sp, xx, kl2, L, rx, ry, s, np_, nr):

    wn = np.zeros((np_,nr))

    if sp.lower() == 'exp':  # exponential
        for n in np.arange(1,np_+1):
            wn[n-1, :] = n* kl2/(n**2 + kl2 *((rx-s)**2+ry**2))**1.5


    if sp.lower() == 'gauss':  #  gaussian
        for n in np.arange(1,np_+1):
            wn[n-1,:] = 0.5 * kl2/n* np.exp(-kl2*((rx-s)**2 + ry**2)/(4*n))

    if sp.lower() == 'x-power':  # x-power
        for n in np.arange(1,np_+1):
            wn[n-1,:] = kl2/(2.**(xx*n-1)*gamma(xx*n))* ( ( (rx-s)**2.
              + ry**2)*L)**(xx*n-1)* besselk(-xx*n+1, L*((rx-s)**2 + ry**2))

    return wn


def xpol_integralfunc_vec(r, phi, sp, xx, ks2, cs,s, kl2, L, er, rss, rvh, n_spec, factorials):

    cs2 = cs **2

    r2 = r ** 2

    if type(r) == float:
        nr = 1
    else:
        nr = len(r)


    sf = np.sin(phi)
    csf = np.cos(phi)
    rx = r * csf
    ry = r * sf

    #-- calculation of the field coefficients
    rp = 1 + rvh
    rm = 1 - rvh

    q = np.sqrt(1.0001 - r2)
    qt = np.sqrt(er - r2)

    a = rp /q
    b = rm /q
    c = rp /qt
    d = rm /qt

    #--calculate cross-pol coefficient
    B3 = rx * ry /cs
    fvh1 = (b-c)*(1- 3*rvh) - (b - c/er) * rp
    fvh2 = (a-d)*(1+ 3*rvh) - (a - d*er) * rm
    Fvh = ( abs( (fvh1 + fvh2) *B3)) **2


    #-- calculate shadowing func for multiple scattering

    au = (q /r /1.414 /rss)



    fsh = (0.2821/au) *np.exp(-au **2) -0.5 *(1- erf(au))
    sha = 1./(1 + fsh)

    #-- calculate expressions for the surface spectra
    wn = spectrm1(sp, xx, kl2, L, rx, ry, s, n_spec, nr)
    wm = spectrm2(sp, xx, kl2, L, rx, ry, s, n_spec, nr)


    #--compute VH scattering coefficient
    acc = np.exp(-2* ks2 *cs2) /(16 * np.pi)

    vhmnsum = 0
    my_range = np.arange(1,n_spec+1)
    for (n, m) in itertools.product(my_range, my_range):
        vhmnsum = vhmnsum + wn[n-1,:]*wm[m-1,:] * (ks2*cs2) **(n+m) /factorials[n]/factorials[m]

    VH = 4 * acc * Fvh * vhmnsum * r

    y = VH * sha

    y =y * 1.e5 # rescale so dblquad() works better.

    return y