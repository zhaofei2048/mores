""" Improved Integral Equation Model (I2EM) for rough surface scattering.
This code is adpated from the implementation of robbiemallett (https://github.com/robbiemallett/IIEM).
The orginal version is the matlab code published by Ulaby & Long, 2014, "Microwave radar and radiometric remote sensing".

Author: Fei Zhao
Create: 2024-01-16
"""
import numpy as np
from scipy.integrate import dblquad
from scipy.special import erfc
from .rough_spectrum import roughness_spectrum


def Mue_upward_I2EMs(f, angle_s, angle_i, epsr, delta, corr_len, spectrum='exp', x=1.5, shadow_flag=True):
    """Fung 2002 I2EM model for single upward scattering. 

    The model is referenced to Fung, 2002, "An improved IEM model for bistatic scattering from rough surfaces". 
    The transition funciton proposed by Wu et al., 2001, TGRS is also adopted. And for bistatic scattering, the
    averaged transition function in Chen et al. 2010, is used.
    
    Args:
        f: frequency (Hz)
        angle_s: (theta_s, phi_s) (deg) scattering angles (theta_s is angle between
            reflected wave and positive z axis)
        angle_i: (theta_i, phi_i) (deg) incidence and azimuth angles in surface scattering coordinates
        epsr: complex relative dielectric constant (er - 1j*eri)
        delta: RMS height (m) of the rough surface
        corr_len: correlation length (m) of the rough surface
        spectrum: spectrum type, one of {'Gauss', 'exp', 'x-power', 'x-exp'}, default is 'exp'
        x: coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correlation function, default is 1.5
        shadow_flag: True (default) or False for shadowing
    
    Returns:
        Mue: real 4 * 4 Mueller matrix for single scattering
    """
    # customize
    frequency = f / 1e9 # -> GHz
    sigma_h = delta
    CL = corr_len
    theta_i = angle_i[0]
    theta_scat = angle_s[0]
    phi_scat = angle_s[1] - angle_i[1]
    er = epsr
    xx = x

    if (theta_i == theta_scat and phi_scat == 180) or (theta_i == -theta_scat and phi_scat == 0):
        is_bistatic = False
    else:
        is_bistatic = True

    if theta_i == theta_scat or theta_i == -theta_scat:
        theta_scat = theta_scat + 0.001

    if theta_i == 0:
        theta_i = theta_i + 0.001

    ##############################################################
    error = 1.0e8

    sigma_h = sigma_h * 100 # change from m to cm scale
    CL = CL * 100

    mu_r = 1 # relative permeability

    k = 2 * np.pi * frequency / 30 # wavenumber in free space. Speed of light is in cm / sec
    theta = theta_i * np.pi / 180 # transform to radian
    phi = 0
    thetas = theta_scat * np.pi / 180
    phis = phi_scat * np.pi / 180

    ks = k * sigma_h # roughness parameter
    kl = k * CL

    ks2 = ks * ks

    cs = np.cos(theta)
    s = np.sin(theta)


    sf = np.sin(phi)
    cf = np.cos(phi)

    ss = np.sin(thetas)
    css = np.cos(thetas)

    cfs = np.cos(phis)
    sfs = np.sin(phis)

    s2 = s ** 2

    kx = k * s * cf
    ky = k * s * sf
    kz = k * cs

    ksx = k * ss * cfs
    ksy = k * ss * sfs
    ksz = k * css

    # -- reflection coefficients
    rt = np.sqrt(er - s2)
    Rvi = (er * cs - rt) / (er * cs + rt)
    Rhi = (cs - rt) / (cs + rt)


    wvnb = k * np.sqrt((ss * cfs - s * cf) ** 2 + (ss * sfs - s * sf) ** 2)

    Ts = 1

    while error > 1.0e-8:
        Ts = Ts + 1
        error = (ks2 * (cs + css) ** 2) ** Ts / np.math.factorial(Ts)


    # ---------------- calculating roughness spectrum - ----------
    wn, rss = roughness_spectrum(spectrum, Ts, sigma_h, CL, wvnb, xx)
    # (wn, rss) = roughness_spectrum(sp,
    #                                xx,
    #                                wvnb,
    #                                sigma_h,
    #                                CL,
    #                                Ts)


    # ----------- compute R - transition - -----------

    Rv0 = (np.sqrt(er) - 1) / (np.sqrt(er) + 1)
    Rh0 = -Rv0


    Ft = 8 * Rv0 ** 2 * ss * (cs + np.sqrt(er - s2)) / (cs * np.sqrt(er - s2))
    a1 = 0
    b1 = 0


    for n in np.arange(1,Ts+1):
        a0 = (ks * cs) ** (2 * n) / np.math.factorial(n)
        a1 = a1 + a0 * wn[n-1]
        b1 = b1 + a0 * (np.abs(Ft / 2 + 2 ** (n + 1) * Rv0 / cs * np.exp(-(ks * cs) ** 2))) ** 2 *wn[n-1]

    St = 0.25 * (np.abs(Ft)) ** 2 * a1 / b1

    St0 = 1 / (np.abs(1 + 8 * Rv0 / (cs * Ft))) ** 2

    Tf = 1 - St / St0

    # ----------- compute average reflection coefficients - -----------
    # -- these coefficients account for slope effects, especially near the
    # brewster angle.They are not important if the slope is small.

    sigx = 1.1 * sigma_h / CL
    sigy = sigx
    xxx = 3 * sigx

    # print( cs, s, er, s2, sigx, sigy)

    #######################
    # -- select proper reflection coefficients
    if is_bistatic is True:
        def complex_dblquad(func, a, b, c, d):

            def real_func(x,y):
                return np.real(func(x,y))

            def imag_func(x,y):
                return np.imag(func(x,y))

            real_integral = dblquad(real_func, a, b, c, d)
            imag_integral = dblquad(imag_func, a, b, c, d)

            return (real_integral[0] + 1j * imag_integral[0])

    #####################

        Rav = complex_dblquad(lambda Zx,Zy : Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy),
                                -xxx, xxx, -xxx, xxx ) # Integrate over Zx, Zy

        Rah = complex_dblquad(lambda Zx,Zy : Rah_integration(Zx, Zy, cs, s, er, s2, sigx, sigy),
                                -xxx, xxx, -xxx, xxx ) # Integrate over Zx, Zy
        Rav = Rav / (2 * np.pi * sigx * sigy)
        Rah = Rah / (2 * np.pi * sigx * sigy)

        # in this case, it is the bistatic configuration and average R is used
        # Rvt = Rav + (Rv0 - Rav) * Tf
        # Rht = Rah + (Rh0 - Rah) * Tf
        Rvt = Rav
        Rht = Rah
    else: # i.e.operating in backscatter mode
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

    fvv = 2 * Rvt * (s * ss - (1 + cs * css) * cfs) / (cs + css)
    fhh = -2 * Rht * (s * ss - (1 + cs * css) * cfs) / (cs + css)

    # ------- Calculate the Fppup(dn) i(s) coefficients - ---
    (Fvvupi, Fhhupi) = Fppupdn_is_calculations(+1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvups, Fhhups) = Fppupdn_is_calculations(+1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvdni, Fhhdni) = Fppupdn_is_calculations(-1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvdns, Fhhdns) = Fppupdn_is_calculations(-1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)

    qi = k * cs
    qs = k * css

    # ----- calculating Ivv and Ihh - ---

    fvv = fvv.conjugate()
    fhh = fhh.conjugate()

    Fvvupi = Fvvupi.conjugate()
    Fvvups = Fvvups.conjugate()
    Fvvdni = Fvvdni.conjugate()
    Fvvdns = Fvvdns.conjugate()

    Fhhupi = Fhhupi.conjugate()
    Fhhups = Fhhups.conjugate()
    Fhhdni = Fhhdni.conjugate()
    Fhhdns = Fhhdns.conjugate()

    Ivv = np.zeros(Ts, dtype=np.complex_)
    Ihh = Ivv.copy()

    for n in np.arange(1,Ts+1):

        Ivv[n-1] = (kz + ksz) ** n * fvv *np.exp(-sigma_h ** 2 * kz * ksz) + \
        0.25 * (Fvvupi * (ksz - qi) ** (n - 1) *np.exp(-sigma_h ** 2 * (qi ** 2 - qi * (ksz - kz))) +
        Fvvdni* (ksz+qi) ** (n - 1) *np.exp(-sigma_h ** 2 * (qi ** 2 + qi * (ksz - kz))) +
        Fvvups * (kz + qs) ** (n - 1) *np.exp(-sigma_h ** 2 * (qs ** 2 - qs * (ksz - kz))) +
        Fvvdns * (kz - qs) ** (n - 1) *np.exp(-sigma_h ** 2 * (qs ** 2 + qs * (ksz - kz))))

        Ihh[n-1] = (kz + ksz) ** n * fhh *np.exp(-sigma_h ** 2 * kz * ksz) + \
        0.25 * (Fhhupi * (ksz - qi) ** (n - 1) *np.exp(-sigma_h ** 2 * (qi ** 2 - qi * (ksz - kz))) +
        Fhhdni* (ksz+qi) ** (n - 1) *np.exp(-sigma_h ** 2 * (qi ** 2 + qi * (ksz - kz))) +
        Fhhups * (kz + qs) ** (n - 1) *np.exp(-sigma_h ** 2 * (qs ** 2 - qs * (ksz - kz))) +
        Fhhdns * (kz - qs) ** (n - 1) *np.exp(-sigma_h ** 2 * (qs ** 2 + qs * (ksz - kz))))

    # -- Shadowing function calculations

    if (theta_i == theta_scat) & (phi_scat == 180): # i.e.working in backscatter mode
        ct = 1 / np.tan(theta)
        cts = 1 / np.tan(thetas)
        rslp = rss
        ctorslp = float((ct / np.sqrt(2) / rslp).real)
        ctsorslp = float((cts / np.sqrt(2) / rslp).real)

        shadf = 0.5 * (np.exp(-ctorslp ** 2) / np.sqrt(np.pi) / ctorslp - erfc(ctorslp))
        shadfs = 0.5 * (np.exp(-ctsorslp ** 2) / np.sqrt(np.pi) / ctsorslp - erfc(ctsorslp))
        ShdwS = 1 / (1 + shadf + shadfs)
    else:
        ShdwS = 1

    if shadow_flag is False:
        ShdwS = 1

    # ------- calculate the values of sigma_note - -------------

    sigmavv = 0
    sigmahh = 0

    for n in np.arange(1,Ts+1):

        a0 = wn[n-1] / np.math.factorial(n) * sigma_h ** (2 * n)


        sigmavv = sigmavv + np.abs(Ivv[n-1]) ** 2 * a0
        sigmahh = sigmahh + np.abs(Ihh[n-1]) ** 2 * a0



    sigmavv = sigmavv * ShdwS * k ** 2 / 2 * np.exp(-sigma_h ** 2 * (kz ** 2 + ksz ** 2))
    sigmahh = sigmahh * ShdwS * k ** 2 / 2 * np.exp(-sigma_h ** 2 * (kz ** 2 + ksz ** 2))

    # ssv = 10 * np.log10(sigmavv.real)
    # ssh = 10 * np.log10(sigmahh.real)

    # sigma_0_vv = ssv
    # sigma_0_hh = ssh
    Mue = np.zeros((4, 4))
    Mue[0, 0] = sigmavv / (4 * np.pi)
    Mue[1, 1] = sigmahh / (4 * np.pi)

    return Mue


#=======================================================
# Routines for IIEM
def Fppupdn_is_calculations(ud, is_, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs):

    if is_ == 1:

        Gqi = ud * kz
        Gqti = ud * k * np.sqrt(er - s  ** 2)
        qi = ud * kz

        c11 = k * cfs * (ksz - qi)
        c21 = cs * (cfs * (k  ** 2 * s * cf * (ss * cfs - s * cf) + Gqi * (k * css - qi))
                     + k  ** 2 * cf * s * ss * sfs  ** 2)
        c31 = k * s * (s * cf * cfs * (k * css - qi) - Gqi * (cfs * (ss * cfs - s * cf) + ss * sfs ** 2))
        c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

        c12 = k * cfs * (ksz - qi)
        c22 = cs * (cfs * (k  ** 2 * s * cf * (ss * cfs - s * cf) + Gqti * (k * css - qi))
                     + k  ** 2 * cf * s * ss * sfs ** 2)
        c32 = k * s * (s * cf * cfs * (k * css - qi) - Gqti * (cfs * (ss * cfs - s * cf) - ss * sfs ** 2))
        c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))


    elif is_ == 2:

        Gqs = ud * ksz
        Gqts = ud * k * np.sqrt(er - ss  ** 2)
        qs = ud * ksz

        c11 = k * cfs * (kz + qs)
        c21 = Gqs * (cfs * (cs * (k * cs + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c31 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c41 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c51 = -css * (k  ** 2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))

        c12 = k * cfs * (kz + qs)
        c22 = Gqts * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c42 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c52 = -css * (k  ** 2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))


    q = kz
    qt = k * np.sqrt(er - s  ** 2)

    vv = (1 + Rvi) * (-(1 - Rvi) * c11 / q + (1 + Rvi) * c12 / qt) + \
        (1 - Rvi) * ((1 - Rvi) * c21 / q - (1 + Rvi) * c22 / qt) + \
        (1 + Rvi) * ((1 - Rvi) * c31 / q - (1 + Rvi) * c32 / er / qt) + \
        (1 - Rvi) * ((1 + Rvi) * c41 / q - er * (1 - Rvi) * c42 / qt) + \
        (1 + Rvi) * ((1 + Rvi) * c51 / q - (1 - Rvi) * c52 / qt)

    hh = (1 + Rhi) * ((1 - Rhi) * c11 / q - er * (1 + Rhi) * c12 / qt) - \
        (1 - Rhi) * ((1 - Rhi) * c21 / q - (1 + Rhi) * c22 / qt) - \
        (1 + Rhi) * ((1 - Rhi) * c31 / q - (1 + Rhi) * c32 / qt) - \
        (1 - Rhi) * ((1 + Rhi) * c41 / q - (1 - Rhi) * c42 / qt) - \
        (1 + Rhi) * ((1 + Rhi) * c51 / q - (1 - Rhi) * c52 / qt)

    return(vv, hh)


def Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy):

    A = cs + Zx * s
    B = er * (1 + Zx  ** 2 + Zy  ** 2)
    CC = s2 - 2 * Zx * s * cs + Zx  ** 2 * cs  ** 2 + Zy  ** 2

    Rv = (er * A - np.sqrt(B - CC)) / (er * A + np.sqrt(B - CC))

    pd = np.exp(-Zx ** 2 / (2 * sigx  ** 2) - Zy  ** 2 / (2 * sigy  ** 2))
    Rav = Rv * pd

    return Rav


def Rah_integration(Zx, Zy, cs, s, er, s2, sigx, sigy):

    A = cs + Zx * s
    B = er * (1 + Zx  ** 2 + Zy  ** 2)
    CC = s2 - 2 * Zx * s * cs + Zx  ** 2 * cs  ** 2 + Zy  ** 2

    Rh = (A - np.sqrt(B - CC)) / (A + np.sqrt(B - CC))

    pd = np.exp(-Zx  ** 2 / (2 * sigx  ** 2) - Zy  ** 2 / (2 * sigy  ** 2))
    Rah = Rh * pd

    return Rah