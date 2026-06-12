""" Fractional Brownian Motion Surfaces via the Small Slope Approximation

Author: Ruirui Yuan & Fei Zhao
Create: 2026-02-01
Update:
    2026-02-01: First version
    2026-02-26: Add convergence check for series expansion, and use numerical integration if not converged
"""
import numpy as np
import scipy.constants as sci_const
from scipy.special import gamma, j0, rgamma
from scipy.integrate import quad
from .common import qprs2Mueller, R1_shadowing


def Mue_upward_fBmSSA(f, angle_s, angle_i, epsr, s0, H, shadow_flag=True):
    """Fractional Brownian Motion Surfaces via the Small Slope Approximation.

    The model is referenced to Iodice, 2023, "Electromagnetic Scattering from Fractional Brownian Motion Surfaces via the Small Slope Approximation". 
    
    Args:
        f: frequency (Hz)
        angle_s: (theta_s, phi_s) (deg) scattering angles (theta_s is angle between
            reflected wave and positive z axis)
        angle_i: (theta_i, phi_i) (deg) incidence and azimuth angles in surface scattering coordinates
        epsr: complex relative dielectric constant (er - 1j*eri)
        s0: RMS slope of the rough surface at the reference scale 1 m
        H: Hurst exponent of the rough surface
        shadow_flag: True (default) or False for shadowing
    
    Returns:
        Mue: real 4 * 4 Mueller matrix for single scattering
    """
    # print(f"Calculating fBmSSA for f={f} Hz, angle_s={angle_s} deg, angle_i={angle_i} deg, epsr={epsr}, s0={s0}, H={H}, shadow_flag={shadow_flag}")
    # key variables
    theta_i = np.deg2rad(angle_i[0])
    phi_i = np.deg2rad(angle_i[1])
    theta_s = np.deg2rad(angle_s[0])
    phi_s = np.deg2rad(angle_s[1])
    er = epsr

    # if theta_i == theta_s or theta_i == -theta_s:
    #     theta_s = theta_s + np.deg2rad(0.001)
    
    if theta_i == 0:
        theta_i = theta_i + np.deg2rad(0.001)

    mur = 1 # relative permeability
    k = 2 * np.pi * f * np.sqrt(mur * 1) / sci_const.speed_of_light

    cs = np.cos(theta_i)
    s = np.sin(theta_i)
    css = np.cos(theta_s)
    ss = np.sin(theta_s)
    phi_s = phi_s - phi_i
    phi_i = 0
    csfs = np.cos(phi_s)
    sfs = np.sin(phi_s)

    v = css * cs
    ux = s - ss * csfs
    uy = -ss * sfs
    uz = -(css + cs)
    up = np.sqrt(ux**2 + uy**2)
    # print(f"up: {up}")

    # Bragg scattering coefficients
    esi = np.sqrt(er - s**2)
    ess = np.sqrt(er - ss**2)
    Bhh = (er - 1) * csfs / ((css + ess) * (cs + esi))
    Bvh = sfs * (er - 1) * ess / ((er*css + ess) * (cs + esi))
    Bhv = sfs * (er - 1) * esi / ((css + ess) * (er*cs + esi))
    Bvv = (er - 1) * (ess * esi * csfs - er * ss * s) / ((er*css + ess) * (er*cs + esi))
    Bqp_ = [Bvv, Bvh, Bhv, Bhh]
    del s

    Sqprs = np.zeros((4, 4), dtype=complex) # single-scattering term
    for i in range(4): # i for qp
        Bqp = Bqp_[i]
        for j in range(4): # j for rs
            Brs = Bqp_[j]
            Bqprsn = Bqp * np.conj(Brs)
            Sqprs[i, j] = Bqprsn
    
    # 计算散射强度
    C0 = 2 * np.abs(2 * k * v / uz)**2 / (4 * np.pi)

    # 计算积分项
    Omega = 0.5 * k**2 * uz**2 * s0**2 / (k**2 * up**2)**H
    myeps = 1e-32
    # print(f'Omega: {Omega}')
    is_converge = False
    if Omega < 0.1:
        # print("Small Omega")
        n = 1
        factorial_n1 = 1
        an = 1
        y = 0
        while np.abs(an) > myeps:
            factorial_n1 = factorial_n1 * (n - 1) if n > 1 else 1
            # an = (-1)**(n+1) * 2**(2*n*H) / factorial_n1 * gamma(1 + n*H) / \
            #     gamma(1 - n*H) * Omega**n / (k**2 * up**2)
            # rgamma for the reciprocal gamma function
            # print(f"n={n}, H={H}")
            # print(f"1+n*H: {1+n*H}, 1-n*H: {1-n*H}")
            # print(f"gamma(1+n*H): {gamma(1+n*H)}, rgamma(1-n*H): {rgamma(1-n*H)}")
            an = (-1)**(n+1) * 2**(2*n*H) / factorial_n1 * gamma(1 + n*H) * \
                rgamma(1 - n*H) * Omega**n / (k**2 * up**2)
            y = y + an
            n = n + 1
            if n > 100:
                break
        if n <= 100:
            y = y * 2 * H
            is_converge = True
    elif Omega > 10:
        # print("Large Omega")
        n = 0
        factorial_n = 1
        an = 1
        y = 0
        while np.abs(an) > myeps:
            factorial_n = factorial_n * n if n > 1 else 1
            an = (-1)**(n) / (2**(2*n) * factorial_n**2) * gamma((n+1)/H) * \
                (k**2 * up**2)**n / (0.5 * k**2 * uz**2 * s0**2)**((n+1)/H)
            y = y + an
            n = n + 1
            if n > 100:
                break
        if n <= 100:
            y = y / (2 * H)
            is_converge = True

    if is_converge == False:    # 不使用级数展开直接积分
        integrand = lambda x: np.exp(-0.5 * k**2 * uz**2 * s0**2 * x**(2*H)) * j0(k * up * x) * x
        y = quad(integrand, 0, np.inf)[0]

    Ms = qprs2Mueller(C0 * y * Sqprs)

    # IEM坐标系转换到FSA坐标系
    U = np.diag([1, 1, -1, -1])
    Mue = U @ Ms

    # shadowing
    if shadow_flag is True:
        # 计算其在波长尺度的RMS slope
        rss = s0 * (k / (2 * np.pi))**(1 - H)
        shadow = R1_shadowing(theta_i, rss)
        Mue = Mue * shadow

    return Mue