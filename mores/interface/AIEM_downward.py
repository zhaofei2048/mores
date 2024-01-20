""" Advanced Integral Equation Model (AIEM) for rough surface scattering

Author: Fei Zhao
Create: 2024-01-16
"""
import numpy as np
import scipy.constants as sci_const
from .rough_spectrum import roughness_spectrum
from .common import qprs2Mueller, R1_shadowing
from .common import Ccoeff, Bcoeff, calc_n1_n2_g


def Mue_downward_AIEMs(f, angle_t, angle_i, epsr, delta, corr_len, spectrum='exp', x=1.5, shadow_flag=True):
    """Chen 2003 AIEM model for single upward scattering.

    The model is referenced to Chen, 2003, "Emission of rough surfaces calculated by the integral equation method 
    with comparison to three-dimensional moment method simulations". 
    The transition funciton proposed by Wu et al., 2001, TGRS is also adopted.
    
    Args:
        f: frequency (Hz)
        angle_t: (theta_t, phi_t) (deg) transmitted scattering angles (theta_t is angle between
           reflected wave and negative z axis)
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
    # if angle_s[0] == -angle_i[0] or angle_s[0] == angle_i[0]:   # 永远不让theta_i和theta_s严格相等
    #     angle_s[0] = angle_s[0] + 0.001; # 确保数值稳定
    
    # key variables
    theta_i = np.deg2rad(angle_i[0])
    phi_i = np.deg2rad(angle_i[1])
    theta_s = np.deg2rad(180 - angle_t[0])
    phi_s = np.deg2rad(angle_t[1])
    er = epsr
    mu1 = 1
    mu2 =  1
    mur = 1 # relative permeability

    eta_r = np.sqrt(mur / er) # relative wave impedance
    eta1 = np.sqrt(mu1 / 1)
    eta2 = np.sqrt(mu2 / er)
    eta_ist = eta_r # 计算透射场则为eta_r

    k1 = 2 * np.pi * f * np.sqrt(mu1 * 1) / sci_const.speed_of_light
    k2 = 2 * np.pi * f * np.sqrt(mu1 * epsr) / sci_const.speed_of_light
    k = k1
    kdel = k * delta # real?
    kdel2 = kdel * kdel

    cs = np.cos(theta_i)
    s = np.sin(theta_i)
    css = np.cos(theta_s)
    ss = np.sin(theta_s)
    phi_s = phi_s - phi_i
    phi_i = 0
    csfs = np.cos(phi_s)
    sfs = np.sin(phi_s)

    # 波矢量
    kx = k1 * s
    ky = 0
    kz = k1 * cs
    ksx = k2 * ss * csfs
    ksy = k2 * ss * sfs
    ksz = k2 * css
    ki_ = np.array([s, 0, -cs])  # 带_的表示矢量
    ks_ = np.array([ss * csfs, ss * sfs, css])
    # 入射、散射极化矢量 (IEM coordinate)
    h_ = np.array([0, 1, 0])
    v_ = np.array([cs, 0, s])
    hs_ = np.array([sfs, -csfs, 0])
    vs_ = np.array([-css * csfs, -css * sfs, ss])

    Zxf = - (ksx - kx) / (ksz + kz)
    Zyf = - (ksy - ky) / (ksz + kz)

    n_ = np.array([-Zxf, -Zyf, 1]) # local normal vector, but not normalized
    normlized_n_ = n_ / np.linalg.norm(n_)
    t_ = np.cross(ki_, normlized_n_)
    normt = np.linalg.norm(t_)
    t_ = t_ / normt

    # 确定迭代次数
    Nmax = 1
    myeps = 1e-32

    err = kdel2
    while (np.abs(err) > myeps):
        Nmax = Nmax + 1
        err = err * kdel2 / Nmax
    
    # 计算粗糙谱
    Kx = ksx - kx
    Ky = ksy - ky
    wvnb = np.sqrt(np.abs(Kx**2 + Ky**2))
    Wn, rss = roughness_spectrum(spectrum, Nmax, delta, corr_len, wvnb, x)

    #============================================================================
    # Fresnel coefficients
    #--------------------------
    # R(theta_i): reflection coefficients based on the incidence angle
    #--------------------------
    tmp1 = np.sqrt(er - s*s)
    rvi = (er * cs - tmp1) / (er * cs + tmp1)   # rv = Hv_s / Hv_i = Ev_s / Ev_i
    tmp2 = np.sqrt(mur * er - s*s);
    rhi = (mur * cs - tmp2) / (mur * cs + tmp2) # rh = Eh_s / Eh_i
    rvhi = (rvi - rhi) / 2.0

    #--------------------------
    # R(theta_sp): reflection coefficients based on the specular angle
    #--------------------------
    # 方法1）直接计算局部入射角余弦（stationary approximation），会出现cssp<0，进而出现奇异点dip
    # cssp = dot(-ki_, normlized_n_);
    # 方法2）计算局部透射角余弦后再计算局部入射角余弦
    csspt = np.dot(ks_, -normlized_n_)
    sspt = np.sqrt(1 - csspt**2)
    ssp = sspt * np.sqrt(er)
    cssp = np.sqrt(1 - ssp**2)

    ssp = np.sqrt(1 - cssp * cssp)
    tmp1 = np.sqrt(er - ssp*ssp)
    rvsp = (er * cssp - tmp1) / (er * cssp + tmp1) # rv = Hv_s / Hv_i = Ev_s / Ev_i
    tmp2 = np.sqrt(mur * er - ssp*ssp)
    rhsp = (mur * cssp - tmp2) / (mur * cssp + tmp2) # rh = Eh_s / Eh_i
    rvhsp = (rvsp - rhsp) / 2.0

    #--------------------------
    # R(0): reflection coefficients based on the 0 incidence angle
    #--------------------------
    rv0 = (np.sqrt(er) - 1.0) / (np.sqrt(er) + 1.0)
    rh0 = -rv0

    #======================================================================
    # Calculate the Kirchhoff coefficients
    # rv = rvsp
    # rh = rhsp

    d_ = np.cross(ki_, t_)
    hsnv = np.dot(hs_, np.cross(n_, v_)) # = 1 when backscattering
    hsnh = np.dot(hs_, np.cross(n_, h_))
    hsnt = np.dot(hs_, np.cross(n_, t_))
    hsnd = np.dot(hs_, np.cross(n_, d_))
    vsnh = np.dot(vs_, np.cross(n_, h_)) # = -1 when backscattering
    vsnv = np.dot(vs_, np.cross(n_, v_))
    vsnt = np.dot(vs_, np.cross(n_, t_))
    vsnd = np.dot(vs_, np.cross(n_, d_))
    hd = np.dot(h_, d_)
    vt = np.dot(v_, t_)

    vpara = (k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist)

    # prepare for factorial calc (avoid overflow)
    nksz = ksz / k
    nkz = kz / k
    nv = np.arange(1, Nmax+1, 1)
    # 计算(k*sigma)**(2*n) / n! = (ksigma2/1) * (ksigma2/2) * (ksigma2/3) * ... * (ksigma2/n)  
    kdel2_n = kdel2 / nv
    kdel2_n = np.cumprod(kdel2_n)

    #-----------------------------------------------------------------
    # R_transition : reflection coefficients based on the a transition function
    #   by T.D. Wu & A.K. Fung (2001)
    #-----------------------------------------------------------------
    #------transition when backscattering--------------
    Ftv = 8.0 * (rv0**2) * s**2 * (cs + np.sqrt(er - s**2)) / (cs * np.sqrt(er - s**2))
    Fth = -8.0 * (rh0**2) * s**2 * (cs + np.sqrt(er - s**2)) / (cs * np.sqrt(er - s**2))
    Spv0 =1.0/((np.abs(1.0 +8.0 *rv0/(cs*Ftv)))**2.0)
    Sph0 =1.0/((np.abs(1.0 +8.0 *rv0/(cs*Fth)))**2.0)
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    temp1 = 1.0
    Wnt, _ = roughness_spectrum(spectrum, Nmax, delta, corr_len, np.sqrt((2*k1*s)**2 + 0**2), x)
    for n in range(Nmax):
        fn = n+1
        temp1= temp1*(1/fn)
        sum1= sum1 + temp1*((kdel*cs)**(2.0 *fn))*Wnt[n]
        sum2= sum2 + temp1*((kdel*cs)**(2.0 *fn))*(np.abs(Ftv+2.0 **(fn+2.0 )*rv0/cs/(np.exp((kdel*cs)**2.0 )))**2.0 )*Wnt[n]
        sum3= sum3 + temp1*((kdel*cs)**(2.0 *fn))*(np.abs(Fth+2.0 **(fn+2.0 )*rv0/cs*(np.exp(-(kdel*cs)**2.0 )))**2.0 )*Wnt[n]

    sigma_vv_c = (np.abs(Ftv)**2.0 )*sum1
    sigma_vv_total = sum2
    sigma_hh_c = (np.abs(Fth)**2.0 )*sum1
    sigma_hh_total = sum3

    # calc rvtran & rhtran
    Spv = sigma_vv_c / sigma_vv_total
    Sph = sigma_hh_c / sigma_hh_total
    Gamv = 1.0 - Spv / Spv0
    Gamh = 1.0 - Sph / Sph0

    if Gamv < 0.0:
        Gamv = 0.0
    
    if Gamh < 0.0:
        Gamh = 0.0
    
    rvtran = rvi + (rvsp - rvi) * Gamv
    rhtran = rhi + (rhsp - rhi) * Gamh
    rvhtran = (rvtran - rhtran) / 2.0

    # update the Kirchhoff coefficient by transition reflection function
    rv = rvtran
    rh = rhtran
    rvh = rvhtran

    # fvv = -((1-rv) * hsnv + eta_ist * (1 + rv) * vsnh) \
    #     - (rh + rv) * vt * (hsnt + eta_ist * vsnd)
    # fhh = ((1 + rh) * vsnh + eta_ist * (1 - rh) * hsnv) \
    #     - (rh + rv) * hd * (vsnd + eta_ist * hsnt)
    # fvh = (-(1 + rh) * hsnh + eta_ist * (1 - rh) * vsnv) \
    #     + (rh + rv) * hd * (hsnd - eta_ist * vsnt)
    # fhv = ((1 - rv) * vsnv - eta_ist * (1 + rv) * hsnh) \
    #     + (rh + rv) * vt * (vsnt - eta_ist * hsnd)
    # 不要(rv + rh)项
    fvv = -((1-rv) * hsnv + eta_ist * (1 + rv) * vsnh)
    fhh = ((1 + rh) * vsnh + eta_ist * (1 - rh) * hsnv)
    fvh = (-(1 + rh) * hsnh + eta_ist * (1 - rh) * vsnv)
    fhv = ((1 - rv) * vsnv - eta_ist * (1 + rv) * hsnh)

    # 计算单次散射<Sqprs*>
    fqp_ = [fvv, fvh, fhv, fhh]

    # 补偿场系数要一直使用入射角
    rv = rvi
    rh = rhi
    rvh = rvhi
    u = -kx; v = -ky
    Fqp1up_ = [Fvv1(u, v, 1, rv, vpara), Fvh1(u, v, 1, rvh, vpara), Fhv1(u, v, 1, rvh, vpara), Fhh1(u, v, 1, rh, vpara)]
    Fqp1down_ = [Fvv1(u, v, -1, rv, vpara), Fvh1(u, v, -1, rvh, vpara), Fhv1(u, v, -1, rvh, vpara), Fhh1(u, v, -1, rh, vpara)]
    Fqp2up_ = [Fvv2(u, v, 1, rv, vpara), Fvh2(u, v, 1, rvh, vpara), Fhv2(u, v, 1, rvh, vpara), Fhh2(u, v, 1, rh, vpara)]
    Fqp2down_ = [Fvv2(u, v, -1, rv, vpara), Fvh2(u, v, -1, rvh, vpara), Fhv2(u, v, -1, rvh, vpara), Fhh2(u, v, -1, rh, vpara)]
    u = -ksx; v = -ksy
    Fqp1ups_ = [Fvv1(u, v, 1, rv, vpara), Fvh1(u, v, 1, rvh, vpara), Fhv1(u, v, 1, rvh, vpara), Fhh1(u, v, 1, rh, vpara)]
    Fqp1downs_ = [Fvv1(u, v, -1, rv, vpara), Fvh1(u, v, -1, rvh, vpara), Fhv1(u, v, -1, rvh, vpara), Fhh1(u, v, -1, rh, vpara)]
    Fqp2ups_ = [Fvv2(u, v, 1, rv, vpara), Fvh2(u, v, 1, rvh, vpara), Fhv2(u, v, 1, rvh, vpara), Fhh2(u, v, 1, rh, vpara)]
    Fqp2downs_ = [Fvv2(u, v, -1, rv, vpara), Fvh2(u, v, -1, rvh, vpara), Fhv2(u, v, -1, rvh, vpara), Fhh2(u, v, -1, rh, vpara)]

    k12 = [k1, k2]
    d12 = [+1, -1]

    Iqpnall = np.zeros((Nmax, 4), dtype=complex)

    for i in range(4):
        fqp = fqp_[i]
        Fqpall = [[Fqp1up_[i], Fqp1down_[i]],  [Fqp2up_[i], Fqp2down_[i]]]
        Fqpsall = [[Fqp1ups_[i], Fqp1downs_[i]],  [Fqp2ups_[i], Fqp2downs_[i]]]
        Iqpn = (nksz + nkz)**nv * fqp * np.exp(-delta**2 * np.real(ksz) * np.real(k))
        # (ii: medium 1/2, jj: up or downward +-)
        for ii in range(2):
            q = np.sqrt(k12[ii]**2 - kx**2 - ky**2)
            nq = q / k; # normalized
            qs = np.sqrt(k12[ii]**2 - ksx**2 - ksy**2)
            nqs = qs / k; # normalized
            for jj in range(2):
                d = d12[jj]
                # (u = -kx, v = -ky) + (u = -ksx, v = -ksy)
                Iqpn = Iqpn + 0.25 * (nksz - d * nq)**nv * Fqpall[ii][jj]  * np.exp(-delta**2 * (q**2 - d * q * ksz + d * q * kz)) \
                    + 0.25 * (nkz + d * nqs)**nv * Fqpsall[ii][jj] * np.exp(-delta**2 * (qs**2 - d * qs * ksz + d * qs * kz))
            
        Iqpnall[:, i] = Iqpn
    
    Sqprs = np.zeros((4, 4), dtype=complex) # single-scattering term
    for i in range(4): # i for qp
        Iqpn = Iqpnall[:, i]
        for j in range(4):
            Irsn = Iqpnall[:, j]
            Iqprsn = Iqpn * np.conj(Irsn)
            Sqprs[i, j] = np.sum(Iqprsn * kdel2_n * Wn)
        
    C0 = np.abs(k2)**2 / (8 * np.pi) * np.exp(-delta**2 * (np.real(ksz)**2 + np.real(kz)**2)) \
        * np.real(1/np.conj(eta2)) / np.real(1/np.conj(eta1))
    Ms = qprs2Mueller(C0 * Sqprs)

    # IEM坐标系转换到FSA坐标系
    U = np.diag([1, 1, -1, -1])
    Mue = U @ Ms

    # shadowing
    if shadow_flag is True:
        shadow = R1_shadowing(theta_i, rss)
        Mue = Mue * shadow

    # transmitted shadowing
    theta_t = np.pi - theta_s
    if np.rad2deg(theta_t) > 55:
        shadow = np.exp(-0.5 * np.tan(theta_t))
        Mue = Mue * shadow

    return Mue


#===============================================
# Complementary field coefficient function for AIEM
# F(+-)vv1(u,v): 媒质1中向上/向下（d=+1,-1）传播的补偿场系数
def Fvv1(u, v, d, rv, vpara):
    # 缩放后的补偿场系数（*(ksz-dqi)*(kz+dqi)
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rvp = (1 + rv)
    rvm = (1 - rv)
    qi = np.sqrt(k1**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = -(rvm / qi) * rvp * C[0] + rvm / qi * rvm * C[1] + rvm / qi * rvp * C[2] \
        + eta_ist * (rvp / qi * rvm * C[3] + rvp / qi * rvp * C[4] + rvp / qi * rvm * C[5])

    return F


# F(+-)vv2(u,v): 媒质2中向上/向下（d=+1,-1）传播的补偿场系数
def Fvv2(u, v, d, rv, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rvp = (1 + rv)
    rvm = (1 - rv)
    qi = np.sqrt(k2**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = (rvp * mur / qi) * rvp * C[0] - rvp / qi * rvm * C[1] - rvp / qi / er * rvp * C[2] \
        + eta_ist * (-rvm *er / qi * rvm * C[3] - rvm / qi * rvp * C[4] - rvm / qi /mur * rvm * C[5])
    
    return F


# F(+-)hh1(u,v)
def Fhh1(u, v, d, rh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rhp = (1 + rh)
    rhm = (1 - rh)
    qi = np.sqrt(k1**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = -rhp / qi * rhm * C[3] - rhp / qi * rhp * C[4] - rhp / qi * rhm * C[5] \
        + eta_ist * (rhm / qi * rhp * C[0] - rhm / qi * rhm * C[1] - rhm / qi* rhp * C[2])

    return F


# F(+-)hh2(u,v)
def Fhh2(u, v, d, rh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rhp = (1 + rh)
    rhm = (1 - rh)
    qi = np.sqrt(k2**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    C = Ccoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = rhm * mur / qi * rhm * C[3] + rhm / qi * rhp * C[4] + rhm / qi /er * rhm * C[5] \
        + eta_ist * (-rhp * er / qi * rhp * C[0] + rhp / qi * rhm * C[1] + rhp / qi / mur * rhp * C[2])

    return F


# F(+-)vh1(u,v)
def Fvh1(u, v, d, rvh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rp = 1 + rvh
    rm = 1 - rvh
    qi = np.sqrt(k1**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = rm / qi * rp * B[3] + rm / qi * rm * B[4] + rm / qi * rp * B[5] \
        + eta_ist * (rp / qi * rm * B[0] - rp / qi * rp * B[1] - rp / qi * rm * B[2])

    return F


# F(+-)vh2(u,v)
def Fvh2(u, v, d, rvh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rp = 1 + rvh
    rm = 1 - rvh
    qi = np.sqrt(k2**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = -rp * mur / qi * rp * B[3] - rp / qi * rm * B[4] - rp / qi / er * rp * B[5] \
        + eta_ist * (-rm * er / qi * rm * B[0] + rm / qi * rp * B[1] + rm / qi / mur * rm * B[2]);

    return F


# F(+-)hv1(u,v)
def Fhv1(u, v, d, rvh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rp = 1 + rvh
    rm = 1 - rvh
    qi = np.sqrt(k1**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = rm / qi * rp * B[0] - rm / qi * rm * B[1] - rm / qi * rp * B[2] \
        + eta_ist * (rp / qi * rm * B[3] + rp / qi * rp * B[4] + rp / qi * rm * B[5])

    return F


# F(+-)hv2(u,v)
def Fhv2(u, v, d, rvh, vpara):
    k1, k2, kx, ky, kz, ksx, ksy, ksz, v_, h_, vs_, hs_, mur, er, eta_ist  = vpara

    rp = 1 + rvh
    rm = 1 - rvh
    qi = np.sqrt(k2**2 - u**2 - v**2)
    n1_, n2_, g_ = calc_n1_n2_g(kx, ky, kz, ksx, ksy, ksz, u, v, d, qi)

    B = Bcoeff(k1, vs_, hs_, n1_, n2_, v_, h_, g_)

    F = -rp * mur / qi * rp * B[0] + rp / qi * rm * B[1] + rp / qi / er * rp * B[2] \
        + eta_ist * (-rm * er / qi * rm * B[3] - rm / qi * rp * B[4] - rm / qi / mur * rm * B[5])

    return F