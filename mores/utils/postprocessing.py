"""
Author: Fei Zhao
Create: 2023-08-24

Description:
    Functions for postprocessing of MORES model
"""

import numpy as np

def Mueller2sigma0(Mue, pol_r, pol_t):
    """
    Convert Mueller matrix to scattering coefficient sigma (Polarization synthesis).
    INPUT:
        Mue: the Mueller matrix in FSA descrbing the scattering characteristics of surface
        pol_r: (Psi_r, Chi_r) are the polarization ellipse parameters of the receiving antenna.
                Note Psi and Chi are orientation and ellipticity angles respectively.
        pol_t: (Psi_t, Chi_t) same as above but for transmitting antenna
    OUTPUT:
        sigma0: the scattering coefficient under specified polarization state
    Note:
        (Psi, Chi) = Polarization state, Psi in [-90, 90], Chi in [-45, 45]
        (0, 0)      = V polarization
        (0, 90)     = H polarzation
        (0, 45)     = LHC (left-circular polarization)
        (0, -45)    = RHC (right-circular polarization)
        (45, 0)     = 45° linear polarization
        (-45, 0)    = -45° linar polarization
    """
    Psi_r, Chi_r = np.deg2rad(pol_r)
    Psi_t, Chi_t = np.deg2rad(pol_t)

    # normalized stokes vectors of transmitting and receiving antennas
    Int = np.array([0.5*(1+np.cos(2*Psi_t)*np.cos(2*Chi_t)), 0.5*(1-np.cos(2*Psi_t)*np.cos(2*Chi_t)), np.sin(2*Psi_t)*np.cos(2*Chi_t), -np.sin(2*Chi_t)])
    # Inr = np.array([0.5*(1+np.cos(2*Psi_r)*np.cos(2*Chi_r)), 0.5*(1-np.cos(2*Psi_r)*np.cos(2*Chi_r)), -np.sin(2*Psi_r)*np.cos(2*Chi_r), np.sin(2*Chi_r)]) # the minus on 3-th and 4-th elements of Inr is due to the inversed propagation
    Inr = np.array([0.5*(1+np.cos(2*Psi_r)*np.cos(2*Chi_r)), 0.5*(1-np.cos(2*Psi_r)*np.cos(2*Chi_r)), np.sin(2*Psi_r)*np.cos(2*Chi_r), -np.sin(2*Chi_r)]) # Ulaby 2014, p208, 这里还值得商榷

    Q = np.diag([1, 1, 0.5, -0.5])
    Q_BSAFSA = np.diag([1, 1, -1, -1])
    # Ulaby, eq. 5.148, p208
    sigma0 = 4 * np.pi * (np.dot(Inr, Q @ Q_BSAFSA @ Mue @ Int))

    return sigma0


def Mueller2sigma0VH(Mue):
    """
    Convert Mueller matrix to linearly polarized scattering coefficient: VV, VH, HV, HH.
    INPUT:
        Mue: the Mueller matrix in FSA descrbing the scattering characteristics of surface
    OUTPUT:
        sigma0_VV
        sigma0_VH
        sigma0_HV
        sigma0_HH
    Note:
        sigma_qp, q is the polarization of receiving and p is that of transmitting
    """
    Vpol = (0, 0)
    Hpol = (0, 90)
    sigma0_VV = Mueller2sigma0(Mue, Vpol, Vpol)
    sigma0_VH = Mueller2sigma0(Mue, Vpol, Hpol)
    sigma0_HV = Mueller2sigma0(Mue, Hpol, Vpol)
    sigma0_HH = Mueller2sigma0(Mue, Hpol, Hpol)

    return sigma0_VV, sigma0_VH, sigma0_HV, sigma0_HH


def Mueller2sigma0LR(Mue):
    """
    Convert Mueller matrix to circularly polarized scattering coefficient: LL, LR, RL, RR.
    INPUT:
        Mue: the Mueller matrix in FSA descrbing the scattering characteristics of surface
    OUTPUT:
        sigma0_LL
        sigma0_LR
        sigma0_RL
        sigma0_RR
    Note:
        sigma_qp, q is the polarization of receiving and p is that of transmitting
    """
    Lpol = (0, 45)
    Rpol = (0, -45)
    sigma0_LL = Mueller2sigma0(Mue, Lpol, Lpol)
    sigma0_LR = Mueller2sigma0(Mue, Lpol, Rpol)
    sigma0_RL = Mueller2sigma0(Mue, Rpol, Lpol)
    sigma0_RR = Mueller2sigma0(Mue, Rpol, Rpol)

    return sigma0_LL, sigma0_LR, sigma0_RL, sigma0_RR


def Mueller2sigma0CHV(Mue, C='L'):
    """Convert Mueller matrix to scattering coefficient with Circularly polarized transmitting and H, V receiving

    Args:
        Mue: the Mueller matrix in FSA descrbing the scattering characteristics of surface
        C: 'L' (default) for left hand circular polarization transmitting and 'R' for right hand circular polarization transmitting.

    Retruns:
        sigma0_VC
        sigma0_HC
        
    Note:
        sigma_qp, q is the polarization of receiving and p is that of transmitting
    """
    Lpol = (0, 45)
    Rpol = (0, -45)
    Vpol = (0, 0)
    Hpol = (0, 90)
    if C=='L':
        Tpol = Lpol
    elif C=='R':
        Tpol = Rpol
    else:
        raise ValueError("Unknown polarization of transmitting, only L(eft) and R(ight) hand circular polarzations are allowed.") 

    sigma0_VC = Mueller2sigma0(Mue, Vpol, Tpol)
    sigma0_HC = Mueller2sigma0(Mue, Hpol, Tpol)

    return sigma0_VC, sigma0_HC
        
    
def scattering_amplitudes_to_Mueller(S):
    """Convert scattering amplitudes matrix to Mueller matrix
    
    Args:
        S: 2x2 scattering amplitudes matrix
    
    Returns:
        Mue: 4x4 Mueller matrix
    """
    Sxx = S[0, 0]
    Sxy = S[0, 1]
    Syx = S[1, 0]
    Syy = S[1, 1]
    Mue = np.array([[np.abs(Sxx)**2, np.abs(Sxy)**2, np.real(Sxx*np.conj(Sxy)), -np.imag(Sxx*np.conj(Sxy))],
                    [np.abs(Syx)**2, np.abs(Syy)**2, np.real(Syx*np.conj(Syy)), -np.imag(Syx*np.conj(Syy))],
                    [2*np.real(Sxx*np.conj(Syx)), 2*np.real(Syx*np.conj(Syy)), np.real(Sxx*np.conj(Syy)+Sxy*np.conj(Syx)), -np.imag(Sxx*np.conj(Syy)-Sxy*np.conj(Syx))],
                    [2*np.imag(Sxx*np.conj(Syx)), 2*np.imag(Syx*np.conj(Syy)), np.imag(Sxx*np.conj(Syy)+Sxy*np.conj(Syx)), np.real(Sxx*np.conj(Syy)-Sxy*np.conj(Syx))]])
    return Mue


def Mueller_matrix_L2M(L):
    """
    Convert Mueller/phase matrix L (for stoeks g) to matrix M (for stokes I)

    Args:
        L: g_s = L@g_i, where g = [|Ex|^2+|Ey|^2, |Ex|^2-|Ey|^2, 2*Re(Ex*Ey_star), 2*Im(Ex*Ey_star)]
    
    Returns:
        M: I_s = M@I_i, where I = [|Ex|^2, |Ey|^2, 2*Re(Ex*Ey_star), 2*Im(Ex*Ey_star)]
    """
    Q = np.array([[1., 1, 0, 0],
                    [1, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]])
    invQ = np.array([[0.5,  0.5,  0.,  0.],
                        [0.5, -0.5, -0., -0.],
                        [0.,  0.,  1.,  0.],
                        [-0., -0., -0., -1.]])
    M = invQ @ L @ Q

    return M