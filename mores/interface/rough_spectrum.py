"""Roughness spectrum of surface

Author: Fei Zhao
Create: 2024-01-16
"""
import numpy as np
from scipy.special import gamma, kv as besselk, jn as besselj
from scipy.integrate import quad

def roughness_spectrum(corr_type, N, delta, L, wvnb, x):
    """Generate N-order spectrum of rough surface
    
    Args:
        corr_type: the spectrum of the rough surface, i.e., the Fourier transform of corr function,
            can be one of {'Gauss', 'exp', 'x-power', 'x-exp'}.
        N: the order of the spectrum
        delta: RMS height (m)
        L: correlation length of rough surface (m)
        x: coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correl. fnc.
    
    Returns:
        Wn: [N*1] array
        rss: root mean square slope (rad)
    """
    corr_type = corr_type.lower()
    Norder = np.arange(1, N+1, 1)
    if corr_type == 'gauss':
        Wn = L**2 / (2 * Norder) * np.exp(-(wvnb * L)**2 / (4 * Norder))
        rss = np.sqrt(2) * delta / L
    elif corr_type == 'exp':
        Wn = L**2 / Norder**2 * (1 + (wvnb * L)**2 / Norder**2)**(-1.5)
        rss = delta / L
    elif corr_type == 'x-power':
        if wvnb == 0:
            Wn = L**2 / (3 * Norder - 2)
        else:
            Wn = L**2 * (wvnb * L)**(-1 + x * Norder) * besselk(1-x*Norder, wvnb*L) \
                / (2**(x*Norder - 1) * gamma(x*Norder))

        rss = np.sqrt(x*2) * delta / L
    elif corr_type == 'x-exp':
        Wn = np.zeros(N)
        for n in range(1, N+1):
            tmp = quad(lambda z: x_exponential_spectrum(z, wvnb, L, n, x), 0, 9)
            Wn[n-1] = L**2 / n**(2/x) * tmp
        
        rss = np.sqrt(4) * delta / L
    elif corr_type == 'gauss-1d':
        Wn = np.sqrt(np.pi / Norder) * L* np.exp(-(wvnb*L)**2 / (4 * Norder))
        rss = np.sqrt(2) * delta / L
    else:
        raise ValueError(f"No matching surface spectrum for: , {corr_type}, use one of ('gauss', 'exp', 'x-power', 'x-exp', ...) instead")

    return Wn, rss


def x_exponential_spectrum(z, wvnb, L, n, xx):
    tmp = np.exp(-np.abs(z)**xx) * besselj(0, z*wvnb*L/(n**(1/xx))) * z