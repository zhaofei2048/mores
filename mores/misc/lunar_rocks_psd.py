"""Lunar rocks distribution function

Author: Fei Zhao
Create: 2024-04-30
"""
import numpy as np

# Default params
# Li and Wu, 2018, JGR-planets "Analysis of Rock Abundance on Lunar Surface From Orbital and Descent Images Using Automatic Rock Detection"
DEFAULT_A = 0.5648
DEFAULT_B = 0.01285
DEFAULT_LUNAR_SURFACE_POROSITY = 0.5    # Carrier, 1991, lunar source book


def lunar_psd_vol(D, k, A=DEFAULT_A, B=DEFAULT_B, porosity=DEFAULT_LUNAR_SURFACE_POROSITY):
    """
    Lunar volume particle size distribution

    Args:
        D:  at which diameter to calculate the psd value, can be array
        k: area fraction of all the rocks (D>=0)
        porosity: porosity of the lunar surface, default is 0.5 according to Carrier et al., 1991, lunar source book
        A: the exponential decay coefficient q=A + B/k, see M. Golombek and D. Rapp, 1997, JGR-plantes
        B: default is A = 0.5648, B = 0.01285 according to Li and Wu, 2018, JGR-planets "Analysis of Rock Abundance on Lunar Surface From Orbital and Descent Images Using Automatic Rock Detection"
    
    Returns:
        psd: Number concentration of rocks at diameter D per unit volume per diameter [#/m^3*m]
    """
    D = np.array(D)
    q = A + B / k
    c = 6 * q * k / (np.pi * (1 - porosity))
    psd = c * np.exp(-3*np.log(D)-q*D)
    # psd = c * np.exp(-q*D)

    if np.shape(D) == ():
        if D==0.0:
            return 0.0
    else:
        psd[D==0.0] = 0.0

    return psd


def lunar_psd_area(D, k, A=DEFAULT_A, B=DEFAULT_B):
    """
    Lunar area particle size distribution

    Args:
        D:  at which diameter to calculate the psd value, can be array
        k: area fraction of all the rocks (D>=0)
        A: the exponential decay coefficient q=A + B/k, see M. Golombek and D. Rapp, 1997, JGR-plantes
        B: default is A = 0.5648, B = 0.01285 according to Li and Wu, 2018, JGR-planets "Analysis of Rock Abundance on Lunar Surface From Orbital and Descent Images Using Automatic Rock Detection"
    
    Returns:
        psd: Number concentration of rocks at diameter D per unit volume per diameter [#/m^3*m]
    """
    D = np.array(D)
    q = A + B / k
    c = 4 * q * k / np.pi
    psd = c * np.exp(-2*np.log(D)-q*D)

    if np.shape(D) == ():
        if D==0.0:
            return 0.0
    else:
        psd[D==0.0] = 0.0

    return psd


class LunarRocksPSD:
    """Lunar rocks particle size distribution calculator"""
    def __init__(self, k, Dmax=None, A=DEFAULT_A, B=DEFAULT_B, porosity=DEFAULT_LUNAR_SURFACE_POROSITY):
        """
        Args:
            k: area fraction of all the rocks (D>=0)
            Dmax: the maximum diameter used for integration, default is 11/q
            A: the exponential decay coefficient q=A + B/k, see M. Golombek and D. Rapp, 1997, JGR-plantes
            B: default is A = 0.5648, B = 0.01285 according to Li and Wu, 2018, JGR-planets "Analysis of Rock Abundance on Lunar Surface From Orbital and Descent Images Using Automatic Rock Detection"

        Returns:
            LunarRocksPSD obj
        """
        self.k = k
        self.A = A
        self.B = B
        self.porosity = porosity

        q = A + B / k

        if Dmax is None:
            self.Dmax = 11/q
        else:
            self.Dmax = Dmax
    

    def lunar_psd_vol(self, D):
        """
        Lunar volume particle size distribution

        Args:
            D:  at which diameter to calculate the psd value, can be array
        """
        return lunar_psd_vol(D, k=self.k, A=self.A, B=self.B, porosity=self.porosity)
    

    def lunar_psd_area(self, D):
        """
        Lunar volume particle size distribution

        Args:
            D:  at which diameter to calculate the psd value, can be array
        """
        return lunar_psd_area(D, k=self.k, A=self.A, B=self.B)