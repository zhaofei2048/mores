"""Dielectric constant models

Author: Fei Zhao
Create: 2024-01-14
"""

import numpy as np


def diel_soil(freq, mv, S=0.25, C=0.25, temp=25, rho_b=1.7):
        """
        Relative Dielectric Constant of soil.
        Computes the real and imaginary parts of the relative
        dielectric constant of soil at a given temperature 0<t<40C, frequency,
        volumetric moisture content, soil bulk density, sand and clay
        fractions.

        Args:
            frequency : int, float or array_like
                Frequency (GHz).
            temp : int, float or array
                Temperature in C° (0 - 30).
            S : int or float
                Sand fraction (0 < S < 1).
            C : int or float
                Clay fraction (0 < C < 1).
            mv : int or float
                Volumetric Water Content (0<mv<1)
            rho_b : int or float (default = 1.7)
                Bulk density in g/cm3 (typical value is 1.7 g/cm3).

        Returns
            eps: Dielectric Constant, complex
        
        Ref: Adapted from https://github.com/ibaris/pyrism
        """
        frequency = freq / 1.0e9

        f_hz = frequency * 1.0e9

        beta1 = 1.27 - 0.519 * S - 0.152 * C
        beta2 = 2.06 - 0.928 * S - 0.255 * C
        alpha = 0.65

        eps_0 = 8.854e-12

        sigma_s = 0
        if frequency > 1.3:
            sigma_s = -1.645 + 1.939 * rho_b - 2.256 * S + 1.594 * C

        if frequency >= 0.3 and frequency <= 1.3:
            sigma_s = 0.0467 + 0.22 * rho_b - 0.411 * S + 0.661 * C

        ew_inf = 4.9
        ew_0 = 88.045 - 0.4147 * temp + 6.295e-4 * temp ** 2 + 1.075e-5 * temp ** 3
        tau_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp ** 2 - 5.096e-16 * temp ** 3) / 2 / np.pi

        epsrW = ew_inf + (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2)

        epsiW = 2 * np.pi * tau_w * f_hz * (ew_0 - ew_inf) / (1 + (2 * np.pi * f_hz * tau_w) ** 2) + (
                2.65 - rho_b) / 2.65 / mv * sigma_s / (2 * np.pi * eps_0 * f_hz)

        epsr = (1 + 0.66 * rho_b + mv ** beta1 * epsrW ** alpha - mv) ** (1 / alpha)
        epsi = mv ** beta2 * epsiW

        eps = np.complex(epsr, -epsi)

        return eps