"""
Author: Fei Zhao
Create: 2023-09-01

Description:
    A layer consists of Rayleigh spherical scatterers, support multi-sizes distribution.
"""
import numpy as np
import scipy
import scipy.constants as sci_const
from scipy.integrate import trapz

from .RayleighSpherekska import RayleighSpherekska
from .particle_utils import scattering_cross_section_rayleigh_sphere, absorption_cross_section_rayleigh_sphere


class RayleighSphere(RayleighSpherekska):
    """
    Rayleigh sphere layers with multi-sizes, see Ulaby 2014, p474
    """
    def __init__(self, f, thickness=None, epsr_background=1.0, 
                 epsr_particle=1.0, particle_size=(1.0, 1.0), num_points=1024):
        """
        INPUT:
            f: frequency (Hz) of the incident waves
            thickness: the thickness (meters) of the layer, if set None, the penetration depth in the medium will be used
            epsr_background: relative complex dielectric constant of the background medium
            epsr_particle: relative complex dielectric constant of the particle
            particle_size: can be a 1)1x2 tuple (radius, fs) = radius of equivalent (volume) sphere, volume fraction of the particles
                              or 2)1x3 tuple (0, Dmax, size_distribution_func) = Dmax is the maximum diameter of scatterers, size_distribution_func is
                              the size distribution function (callable) of particles defined in (0, Dmax), the integration of size_distribution_func between 0 to
                              Dmax should equal to n0, i.e., the number concentration of particles in a unit volume. Note the unit should be meters.
                              e.g. size_distribution_func(D) -> N(D)
            num_points: num of discrete points to calculate the size averaged parameters
        """
        # particle_orientation = (0, 0)
        # particle_shape=(1, 'SPHEROID')
        # super(RayleighSphere, self).__init__(f, epsr_background, epsr_particle, particle_size, particle_orientation,
        #                                      particle_shape, thickness)
        self.epsr_particle = epsr_particle
        self.epsr = epsr_particle / epsr_background # relative complex diel. of particles to that of background medium
        self.refractive_index = np.sqrt(self.epsr)
        epsr = epsr_particle / epsr_background
        Lambda0 = sci_const.speed_of_light / f  # wavelength in the free space
        Lambda = Lambda0 / np.sqrt(np.real(epsr_background)) # wavelength in the background medium
        # k = 2 * np.pi / Lambda  

        # particle size
        if len(particle_size) == 2:
            self.is_multi_sizes = False
            self.radius = particle_size[0]
            self.fs = particle_size[1]
            # a = self.radius
            # b = a * self.axis_ratio
            vol = np.pi * 4 / 3 * (self.radius**3)  # volume of single particles
            self.n0 = self.fs / vol
        elif len(particle_size) == 3:
            self.is_multi_sizes = True
            self.Dmax = particle_size[1] 
            self.size_psd = particle_size[2]
            # equivalent total number concentration of particles per unit volume
            self.n0, err = scipy.integrate.quad(self.size_psd, 0.0001, self.Dmax) # using 0.0001 as lower limit to avoid division by 0 (Indujaa)
            # Assume that the D is the diameter of the equivalent-volume sphere
            fun_vol = lambda D: 4/3*np.pi * (D*0.5)**3 * self.size_psd(D)
            self.fs, err = scipy.integrate.quad(fun_vol, 0.0001, self.Dmax)
        else:
            raise ValueError("Parameter particle_size should be a tuple with 2 or 3 elements!")
        
        # volume scattering coefficient & volume absorption coefficient
        if self.is_multi_sizes is False:
            Chi = 2 * np.pi * self.radius / Lambda
            Qs = scattering_cross_section_rayleigh_sphere(Lambda, epsr, Chi)
            Qa = absorption_cross_section_rayleigh_sphere(Lambda, epsr, Chi)
            ks = self.n0 * Qs
            ka = self.n0 * Qa
        else:   # multi-sizes
            psd_D = np.linspace(self.Dmax/num_points, self.Dmax, num_points)
            psd_W = self.size_psd(psd_D)
            Chi = np.pi * psd_D / Lambda
            Qs = scattering_cross_section_rayleigh_sphere(Lambda, epsr, Chi)
            Qa = absorption_cross_section_rayleigh_sphere(Lambda, epsr, Chi)
            ks = trapz(Qs * psd_W, psd_D)
            ka = trapz(Qa * psd_W, psd_D)

        super(RayleighSphere, self).__init__(f=f, thickness=thickness, epsr_background=epsr_background, 
                                             ks=ks, ka=ka, vol_frac=self.fs)

