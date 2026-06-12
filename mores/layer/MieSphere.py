"""Layer of Mie scattering spheres

Author: Fei Zhao
Create: 2024-04-30
"""
import numpy as np
import scipy
import scipy.constants as sci_const
# from scipy.integrate import trapz
from ..core.Layer import Layer
import miepython
from .miepython_utils import mie_cross_sections_psd, mie_cross_sections, mie_phase_matrix_elements, mie_phase_matrix_elements_psd, mie_phase_matrix_Csca
from ..utils.postprocessing import Mueller_matrix_L2M

class MieSphere(Layer):
    """
    Mie scattering spheres
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
        # initialization of the background medium and wavelength
        super(MieSphere, self).__init__(f=f, thickness=thickness, epsr_background=epsr_background)
        self.epsr_particle = epsr_particle              # particle原始的相对空气的介电常数
        # 注意这里背景媒质的介电常数虚部被忽略了, 防止出现颗粒相对介电常数虚部为正的情况
        self.epsr = epsr_particle / np.real(epsr_background)     # relative complex diel. of particles to that of background medium
        self.refractive_index = np.sqrt(self.epsr)               # relative complex index of refraction of the particle

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
            self.Dmin = None
            self.Dmax = particle_size[1] 
            self.size_psd = particle_size[2]
            # equivalent total number concentration of particles per unit volume
            # self.n0, err = scipy.integrate.quad(self.size_psd, D_lower_limit, self.Dmax) # using 0.0001 as lower limit to avoid division by 0 (Indujaa)
            # Assume that the D is the diameter of the equivalent-volume sphere
            # fun_vol = lambda D: 4/3*np.pi * (D*0.5)**3 * self.size_psd(D)
            # self.fs, err = scipy.integrate.quad(fun_vol, D_lower_limit, self.Dmax)
            # 对于指数分布的石块Fk(D), 积分得到体积占比不会收敛
            self.n0 = 0
            self.fs = 0
        else:
            raise ValueError("Parameter particle_size should be a tuple with 2 or 3 elements!")
        
        # 强制设置Dmin和Dmax，便于测试
        # self.Dmin = 0.01 * self.Lambda    # 1% of wavelength
        if self.Dmax > 10 * self.Lambda:
            self.Dmax = 10 * self.Lambda    # 便于加速计算

        self.Kab = -2 * self.k0 * np.sqrt(self.epsr_background).imag * (1 - self.fs)
        self.__init_ks_ka_ke_backPha()


    def __init_ks_ka_ke_backPha(self):
        """Init the ks, ka, ke and the phase matrix in backscattering direction"""
        if self.is_multi_sizes == False:
            # geometric_cross_section = np.pi * self.radius**2
            # x = 2*np.pi*self.radius / self.Lambda
            # qext, qsca, qback, g = miepython.mie(m=self.refractive_index, x=x)
            Cext, Csca, Cback = mie_cross_sections(m=self.refractive_index, wavelength=self.Lambda, r=self.radius)
            self.ke = Cext * self.n0
            self.ks = Csca * self.n0
        else:
            Cext, Csca, Cback = mie_cross_sections_psd(m=self.refractive_index, 
                                                       wavelength=self.Lambda, 
                                                       psd=self.size_psd, 
                                                       Dmin=self.Dmin, 
                                                       Dmax=self.Dmax)
            self.ke = Cext
            self.ks = Csca

        self.ka = self.ke - self.ks
        self.ssa = self.ks / (self.ke + self.Kab)     
        
        Z = np.zeros((4, 4))
        Z[0, 0] = Cback / (4 * np.pi)
        Z[1, 1] = Cback / (4 * np.pi)
        Z[2, 2] = -Cback / (4 * np.pi)
        Z[3, 3] = -Cback / (4 * np.pi)

        self.back_phase_matrix = Mueller_matrix_L2M(Z)
  
  
    def phase_matrix(self, geom=None):
        """
        Phase matrix used in VRT equation: P(theta_s, phi_s; theta_i, phi_i)
        INPUT:
            geom (tuple): (not needed) observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles.
                            Note that theta_s and theta_i belong to [0, 180] defined in volume scattering coordinate.
                            theta_s is the angle between z and ks, while theta_i is the angle between z and ki.
        OUTPUT:
            P: 4x4 phase matrix, note this phase matrix has already been multiplied by n0
        """
        ths, phis, thi, phi = np.deg2rad(geom)
        # the directional norm vector
        ki_ = np.array([np.sin(thi)*np.cos(phi), np.sin(thi)*np.sin(phi), np.cos(thi)])
        ks_ = np.array([np.sin(ths)*np.cos(phis), np.sin(ths)*np.sin(phis), np.cos(ths)])
        mu = np.sum(ki_ * ks_)
        if mu == -1.0:  # backscattering
            P = self.back_phase_matrix
        else:
            if self.is_multi_sizes == False:
                Z = mie_phase_matrix_Csca(m=self.refractive_index, wavelength=self.Lambda, r=self.radius, mu=mu)
                Z = Z * self.n0
            else:
                Z = mie_phase_matrix_elements_psd(m=self.refractive_index, 
                                                wavelength=self.Lambda, 
                                                psd=self.size_psd, 
                                                Dmin=self.Dmin, 
                                                Dmax=self.Dmax, 
                                                mu=mu)
            P = Mueller_matrix_L2M(Z)
            
        return P
    

    def extinction_matrix(self, direction=None):
        """
        Compute extinction matrix Ke.
        *Note that extinction matrix is only related to the incident direction, hence for spherical particles, 
            it is the same for both polarizations and is given by the diagonal form.
        INPUT:
            direction (tuple): (not needed) direction of propagation (theta, phi) in degree
                            Note that theta belongs to [0, 180] defined in volume scattering coordinate.
                            theta is the angle between z and k.
        OUTPUT:
            Ke: 4x4 extinction matrix of the discrete scatterers
        """
        return np.diag([self.ke]*4)