"""One layer of medium above substrate model
Author: Fei Zhao
Create: 2024-01-19
"""

import numpy as np
import scipy.constants as sci_const
from ..core.VRT import VRT
from ..utils.app_utils import do_import_class


class OneLayer(VRT):
    """
    One Layer scattering model
    """
    def __init__(self, f=2.38e9, epsr0=1.0, epsr1=3.0, epsr2=6.0, thickness=1.0, 
                 surface_para={}, subsurface_para={}, inclusion_para={}):
        """
        Init the one layer model.

        Args:
            f: frequency (Hz) of the radar
            epsr0: relative dielectric constant of the upper medium
            epsr1: background relative dielectric constant of the layer medium
            epsr2: relative dielectric constant of the lower medium
            thickness: thickness of the layer (m)
            surface_para: A parameters dict (please reference to the specified surface scattering model) to characterize the upper interface, the 'model' parameter can be used to specify the surface model (default is 'RoughSurface'); and the epsilon_r parameter is not needed; and the frequency-dependent parameters kdel, kcor should be replaced with frequency-independent paramters delta, corr_len.
            subsurface_para: A parameters dict {similar to surface} to characterize the lower interface.
                **Note that the subsurface roughness paramters delta, corr_len are implicityly assumed in free space. However, the upper medium above subsurface gernerally has dielectric constant > 1, and hence shorter wavelength. Therefore, same roughness parameters will be generally rougher for subsurface than for surface. The roughness parameters will be amplified by a factor of np.real(np.sqrt(epsr1))
            inclusion_para: A parameters dict (please reference to the specified layer scatterer model) to chacterize the inclusions, the 'model' parameter can be used to specify the layer model (default is 'RayleighSpherekska'); and the parameters f, thickness, epsr_background are not needed. Other parameters should be consistent with selected layer model.
        Returns:

        """
        # default
        # default_surface_para = {'delta': 0.1e-2, 'corr_len': 1.0e-2, 'corr_fun': 'exp', 'model': 'RoughSurface'}
        # default_subsurface_para = {'delta': 0.1e-2, 'corr_len': 1.0e-2, 'corr_fun': 'exp', 'model': 'RoughSurface'}
        # default_inclusion_para = {'epsr_particle': 10.0-0.5j, 'ks': 0.3, 'ka': 0.8, 'model': 'RayleighSpherekska'}
        default_surface_para = {'model': 'RoughSurface'}
        default_subsurface_para = {'model': 'RoughSurface'}
        default_inclusion_para = {'model': 'RayleighSpherekska'}

        default_surface_para.update(surface_para)
        default_subsurface_para.update(subsurface_para)
        default_inclusion_para.update(inclusion_para)

        # self.surface_para = {'epsilon_r': epsr1/epsr0, 'kdel': None, 'kcor': None, 'corr_fun': 'exp'}
        # self.subsurface_para = {'epsilon_r': epsr2/epsr1, 'kdel': None, 'kcor': None, 'corr_fun': 'exp'}
        # self.inclusion_para = {'f':f, thickness: thickness}

        Lambda0 = sci_const.speed_of_light / f
        k0 = 2.0 * np.pi / Lambda0
        # surface
        kdel1 = k0 * default_surface_para['delta']
        kcor1 = k0 * default_surface_para['corr_len']
        surface_model = default_surface_para['model']
        del default_surface_para['delta']
        del default_surface_para['corr_len']
        del default_surface_para['model']
        self.surface_para = {'epsilon_r': epsr1/epsr0, 'kdel': kdel1, 'kcor': kcor1, **default_surface_para}
        # subsurface
        kdel2 = k0 * default_subsurface_para['delta']
        kcor2 = k0 * default_subsurface_para['corr_len']
        subsurface_model = default_subsurface_para['model']
        del default_subsurface_para['delta']
        del default_subsurface_para['corr_len']
        del default_subsurface_para['model']
        # Note that the roughness parameters are amplified due to a higher dielectric constant (>1.0) of upper medium.
        self.subsurface_para = {'epsilon_r': epsr2/epsr1, 'kdel': kdel2*np.real(np.sqrt(epsr1)), 'kcor': kcor2*np.real(np.sqrt(epsr1)), **default_subsurface_para}
        # inclusion
        layer_model = default_inclusion_para['model']
        del default_inclusion_para['model']
        self.inclusion_para = {'f': f, 'thickness': thickness, 'epsr_background': epsr1, **default_inclusion_para}

        # implementation
        interface_cls = do_import_class('mores.interface.' + surface_model)
        surface = interface_cls(**self.surface_para)
        interface_cls = do_import_class('mores.interface.' + subsurface_model)
        subsurface = interface_cls(**self.subsurface_para)
        layer_cls = do_import_class('mores.layer.' + layer_model)
        layer = layer_cls(**self.inclusion_para)

        super(OneLayer, self).__init__(surface=surface, layer=layer, subsurface=subsurface)
    

    def Mue_total(self, geom):
        """Mueller matrix for total scattering components in forward scattering alignment (FSA) convention.

        The subsurface-volume interaction terms are incoherently added.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles

        Returns:
            Mue_total: 4x4 real Mueller matrix
        """
        Mue_sur = self.Mue_surface(geom)
        Mue_vol = self.Mue_volume(geom)
        Mue_sub = self.Mue_subsurface(geom)
        Mue_sub_vol = self.Mue_subsurface_volume(geom)
        Mue_total = Mue_sur + Mue_vol + Mue_sub + Mue_sub_vol

        return Mue_total
