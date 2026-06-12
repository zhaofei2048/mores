"""One layer of medium above substrate model for Lunar surface
Author: Fei Zhao
Create: 2024-04-28
"""

import numpy as np
import scipy.constants as sci_const
from ..core.VRT import VRT
from ..utils.app_utils import do_import_class


class LunarOneLayer(VRT):
    """
    Lunar One Layer scattering model
    Compared to "OneLayer" model, the differences are:
        1. The subsurface-volume interaction terms are not included.
        2. The subsurface is modeled by Hagfors model
    """
    def __init__(self, f=2.38e9, epsr0=1.0, epsr1=3.0, epsr2=6.0, thickness=1.0, 
                 surface_para={}, C_H=100, inclusion_para={}, surface_rocks_para={}):
        """
        Init the one layer model.

        Args:
            *f: frequency (Hz) of the radar
            *epsr0: relative dielectric constant of the upper medium
            *epsr1: background relative dielectric constant of the layer medium
            *epsr2: relative dielectric constant of the lower medium
            *thickness: thickness of the layer (m)
            *surface_para: A parameters dict (please reference to the specified surface scattering model) to characterize the upper interface, the 'model' parameter can be used to specify the surface model (default is 'RoughSurface'); and the epsilon_r parameter is not needed; and the frequency-dependent parameters kdel, kcor should be replaced with frequency-independent paramters delta, corr_len.
            *C_H: Hargfors roughness parameter, C_H >> 6.25 should be satisfied for the GO limit. 1/sqrt(C_H) can be regarded as the effective rms slope; Default is 100.
            *inclusion_para: A parameters dict (please reference to the specified layer scatterer model) to chacterize the inclusions, the 'model' parameter can be used to specify the layer model (default is 'RayleighSpherekska'); and the parameters f, thickness, epsr_background are not needed. Other parameters should be consistent with the selected layer model.
            *surface_rocks_para: a dict {'area_psd': psd, 'Dmax': xxx} needed by layer.SurfaceMieRocks
        Returns:

        """
        default_surface_para = {'model': 'RoughSurface'}
        default_subsurface_para = {'model': 'HagforsSurface', 'C': C_H}
        default_inclusion_para = {'model': 'RayleighSpherekska'}
        default_surface_rocks_para = {'model': 'SurfaceMieRocks'}

        default_surface_para.update(surface_para)
        default_inclusion_para.update(inclusion_para)
        if len(surface_rocks_para) > 0:
            account_for_surface_rocks = True
            default_surface_rocks_para.update(surface_rocks_para)
        else:
            account_for_surface_rocks = False

        Lambda0 = sci_const.speed_of_light / f
        k0 = 2.0 * np.pi / Lambda0
        # surface
        surface_model = default_surface_para['model']
        if 'delta' in default_surface_para.keys(): # Stationary rough surface model
            kdel1 = k0 * default_surface_para['delta']
            kcor1 = k0 * default_surface_para['corr_len']
            default_surface_para['kdel'] = kdel1
            default_surface_para['kcor'] = kcor1
            del default_surface_para['delta']
            del default_surface_para['corr_len']
        else:   # Fractal rough surface model
            default_surface_para['f'] = f
        
        del default_surface_para['model']
        self.surface_para = {'epsilon_r': epsr1/epsr0, **default_surface_para}
        # subsurface
        subsurface_model = default_subsurface_para['model']
        del default_subsurface_para['model']
        # Note that the roughness parameters are amplified due to a higher dielectric constant (>1.0) of upper medium.
        self.subsurface_para = {'epsilon_r': epsr2/epsr1, **default_subsurface_para}
        # inclusion
        layer_model = default_inclusion_para['model']
        del default_inclusion_para['model']
        self.inclusion_para = {'f': f, 'thickness': thickness, 'epsr_background': epsr1, **default_inclusion_para}
        # surface rocks
        if account_for_surface_rocks == True:
            surface_rocks_model = default_surface_rocks_para['model']
            del default_surface_rocks_para['model']
            self.surface_rocks_para = {'f': f, 'epsr_particle': default_inclusion_para['epsr_particle'],
                                       **default_surface_rocks_para}

        # implementation
        interface_cls = do_import_class('mores.interface.' + surface_model)
        surface = interface_cls(**self.surface_para)
        interface_cls = do_import_class('mores.interface.' + subsurface_model)
        subsurface = interface_cls(**self.subsurface_para)
        layer_cls = do_import_class('mores.layer.' + layer_model)
        layer = layer_cls(**self.inclusion_para)
        if account_for_surface_rocks == True:
            surface_rocks_cls = do_import_class('mores.layer.' + surface_rocks_model)
            self.surface_rocks = surface_rocks_cls(**self.surface_rocks_para)
        else:
            self.surface_rocks = None

        super(LunarOneLayer, self).__init__(surface=surface, layer=layer, subsurface=subsurface)
    

    def Mue_surface_rock_odd(self, geom):
        """Mueller matrix for total scattering components in forward scattering alignment (FSA) convention.

        The subsurface-volume interaction terms are not included in this model for the coherent reflection of the sub-HagforsSurface is zero.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_i is in surface scattering coordinate, i.e., the angle between -ki and z

        Returns:
            Mue_sur_rock_odd: 4x4 real Mueller matrix
        """
        if self.surface_rocks is None:
            print('No surface rocks model can be used.')
            return None
        else:
            theta_s, phi_s, theta_i, phi_i = geom
            return self.surface_rocks.phase_matrix(geom=(theta_s, phi_s, 180-theta_i, phi_i))


    def Mue_surface_rock_double(self, geom):
        """Mueller matrix for total scattering components in forward scattering alignment (FSA) convention.

        Only backscattering is accounted for. 二面角散射, 石块和表面形成二面角

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_i is in surface scattering coordinate, i.e., the angle between -ki and z

        Returns:
            Mue_sur_rock_doubble: 4x4 real Mueller matrix
        """
        if self.surface_rocks is None:
            print('No surface rocks model can be used.')
            return None
        else:
            theta_s, phi_s, theta_i, phi_i = geom
            angle_s = 180 - 2 * theta_i
            P = self.surface_rocks.phase_matrix(geom=(angle_s, 0, 0, 0))
            Rc = self.surface.M_coh_R(theta_i) 
            # BSA to FSA Rc
            Q_BSA2FSA = np.diag([1, 1, -1, -1]) # 对吗？
            Rc = Q_BSA2FSA @ Rc # 对吗？

            Mue = 2 * (Rc@P + P@Rc) # 2 for coherent enhancement, path1 = path2

            return Mue


    def Mue_total(self, geom):
        """Mueller matrix for total scattering components in forward scattering alignment (FSA) convention.

        The subsurface-volume interaction terms are not included in this model for the coherent reflection of the sub-HagforsSurface is zero.

        Args:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_i is in surface scattering coordinate, i.e., the angle between -ki and z

        Returns:
            Mue_total: 4x4 real Mueller matrix
        """
        Mue_sur = self.Mue_surface(geom)
        Mue_vol = self.Mue_volume(geom)
        Mue_sub = self.Mue_subsurface(geom)
        Mue_total = Mue_sur + Mue_vol + Mue_sub

        return Mue_total, Mue_sur, Mue_vol, Mue_sub
    
