"""
Author: Fei Zhao
Create: 2023-08-24

Description:
    The MOon REgolith Scattering (MORES) model
"""

import tomli
from pytmatrix.psd import ExponentialPSD

class MORES:
    """
    The MOon REgolith Scattering (MORES) class
    """
    def __init__(self, f=2.38e9, delta=1.26e-2, corr_len=12.6e-2, 
                 thickness=5.0, epsilon_r_background=2.7-0.1j, epsilon_r_particle=8.0-0.1j, axis_ratio=1.0, particle_size=(1.26e-2, 1), particle_orientation=(0, 0),
                 delta_sub=1.26e-2, corr_len_sub=12.6e-2, epsilon_r_sub=8-0.1j):
        """
        Init a MORES model.
        INPUT:
            see model.toml for reference
        OUTPUT:
            mores: The mores object
        """
        self.freq = f
        self.delta = delta
        self.corr_len = corr_len
        self.thickness = thickness
        self.epsilon_r_background = epsilon_r_background
        self.epsilon_r_particle = epsilon_r_particle
        self.axis_ratio = axis_ratio
        self.particle_size = particle_size
        self.particle_orientation = particle_orientation
        self.delta_sub = delta_sub
        self.corr_len_sub = corr_len_sub
        self.epsilon_r_sub = epsilon_r_sub
    
    def init_from_model_file(self, toml_file):
        """
        Init the model using parameters from a toml description file
        Only Exponential PSD is supported by this method. For other PSDs use `MORES(param...)`
        """
        with open(toml_file, "rb") as f:
            doc = tomli.load(f)
        
        # radar
        self.freq = doc['radar']['f']
        # surface
        self.delta = doc['surface']['delta']
        self.corr_len = doc['surface']['corr_len']

        # layer
        self.thickness = doc['layer']['thickness']
        self.epsilon_r_background = complex(*doc['layer']['epsilon_r_background'])
        self.epsilon_r_particle = complex(*doc['layer']['epsilon_r_particle'])
        particle_size = tuple(doc['layer']['particle_size'])
        if len(particle_size) == 4:
            # multi-sizes
            assert(particle_size[0] == 0)
            self.particle_size = self.particle_size[0:2] + \
                (ExponentialPSD(N0=self.particle_size[2], Lambda=self.particle_size[3], D_max=self.particle_size[1]),)

        particle_orientation = doc['layer']['particle_orientation']
        if type(particle_orientation) == list:
            self.particle_orientation = tuple(doc['layer']['particle_orientation'])
        else:
            self.particle_orientation = particle_orientation
        self.axis_ratio = doc['layer']['axis_ratio']