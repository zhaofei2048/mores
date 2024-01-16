"""
Author: Fei Zhao
Create: 2023-08-30

Description:
    Tools for init the models
"""
import numpy as np
import scipy.constants as C

from app_utils import do_import_class
# from Prescribedkskaeps import Prescribedkskaeps
from RayleighSphereKsKa import RayleighSphereKsKa
from TmatrixScatterer import TmatrixScatterer


def init_surface_from_mores(surface_class_name, mores):
    """
    Init a surface using parameters from a MORES object
    INPUT:
        surface_class_name: should be an instance of class inherited from Surface
        mores: a MORES object
    OUTPUT:
        surface: an initialized surface
    """
    f = mores.freq
    Lambda = C.speed_of_light / f
    k = 2 * np.pi / Lambda
    kdel = k * mores.delta
    kcor = k * mores.corr_len
    epsr_background = mores.epsilon_r_background

    surface_class = do_import_class(surface_class_name)
    if str.lower(surface_class_name) == 'aiem':
        surface = surface_class(epsr_background, kdel, kcor)
    elif str.lower(surface_class_name) == 'transparentsurface':
        surface = surface_class(epsr_background)
    elif str.lower(surface_class_name) == 'smoothsurface':
        surface = surface_class(epsr_background)
    else:
        raise ValueError('Un-supported surface class')

    return surface


def init_layer_from_mores(layer_class_name, mores):
    """
    Init a layer using parameters from a MORES object
    INPUT:
        layer_class_name: should be an instance of class inherited from Layer
        mores: a MORES object
    OUTPUT:
        layer: an initialized layer
    """
    f = mores.freq
    # for layer
    thickness = mores.thickness
    epsr_background = mores.epsilon_r_background
    epsr_particle = mores.epsilon_r_particle
    particle_size = mores.particle_size
    radius = particle_size[0]
    fs = particle_size[1]

    layer_class = do_import_class(layer_class_name)
    if str.lower(layer_class_name) == 'rayleighsphere':
        layer = layer_class(f, epsr_background, epsr_particle, particle_size, thickness)
    elif str.lower(layer_class_name) == 'tmatrixscatterer':
        particle_orientation = mores.particle_orientation
        particle_shape = (mores.axis_ratio, 'SPHEROID')
        layer = layer_class(f, epsr_background, epsr_particle, particle_size, particle_orientation, particle_shape, thickness)
    else:
        raise ValueError('Un-supported surface class')

    return layer