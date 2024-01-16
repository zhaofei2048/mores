from icecream import ic

def print_scatterer(scatterer):
    """
    Print the parameters of a pytmatrix scatterer
    """
    print('---Configuration of the scatterer---')
    print('radius: {}'.format(scatterer.radius))
    print('radius_type: {}'.format(scatterer.radius_type))
    print('wavelength: {}'.format(scatterer.wavelength))
    print('m: {}'.format(scatterer.m))
    print('axis_ratio: {}'.format(scatterer.axis_ratio))
    print('shape: {}'.format(scatterer.shape))
    print('orient: {}'.format(scatterer.orient))
    print('psd_integrator: {}'.format(scatterer.psd_integrator))