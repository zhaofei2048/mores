"""
Author: Fei Zhao
Create: 2023-09-01

Description:
    Tool functions.
"""

import numpy as np
import importlib
import inspect


def do_import_class(modulename, classname=None):
    """Import the class, adapted from SMRT
    """
    # check the module
    #spec = importlib.util.find_spec(modulename)
    #if spec is None:
    #    return None

    # import the module

    try:
        module = importlib.import_module(modulename)
    except ModuleNotFoundError:
        return None

    if classname is None:  # search for the first class defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == modulename:  # the second condition check if the class was defined in this module
                classname = name
                break

    if classname is None:
        raise ValueError("Unable to find a class in the module '%s'" % modulename)

    # get the class
    return getattr(module, classname)