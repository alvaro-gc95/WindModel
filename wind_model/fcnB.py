from wind_model.continuity import continuity_equation
from wind_model.fcnHLvektor import fcnHLvektor
import numpy as np
import numba as nb


def fcnB(x, y, z, zx, zy, u_original, v_original, w_orography, field_dimensions):
    # Calculate the divergence in the input wind field
    flux = continuity_equation(x, y, z, zx, zy, u_original, v_original, w_orography, field_dimensions)
    # Make a vector of the matrix
    b = fcnHLvektor(flux, field_dimensions)
    # Change the sign of the vector
    b = -1. * np.matrix(b).getH()
    #b = -1 * b.conj().transpose()

    return b
