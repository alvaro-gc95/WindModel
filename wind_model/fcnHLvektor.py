import numpy as np
import numba as nb


@nb.njit
def fcnHLvektor(flux, field_dimensions):
    # Make a vector of the matrix
    n = 0
    b = np.zeros(flux.shape[0] * flux.shape[1] * flux.shape[2])
    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):
                b[n] = flux[i, j, k]
                n = n + 1

    #b = np.reshape(flux, (flux.shape[0] * flux.shape[1] * flux.shape[2]))

    return b
