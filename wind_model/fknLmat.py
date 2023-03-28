import numpy as np


def fknLmat(L, field_dimensions):

    Lmat = np.zeros((field_dimensions[0], field_dimensions[1], field_dimensions[2]))
    # Make a matrix of the vector L
    n = 0
    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):

                Lmat[i, j, k] = L[n]
                n = n + 1

    return Lmat
