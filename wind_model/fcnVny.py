import numpy as np
from wind_model.discretization import differenciate, differenciate_boundaries


def fcnVny(y, v_original, zy, Lmat, field_dimensions, alfa):
    aY = 1 / alfa[1] ** 2
    Vny = np.zeros((field_dimensions[0], field_dimensions[1], field_dimensions[2]))
    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):
                if k == 0:
                    Vny[i, j, k] = 0

                elif j == 0:
                    Vny[i, j, k] = v_original[i, j, k] + aY * differenciate_boundaries(y[j], y[j + 1], Lmat[i, j, k], Lmat[i, j + 1, k])

                elif j == field_dimensions[1]-1:
                    Vny[i, j, k] = v_original[i, j, k] + aY * differenciate_boundaries(y[j - 1], y[j], Lmat[i, j - 1, k], Lmat[i, j, k])
                else:
                    Vny[i, j, k] = v_original[i, j, k] + aY * differenciate(y[j - 1], y[j], y[j + 1], Lmat[i, j - 1, k],
                                                                            Lmat[i, j, k],
                                                                            Lmat[i, j + 1, k]) + aY * zy[i, j] * Lmat[i, j, k]

    return Vny
