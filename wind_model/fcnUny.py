from wind_model.discretization import differenciate, differenciate_boundaries
import numpy as np
import numba as nb


@nb.njit
def fcnUny(x, u_original, zx, Lmat, field_dimensions, alfa):

    aX = alfa[0] ** (-2)
    Uny = np.zeros((field_dimensions[0], field_dimensions[1], field_dimensions[2]))

    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):
                if k == 0:
                    Uny[i, j, k] = 0

                elif i == 0:
                    Uny[i, j, k] = u_original[i, j, k] + aX * differenciate_boundaries(x[i], x[i + 1], Lmat[i, j, k], Lmat[i + 1, j, k])

                elif i == field_dimensions[0]-1:

                    Uny[i, j, k] = u_original[i, j, k] + aX * differenciate_boundaries(x[i - 1], x[i], Lmat[i - 1, j, k], Lmat[i, j, k])
                else:

                    Uny[i, j, k] = u_original[i, j, k] + aX * differenciate(x[i - 1], x[i], x[i + 1], Lmat[i - 1, j, k],
                                                                            Lmat[i, j, k],
                                                                            Lmat[i + 1, j, k]) + aX * zx[i, j] * Lmat[i, j, k]

    return Uny
