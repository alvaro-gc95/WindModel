import numpy as np
from wind_model.discretization import differenciate, differenciate_boundaries


def fcnWny(z, w_original, Lmat, field_dimensions, alfa):
    aZ = alfa[2] ** (-2)
    Wny = np.zeros((field_dimensions[0], field_dimensions[1], field_dimensions[2]))
    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):
                if k == 0:
                    Wny[i, j, k] = w_original[i, j, k] + aZ * differenciate(-z[k + 1], z[k], z[k + 1], Lmat[i, j, k], Lmat[i, j, k],
                                                                            Lmat[i, j, k + 1])

                elif k == field_dimensions[2]-1:
                    Wny[i, j, k] = w_original[i, j, k] + aZ * differenciate_boundaries(z[k - 1], z[k], Lmat[i, j, k - 1], Lmat[i, j, k])
                else:
                    Wny[i, j, k] = w_original[i, j, k] + aZ * differenciate(z[k - 1], z[k], z[k + 1], Lmat[i, j, k - 1],
                                                                            Lmat[i, j, k],
                                                                            Lmat[i, j, k + 1])

                return Wny
