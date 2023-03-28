import numpy as np
from wind_model.discretization import differenciate
from numba import njit


@njit
def continuity_equation(x, y, z, zx, zy, u, v, w, field_dimensions):
    """
    Continuity equation in "following the terrain" coordinates
    """

    flux_continuity = np.zeros((field_dimensions[0], field_dimensions[1], field_dimensions[2]))

    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):

                if (
                        i == 0 or i == field_dimensions[0] - 1 or
                        j == 0 or j == field_dimensions[1] - 1 or
                        k == 0 or k == field_dimensions[2] - 1
                ):

                    flux_continuity[i, j, k] = 0

                else:

                    # Calculate the divergence from Eq. (4.17)
                    x_diff = differenciate(x[i - 1], x[i], x[i + 1], u[i - 1, j, k], u[i, j, k], u[i + 1, j, k])
                    y_diff = differenciate(y[j - 1], y[j], y[j + 1], v[i, j - 1, k], v[i, j, k], v[i, j + 1, k])
                    z_diff = differenciate(z[k - 1], z[k], z[k + 1], w[i, j, k - 1], w[i, j, k], w[i, j, k + 1])

                    # Orography terms
                    orography_x = - zx[i, j] * u[i, j, k]
                    orography_y = - zy[i, j] * v[i, j, k]

                    # Continuity
                    flux_continuity[i, j, k] = x_diff + y_diff + z_diff + orography_x + orography_y

    return flux_continuity
