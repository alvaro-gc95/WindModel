import numba as nb


@nb.njit
def orography_to_cartesian(z, z_ground, zx, zy, u, v, w_orography, field_dimensions):
    """"
    Transform the vertical wind component to Cartesian coordinate system, see Eq. (4.8)
    """

    w = w_orography.copy()

    for i in range(field_dimensions[0]):
        for j in range(field_dimensions[1]):
            for k in range(field_dimensions[2]):
                x_term = (- u[i, j, k] * (z[k] - z[field_dimensions[2]-1])) * zx[i, j]
                y_term = (- v[i, j, k] * (z[k] - z[field_dimensions[2]-1])) * zy[i, j]
                # z_term = w_orography[i, j, k] * 1 / z[field_dimensions[2]-1] * (z[field_dimensions[2]-1] - z_ground[i, j])
                z_term = w_orography[i, j, k] * z[field_dimensions[2] - 1] / \
                         (z[field_dimensions[2] - 1] - z_ground[i, j])
                w[i, j, k] = x_term + y_term + z_term

    return w
