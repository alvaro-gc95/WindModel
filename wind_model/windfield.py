import numpy as np

z0 = 0.01  # Roughness length


def calculate_logarithmic_profile(u, v, z, levels):

    for n in range(levels):
        u[:, :, n] = u[:, :, n] * np.log(z[n]/z0)
        v[:, :, n] = v[:, :, n] * np.log(z[n]/z0)
        #u[:, :, n] = 1.174888 * np.log(z[n]) + 5.4035
    return u, v
