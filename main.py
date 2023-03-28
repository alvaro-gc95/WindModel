import numpy as np
# import dask.array as da
from wind_model.fcnKorM2 import fcnKorM2
import matplotlib.pyplot as plt
import wind_model.windfield
import wind_model.orography
import time


t0 = time.time()

# Load Digital Elevation Model
resolution = 200
dem_path = '/home/alvaro/data/DEM/'
dem_file = 'dtm_' + str(resolution) + '.nc'

dem = wind_model.orography.open_dem(dem_path + dem_file)
dem_delta = dem.dem_map_delta

# Average each dx cells
dx = 1

# All level heights [m]
levels = [0, 2, 4, 10, 20, 50, 100, 150, 200, 300]
levels.extend(list(np.arange(400, 2200, 200)))

# Model top
model_top = levels[-1]

# Number of levels to compute
n_levels = 10

# Orography values
x = [i * dx for i in range(dem_delta.shape[0])]
y = [j * dx for j in range(dem_delta.shape[1])]

# Get orography gradients (dzdx, dzdy) and x and y coordinate coefficients (zx, zy)
dzdx, dzdy, zy, zx = dem.get_gradients(dx, model_top)

# Original wind components [x, y, level]
u_original = np.ones((dem_delta.shape[0], dem_delta.shape[1], n_levels))
v_original = np.ones((dem_delta.shape[0], dem_delta.shape[1], n_levels))*-1
w_original = np.zeros((dem_delta.shape[0], dem_delta.shape[1], n_levels))

# Get wind profile
u_original, v_original = wind_model.windfield.calculate_logarithmic_profile(u_original, v_original, levels, levels=n_levels)
field_dimensions = [dem_delta.shape[0], dem_delta.shape[1], n_levels]

# Gaussian Precision Moduli
alpha = [1, 1, 4]

# Calculate new field
u_new, v_new, w_orography_new, w_new = fcnKorM2(x, y, levels, dem_delta, zx, zy, u_original, v_original, w_original, field_dimensions, alpha)

# Get horizontal wind speed
horizontal_wind_speed = np.sqrt(u_new ** 2 + v_new ** 2)

# Get wind speed
wind_speed = np.sqrt(u_new ** 2 + v_new ** 2 + w_new ** 2)

t1 = time.time()

total = t1-t0

print(total)

if __name__ == '__main__':

    x_mesh, y_mesh = np.meshgrid(x, y)
    """
    fig = plt.figure()
    ax = fig.subplots(1)
    ax.plot(u_original[0, 0, :], range(n_levels))
    ax.plot(v_original[0, 0, :], range(n_levels))
    plt.show()

    # Plot the orography
    fig = plt.figure(figsize=(20, 5))
    ax = fig.subplots(1, 3)
    elevation_cont = ax[0].contourf(dem_delta, 500, cmap='gist_earth')
    cnt = ax[0].contour(dem_delta, 20, cmap='Greys', linewidth=0.5)
    ax[0].set_title('elevation')
    plt.colorbar(elevation_cont, ax=ax[0])

    xgrad_cont = ax[1].contourf(dzdx, 500, cmap='coolwarm')
    cnt = ax[1].contour(dem_delta, 20, cmap='Greys', linewidth=0.5)
    ax[1].set_title('x gradient')
    plt.colorbar(xgrad_cont, ax=ax[1])

    ygrad_cont = ax[2].contourf(dzdy, 500, cmap='coolwarm')
    cnt = ax[2].contour(dem_delta, 20, cmap='Greys', linewidth=0.5)
    ax[2].set_title('y gradient')
    plt.colorbar(ygrad_cont, ax=ax[2])
    plt.show()

    # Plot wind profiles
    fig = plt.figure()
    ax = fig.subplots(1)
    ax.plot(u_new[0, 0, :], range(n_levels), color='blue')
    ax.plot(v_new[0, 0, :], range(n_levels), color='red')
    ax.plot(w_new[0, 0, :], range(n_levels), color='green')
    plt.show()

    # Plot wind fields
    for level in range(n_levels):
        fig = plt.figure(figsize=(30, 10))
        ax = fig.subplots(3, 2)

        fig.suptitle('LEVEL ' + str(level) + ': ' + str(levels[level]))

        uo_contour = ax[0, 0].contourf(u_original[:, :, level], 500, cmap='coolwarm')
        ax[0, 0].contour(dem_delta, 20, cmap='Greys')
        ax[0, 0].set_title('u component')
        fig.colorbar(uo_contour, ax=ax[0, 0])

        vo_contour = ax[0, 1].contourf(v_original[:, :, level], 500, cmap='coolwarm')
        ax[0, 1].contour(dem_delta, 20, cmap='Greys')
        ax[0, 1].set_title('v component')
        fig.colorbar(vo_contour, ax=ax[0, 1])

        u_contour = ax[1, 0].contourf(u_new[:, :, level], 500, cmap='coolwarm')
        ax[1, 0].contour(dem_delta, 20, cmap='Greys')
        ax[1, 0].set_title('u component')
        fig.colorbar(u_contour, ax=ax[1, 0])

        v_contour = ax[1, 1].contourf(v_new[:, :, level], 500, cmap='coolwarm')
        ax[1, 1].contour(dem_delta, 20, cmap='Greys')
        ax[1, 1].set_title('v component')
        fig.colorbar(v_contour, ax=ax[1, 1])

        w_contour = ax[2, 0].contourf(w_new[:, :, level], 500, cmap='coolwarm')
        ax[2, 0].contour(dem_delta, 20, cmap='Greys')
        ax[2, 0].set_title('w component')
        fig.colorbar(w_contour, ax=ax[2, 0])

        hspeed_contour = ax[2, 1].contourf(horizontal_wind_speed[:, :, level], 500)
        ax[2, 1].contour(dem_delta, 20, cmap='Greys')
        # ax[2, 1].quiver(y_mesh, x_mesh, u_new[:, :, level], v_new[:, :, level])
        ax[2, 1].set_title('horizontal module')
        fig.colorbar(hspeed_contour, ax=ax[2, 1])
        plt.show()
    """
