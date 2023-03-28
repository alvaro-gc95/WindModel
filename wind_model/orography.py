import itertools

import netCDF4 as nc
import numpy as np


class DigitalTerrainModel:

    def __init__(self, lon_ini, lon_end, lat_ini, lat_end, longitudes, latitudes, dem_map, skv):
        self.lon_ini = lon_ini
        self.lon_end = lon_end
        self.lat_ini = lat_ini
        self.lat_end = lat_end
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.dem_map = dem_map
        self.skv = skv
        self.dem_map_delta = self.get_height_difference()

    def get_height_difference(self):
        dem_map_min = min(self.dem_map.flatten())
        dem_map_delta = self.dem_map - dem_map_min

        return dem_map_delta

    def get_gradients(self, dx, top):
        dzdy = np.zeros(self.dem_map_delta.shape)
        dzdx = np.zeros(self.dem_map_delta.shape)
        zy = np.zeros(self.dem_map_delta.shape)
        zx = np.zeros(self.dem_map_delta.shape)

        for i, j in itertools.product(range(self.dem_map_delta.shape[0] - 1), range(self.dem_map_delta.shape[1] - 1)):
            # X gradient
            dzdx[i, j] = (self.dem_map_delta[i, j + 1] - self.dem_map_delta[i, j]) / dx
            # Y gradient
            dzdy[i, j] = (self.dem_map_delta[i + 1, j] - self.dem_map_delta[i, j]) / dx
            # Orography terms for "follow the orography" coordinates
            zy[i, j] = dzdy[i, j] / (top - self.dem_map_delta[i, j])
            zx[i, j] = dzdx[i, j] / (top - self.dem_map_delta[i, j])

        return dzdx, dzdy, zy, zx

    def get_slopes(self, dx, top, resolution):
        dzdx, dzdy, _, _ = self.get_gradients(dx, top)

        slope_x = np.degrees(np.arcsin(dzdx / np.sqrt(dzdx ** 2 + resolution ** 2)))
        slope_y = np.degrees(np.arcsin(dzdy / np.sqrt(dzdy ** 2 + resolution ** 2)))

        return slope_x, slope_y


def open_dem(path, skyview=False):
    dem = nc.Dataset(path)

    lon_ini = dem['XLONG'][0, 0, 0]
    lon_end = dem['XLONG'][0, 0, -1]
    lat_ini = dem['XLAT'][0, 0, 0]
    lat_end = dem['XLAT'][0, -1, 0]
    longitudes = dem['XLONG'][0, 0, :]
    latitudes = dem['XLAT'][0, :, 0]
    dem_map = np.squeeze(dem['HGT'][0, :, :])
    if skyview:
        skv = np.squeeze(dem['SKV'][0, :, :])
    else:
        skv = np.ones(dem_map.shape)

    return DigitalTerrainModel(lon_ini, lon_end, lat_ini, lat_end, longitudes, latitudes, dem_map, skv)
