from wind_model.fcnB import fcnB
from wind_model.fcnMat2 import fcnMat2
from wind_model.fknLmat import fknLmat
from wind_model.fcnUny import fcnUny
from wind_model.fcnVny import fcnVny
from wind_model.fcnWny import fcnWny
from wind_model.coordinates import orography_to_cartesian
import numpy as np
import dask.array as da
import numba as nb
import cupy as cp
from cupy.linalg import cublas, cusolver


def solve_system(a, b):
    # Copy the input matrices to the GPU
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)

    # Create a cuBLAS handle for matrix operations
    cublas_handle = cublas.create()

    # Create a cuSolver handle for solving the linear system
    cusolver_handle = cusolver.create()

    # Compute the least-squares solution on the GPU
    x_gpu, _, _, _ = cusolver.lstsq(cusolver_handle, cublas_handle, a_gpu, b_gpu)

    # Copy the result back to the CPU
    x = cp.asnumpy(x_gpu)

    return x




def fcnKorM2(x, y, z, z_ground, zx, zy, u_original, v_original, w_orography, field_dimensions, alfa):
    # Creates right - hand side
    b = fcnB(x, y, z, zx, zy, u_original, v_original, w_orography, field_dimensions)

    # Creates matrix A
    A = fcnMat2(x, y, z, zx, zy, alfa, field_dimensions)

    # Transform to dask
    A = da.from_array(A).rechunk({0: 'auto', 1: -1})
    b = da.from_array(b).rechunk({0: 'auto', 1: -1})

    # Solve the equation system from Eq. (4.16)
    L = solve_system(A.compute(), b.compute())
    # L, _, _, _ = np.linalg.lstsq(A.compute(), b.compute())

    # Make the matrix of the vector
    Lmat = fknLmat(L, field_dimensions)

    # Calculate the new wind field
    u_new = fcnUny(x, u_original, zx, Lmat, field_dimensions, alfa)
    v_new = fcnVny(y, v_original, zy, Lmat, field_dimensions, alfa)
    w_orography_new = fcnWny(z, w_orography, Lmat, field_dimensions, alfa)

    # Transform the vertical wind component to Cartesian coordinate system
    w_new = orography_to_cartesian(z, z_ground, zx, zy, u_new, v_new, w_orography_new, field_dimensions)

    return u_new, v_new, w_orography_new, w_new
