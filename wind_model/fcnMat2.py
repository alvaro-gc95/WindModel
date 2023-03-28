from scipy.sparse import csr_matrix
# import dask.array as da
import numpy as np
import tqdm
import itertools


def fcnMat2(x, y, Z, zx, zy, alfa, field_dimensions):

    a_x = 1 / alfa[0] ** 2
    a_y = 1 / alfa[1] ** 2
    a_z = 1 / alfa[2] ** 2

    M = np.zeros((field_dimensions[0] * field_dimensions[1] * field_dimensions[2],
                  field_dimensions[0] * field_dimensions[1] * field_dimensions[2]))
    # M = csr_matrix((st[0] * st[1] * st[2], st[0] * st[1] * st[2])).todense()
    #M = M.map_blocks(csr_matrix)
    #M = M.rechunk(chunks=(M.shape))
    # M = M.compute().toarray()

    # n = 0
    # for i in tqdm.tqdm(range(st[0]-1)):
    #     for j in range(st[1]-1):
    #         for k in range(st[2]-1):

    for n, idx in tqdm.tqdm(enumerate(itertools.product(range(field_dimensions[0]), range(field_dimensions[1]), range(field_dimensions[2])))):

        i, j, k = idx

        M[n, n] = 1

        if (
                i == 0 or i == field_dimensions[0]-1 or
                j == 0 or j == field_dimensions[1]-1 or
                k == 0 or k == field_dimensions[2]-1
        ):
            if n != field_dimensions[0]*field_dimensions[1]*field_dimensions[2]-1:
                M[n, n + 1] = 1
            M[n, n] = -1

        else:
            x1 = x[i] - x[i - 1]
            x2 = x[i + 1] - x[i]

            y1 = y[j] - y[j - 1]
            y2 = y[j + 1] - y[j]

            z1 = Z[k] - Z[k - 1]
            z2 = Z[k + 1] - Z[k]

            M[n, n] = (-2 * a_y / (y1 * y2)) \
                      - (2 * a_x / (x1 * x2)) \
                      - (2 * a_z / (z1 * z2)) \
                      - a_x * zx[i, j] ** 2 \
                      - a_y * zy[i, j] ** 2

            M[n, n - 1] = 2 * a_z / (z1 * (z1 + z2))
            M[n, n + 1] = 2 * a_z / (z2 * (z1 + z2))

            M[n, n - field_dimensions[2]] = 2 * a_y / (y1 * (y1 + y2)) \
                                            + a_y * zy[i, j] * y2 / (y1 * (y1 + y2)) \
                                            - a_y * zy[i, j - 1] * y2 / (y1 * (y1 + y2))
            M[n, n + field_dimensions[2]] = 2 * a_y / (y2 * (y1 + y2)) \
                                            - a_y * zy[i, j] * y1 / (y2 * (y1 + y2)) \
                                            + a_y * zy[i, j + 1] * y1 / (y2 * (y1 + y2))

            M[n, n - (field_dimensions[2] * field_dimensions[1])] = 2 * a_x / (x1 * (x1 + x2)) \
                                                                    + a_x * zx[i, j] * x2 / (x1 * (x1 + x2)) \
                                                                    - a_x * zx[i - 1, j] * x2 / (x1 * (x1 + x2))
            M[n, n + (field_dimensions[2] * field_dimensions[1])] = 2 * a_x / (x2 * (x1 + x2)) \
                                                                    - a_x * zx[i, j] * x1 / (x2 * (x1 + x2)) \
                                                                    + a_x * zx[i + 1, j] * x1 / (x2 * (x1 + x2))

    # n = n + 1

    return M
