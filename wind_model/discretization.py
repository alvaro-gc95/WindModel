import numba as nb

@nb.njit
def differenciate(x1, x2, x3, y1, y2, y3):
    """
    Kapotza & Eppel (1986) first order differential operator. Eq. (4.15a)
    Subindex naming:
        - 1 = i-1
        - 2 = i
        - 3 = i+1
    """

    h1 = x2 - x1
    h2 = x3 - x2
    d = ((h1**2) * y3 - ((h1 ** 2) - (h2 ** 2)) * y2 - (h2 ** 2) * y1) / (h1 * h2 * (h1 + h2))

    return d

@nb.njit
def differenciate_boundaries(x1, x2, y1, y2):
    """
    First order differential operator for boundary points
    """
    d = (y2 - y1) / (x2 - x1)
    return d
