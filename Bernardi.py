import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from colossus.cosmology import cosmology
from scipy import interpolate
from scipy.interpolate import griddata

def bernardi_k(n, Rre):
    """ (internal) function to interpolate the constant K in bernardi's sigam formalism.

    params:
        n (float, single value/array) : the sersic index (dimensionless)
        Rre (float, single value/array) : the ratio of the aperture to radius
        (dimensionless)
    returns:
        (float, single value/array) : the value of K (dimensionless)
    """

    bernardi_table = np.array([[7.38, 7.20, 6.80, 6.78, 6.97, 7.30],
                               [6.59, 6.46, 6.23, 6.36, 6.63, 6.97],
                               [5.84, 5.76, 5.69, 5.96, 6.27, 6.62],
                               [5.18, 5.15, 5.21, 5.57, 5.92, 6.27],
                               [4.62, 4.62, 4.79, 5.21, 5.58, 5.93],
                               [4.14, 4.17, 4.42, 4.88, 5.26, 5.60],
                               [3.74, 3.79, 4.09, 4.58, 4.95, 5.29],
                               [3.39, 3.46, 3.79, 4.29, 4.67, 4.99],
                               [3.10, 3.17, 3.52, 4.03, 4.40, 4.71],
                               [2.84, 2.92, 3.28, 3.78, 4.14, 4.44],
                               [2.61, 2.70, 3.06, 3.56, 3.91, 4.19],
                               [2.41, 2.50, 2.86, 3.35, 3.68, 3.95],
                               [2.23, 2.32, 2.68, 3.15, 3.47, 3.73],
                               [2.07, 2.16, 2.51, 2.96, 3.27, 3.52],
                               [1.92, 2.01, 2.36, 2.79, 3.08, 3.32],
                               [1.79, 1.88, 2.21, 2.63, 2.91, 3.13],
                               [1.67, 1.75, 2.08, 2.48, 2.74, 2.95]])

    #Rre[Rre < 0.1] = 0.1

    bernardi_rre_range = np.array([0.1, 0.125, 0.25, 0.5, 0.75, 1])
    bernardi_n_range = np.array([2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00,
                                 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00])
    bernardi_x = np.tile(bernardi_rre_range, len(bernardi_n_range)).ravel()
    bernardi_y = np.tile(np.vstack(bernardi_n_range), len(bernardi_rre_range)).ravel()
    bernardi_data = (bernardi_x, bernardi_y)
    sample = (Rre, n)
    grid2 = griddata(bernardi_data, bernardi_table.ravel(), sample, method='cubic')

    return grid2


def bernardi_sigma(Aperture, radius, n, sm):
    """ Function to return value of sigma, based on Bernardi's formalism.
    params:
        Aperture (float, single value/array) : the size of the aperture (kpc)
        radius (float, single value/array) : half light radius of the galaxy (kpc)
        n (float, single value/array): sersic index of the galaxy (dimensionless)
        sm (float, single value/array): stellar mass of the galaxy (*M_sun*)
    returns:
        (float, single value/array) : the velocity dispersion of the galaxy (kms^-1)
    """
    return ((4.3009125e-6*sm)/(bernardi_k(n, Aperture/radius) * radius))**0.5

if __name__=="__main__":
    print(bernardi_sigma(6.6, 6.6, 4, 10**11))
