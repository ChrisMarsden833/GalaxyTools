import numpy as np
import pandas as pd
import scipy.interpolate as spi
import scipy as sp
from colossus.lss import mass_function

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def getCold(halo_mass, z, path="/Users/chris/data/haloConcentration/cM_planck18.txt"):
    """function to get the halo concentration parameter from the available
    file.
    """
    df = pd.read_csv(path, comment='#', sep='\s+')
    z_array = df['z']
    M200c = df['M200c']
    c200c = df['c200c']

    if hasattr(z, "__len__"):
        assert len(halo_mass) == len(z), "halo_mass (len {}) and z (len {}) must have the same length".format(len(halo_mass), len(z))
        c = np.zeros_like(halo_mass)
        for i, zi in enumerate(z):
            zi = find_nearest(np.unique(z_array), zi)
            m2c = spi.interp1d(np.log10(M200c[z_array==zi]), c200c[z_array==zi])
            c[i] = m2c(halo_mass[i])
        return c
    else:
        z = find_nearest(np.unique(z_array), z)
        m2c = spi.interp1d(np.log10(M200c[z_array==z]), c200c[z_array==z]) #  bounds_error = 'False', fill_value="extrapolate" )
        return m2c(halo_mass)

def getC(halo_mass, z, path="/Users/chris/data/haloConcentration/cM_planck18.txt", width=0.5):
    """A better function to get the halo concentration from the available file"""
    
    # File read in 
    df = pd.read_csv(path, comment='#', sep='\s+')
    z_array = df['z']
    M200c = np.log10(df['M200c'])
    c200c = df['c200c']
    
    # Make sure z and halo_mass are the same length
    halo_is_array = True if hasattr(halo_mass, "__len__") else False
    z_is_array = True if hasattr(z, "__len__") else False
    
    if halo_is_array and not z_is_array:
        z = np.ones_like(halo_mass) * z
    elif z_is_array and not halo_is_array:
        halo = np.ones_like(z) * halo_mass
    elif not z_is_array and not halo_is_array:
        halo_mass = np.array([halo_mass])
        z = np.array([z])
    elif z_is_array and halo_is_array:
        assert len(z) == len(halo_mass), "halo mass and z are not the same length ({} and {})".format(len(halo_mass), len(z))
    else:
        assert False, "Unreachable code: halo_is_array = {}, z_is_array = {}".format(halo_is_array, z_is_array)

    # Now everything should be ready - from now on we assume everything is an array with equal lengths!

    length = len(halo_mass)

    reservedC = np.zeros(length)

    for i, halo in enumerate(halo_mass):
        mask = (M200c > halo-width/2 ) & (M200c < halo+width/2)
        zs = z_array[mask]
        uniquez = np.unique(zs)
        meanC = np.zeros_like(uniquez)
        for j, zi in enumerate(uniquez):
            meanC[j] = np.mean(c200c[mask & (z_array == zi)])
        z2c = spi.interp1d(uniquez, meanC, kind = 'quadratic', bounds_error = False,
                fill_value='extrapolate')
        reservedC[i] =  z2c(z[i])

    if length > 1:
        return reservedC
    else:
        assert length == 1, "Length not 1 when length < 1, value: {}".format(length)
        return reservedC[0]



def generate_catalogue(cube_side, mass_params, z, h):
    """
    Function to generate the semi analytic halo catalogue (without coordinates) for galaxy testing
    :param catalogue_volume: float, cosmological volume within which to generate the catalog. [Mpc/h]^3
    :param mass_params: tuple, (mass low, mass high, spacing). log[Msun]
    :param z: float, redshift.
    :param h: float, reduced hubble constant.
    :return array, of halo masses. log[Msun]
    """

    print("Generating catalogue for a volume of ({:.2f} Mpc/h)^3\n".format(cube_side))

    catalogue_volume = cube_side**3

    # Get the bin width and generate the bins.
    bin_width = mass_params[2]
    mass_range = 10 ** np.arange(mass_params[0], mass_params[1], mass_params[2]) #log[Msun]

    # Generate the mass function itself - this is from the colossus toolbox
    local_mass_function = mass_function.massFunction(mass_range, z, mdef='200m', model='tinker08', q_out='dndlnM') \
        * np.log(10) / h  # dn/dlog10M

    # We determine the Cumulative HMF starting from the high mass end, multiplied by the bin width.
    # This effectively gives the cumulative probability of a halo existing.
    cumulative_mass_function = np.flip(np.cumsum(np.flip(local_mass_function, 0)), 0) * bin_width

    ########################################################################
    # Interpolation Tests
    # Interpolator for the testing - we will update this with the volume in a second.
    # This is essentially for a volume of size unity.
    interpolator = sp.interpolate.interp1d(cumulative_mass_function, mass_range)

    sample_index = int(np.floor(len(cumulative_mass_function) / 2))  # index of the half way point
    num_test = cumulative_mass_function[sample_index]  # The value of the cum function at this index
    mass_test = interpolator(num_test)  # Interpolate to get the mass that this predicts
    # Check that these values are equivalent.
    assert mass_range[sample_index] == mass_test, \
        "Interpolation method incorrect: Back interpolation at midpoint failed"
    # Check first element is equivalent to the total to 10 SF accuracy
    assert np.round(cumulative_mass_function[0], 10) ==\
        np.round(np.sum(local_mass_function) * bin_width, 10), "Final cum sum element != total sum"
    ########################################################################

    # Multiply by volume
    cumulative_mass_function = cumulative_mass_function * catalogue_volume

    # Get the maximum cumulative number.
    max_number = np.floor(cumulative_mass_function[0])
    range_numbers = np.arange(max_number)

    # Update interpolator
    interpolator = sp.interpolate.interp1d(cumulative_mass_function, mass_range)
    mass_catalog = interpolator(range_numbers[range_numbers >= np.amin(cumulative_mass_function)])

    print("Number of halos generated: {:d}\n".format(len(mass_catalog)))

    mass_catalog = np.log10(mass_catalog)
    return mass_catalog


if __name__ == "__main__":
    getC(14, 0)
