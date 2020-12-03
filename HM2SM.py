import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d


def halo_mass_to_stellar_mass(halo_mass, z, formula="Grylls19", scatter=0.11):
    """Function to generate stellar masses from halo masses.
    This is based on Grylls 2019, but also has the option to use the
    parameters from Moster. This is a simplified version of Pip's
    DarkMatterToStellarMass() function.
    :param halo_mass: array, of halo masses (log10)
    :param z: float, the value of redshift
    :param formula: string, the method to use. Options currently include "Grylls19" and "Moster"
    :param scatter: bool, to scatter or not
    :return array, of stellar masses (log10).
    """

    # If conditions to set the correct parameters.
    if formula == "Grylls19":
        z_parameter = np.divide(z - 0.1, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.95, 0.032, 1.61, 0.54
        m_11, shm_norm_11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
    elif formula == "Moster":
        z_parameter = np.divide(z, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.590, 0.0351, 1.376, 0.608
        m_11, shm_norm_11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    elif formula == "NoKnee":
        z_parameter = np.divide(z, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.590, 0.0351, 0.0, 0.608
        m_11, shm_norm_11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    else:
        assert False, "Unrecognised formula"

    # Create full parameters
    m = m_10 + m_11 * z_parameter
    n = shm_norm_10 + shm_norm_11 * z_parameter
    b = beta10 + beta11 * z_parameter
    g = gamma10 + gamma11 * z_parameter
    # Full formula
    internal_stellar_mass = np.log10(np.power(10, halo_mass) *\
                                     (2 * n * np.power((np.power(np.power(10, halo_mass - m), -b)\
                                                        + np.power(np.power(10, halo_mass - m), g)), -1)))

    if formula == "Grylls19":
        internal_stellar_mass -= 0.1

    # Add scatter, if requested.
    if not scatter == False:
        internal_stellar_mass += np.random.normal(scale=scatter, size=np.shape(internal_stellar_mass))

    return internal_stellar_mass

def stellar_mass_to_halo_mass(stellar_mass, z, formula="Grylls19"):

    if formula == "Grylls19":
        mdef = 'vir'
    elif formula == "Moster":
        mdef = '200c'


    def get_sm_fixedz(sm, z_fixed):
        max_guess = 25
        min_guess = 1
        catalog = np.random.rand(1000000) *  (max_guess-min_guess) + min_guess

        catalog_sm = halo_mass_to_stellar_mass(catalog, z_fixed) # With scatter
        width = 0.05
        sm_range = np.arange(8, 15, width)
        means = binned_statistic(catalog_sm, catalog, bins=sm_range)[0]
        sm_range = sm_range[:-1] + width/2

        sm2hm = interp1d(sm_range, means, bounds_error = False, fill_value="extrapolate")
        mass = sm2hm(sm)
        return mass


    stellar_mass = np.array(stellar_mass)
    if hasattr(z, "__len__"):
        assert len(z) == len(stellar_mass), "Length of sm ({}) != length z ({})".format(len(haloes), len(z))
        unique_z, indexes = np.unique(z, return_index=True)

        haloes_store = np.zeros_like(stellar_mass)

        for i, zi in enumerate(unique_z):
            hm = get_sm_fixedz(stellar_mass[z  == zi], zi)
            haloes_store[z == zi] = hm
        return haloes_store
    else:
        hm = get_sm_fixedz(stellar_mass, z)
        return hm


if __name__ == "__main__":
    stellar_masses  = [11, 10.5, 12]
    z = [0, 1, 1.0]
    haloes = stellar_mass_to_halo_mass(stellar_masses, z)
