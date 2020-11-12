import numpy as np
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
    # Add scatter, if requested.
    if not scatter == False:
        internal_stellar_mass += np.random.normal(scale=scatter, size=np.shape(internal_stellar_mass))

    return internal_stellar_mass

def stellar_mass_to_halo_mass(stellar_mass, z, formula="Grylls19"):
    stellar_mass = np.array(stellar_mass)
    if hasattr(z, "__len__"):
        z = np.array(stellar_mass)
        halo_masses = np.zeros_like(stellar_mass)
        for i, zi in enumerate(z):
            halo_array = np.linspace(5, 20, 1000)
            SM_array = halo_mass_to_stellar_mass(halo_array, zi, formula)
            sm2hm = interp1d(SM_array, halo_array)
            halo_masses[i] = sm2hm(stellar_mass[i])
        return halo_masses
    else:
        halo_array = np.linspace(5, 20, 1000)
        SM_array = halo_mass_to_stellar_mass(halo_array, z, formula)
        sm2hm = interp1d(SM_array, halo_array)
        return sm2hm(stellar_mass)
