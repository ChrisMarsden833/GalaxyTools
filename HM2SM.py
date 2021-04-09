import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(0, "/Users/chris/Documents/ProjectSigma/DREAM/dream/")
from semi_analytic_catalog import generate_parents_catalogue
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology("planck18")


def halo_mass_to_stellar_mass(halo_mass, z, formula="Moster", scatter=0.11):
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


def similar_function(halo_mass, n, m, b, g):
    res = internal_stellar_mass = np.log10(np.power(10, halo_mass) *\
              (2 * n * np.power((np.power(np.power(10, halo_mass - m), -b)\
                        + np.power(np.power(10, halo_mass - m), g)), -1)))

    return res

def stellar_mass_to_halo_mass(stellar_mass, z, formula="Grylls19", mdef = 'vir'):

    try:
        cosmo = cosmology.get_cosmology()
    except:
        cosmo = cosmology.setCosmology("planck18")

    if formula == "Grylls19":
        mdef = 'vir'
    elif formula == "Moster":
        mdef = '200c'

    def similar_function(m, n, mn, beta, gamma, delta):
        res = 2. * n * m * ((10**(m-mn))**-beta + (10**(m-mn))**gamma)**-delta
        return res

    def get_sm_fixedz(sm, z_fixed, volume = 50):

        try:
            catalog = generate_parents_catalogue("tinker08", volume, (8, 16, 0.1), z_fixed, cosmo.h, mdef=mdef)
        except Exception as e:
            try:                
                catalog = generate_parents_catalogue("tinker08", volume, (8, 10, 0.1), z_fixed, cosmo.h, mdef=mdef)
            except Exception as e:
                raise Exception("Failed In generating catalog, exception was: {}. z was {}.".format(e, z))

        catalog_sm = halo_mass_to_stellar_mass(catalog, z_fixed, formula=formula) # With scatter
        width = 0.05
        sm_range = np.arange(5, 15, width)

        means = binned_statistic(catalog_sm, catalog, bins=sm_range)[0]

        sm_range = sm_range[:-1] + width/2

        safe_mask = ~np.isnan(means)

        snip_low = 10

        p0 = [0.5, 12, 0.52, 0.02, 1] # Initial Guess - this speeds up the process 'considerably'

        try:
            params = curve_fit(similar_function, means[safe_mask][snip_low:],
                                sm_range[safe_mask][snip_low:], p0 = p0)
            params = params[0]
    
            halo_domain = np.arange(6, 20, 0.1)
            ism_domain = similar_function(halo_domain, params[0], params[1], params[2], params[3], params[4])
            sm2hm = interp1d(ism_domain, halo_domain, bounds_error = False, fill_value="extrapolate")
            mass = sm2hm(sm)
            return mass
        except Exception as e:
            print(e) # You get into trouble otherwise
            sm2hm = interp1d(sm_range, means, bounds_error=False, fill_value='extrapolate')
            return sm2hm(sm)


    stellar_mass = np.array(stellar_mass)
    if hasattr(z, "__len__"):
        assert len(z) == len(stellar_mass), "Length of sm ({}) != length z ({})".format(len(haloes), len(z))

        binwidth = 0.05
        z_bins = np.arange(np.amin(z), np.amax(z), binwidth)
        indexes = np.digitize(z, z_bins)-1

        assert np.amax(indexes) < len(z_bins), "Maximum value of indexes ({}) is >= the number of bin elements ({}), which should not be possible".\
            format(np.amax(indexes), len(z_bins))
        assert np.amin(indexes) >= 0, "Minimum value of indexes ({}) is < 0.".format(np.amin(indexes))

        haloes_store = np.ones_like(stellar_mass) * -1

        track = 0

        print("Working through {} unique redshift steps.".format(len(np.unique(indexes))))
        for i, zstep in enumerate(tqdm(z_bins + binwidth/2.)):
            indexes_in_step = indexes == i
            track += np.sum(indexes_in_step)
            redshift_sample = stellar_mass[indexes_in_step]
            hm = get_sm_fixedz(redshift_sample, zstep)
            haloes_store[indexes_in_step] = hm

        assert np.sum(haloes_store == -1) == 0, "Logical error: {} elements of {} are still unassigned inside haloes_store. Track recorded {}"\
            .format(np.sum(haloes_store == -1), len(stellar_mass), track)

        return haloes_store
    else:
        hm = get_sm_fixedz(stellar_mass, z)
        return hm


if __name__ == "__main__":
    stellar_masses  = [11, 10.5, 12]
    z = [0, 1, 1.0]
    haloes = stellar_mass_to_halo_mass(stellar_masses, z)
