import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm
import os.path

sys.path.insert(0, "/Users/chris/Documents/ProjectSigma/DREAM/dream/")
from semi_analytic_catalog import generate_parents_catalogue
from halo_growth import Mass_acc_history_VDB_FS
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
    elif formula == "Uluk":
        z_parameter = np.divide(z - 0.1, z + 1)
        m_10, shm_norm_10, beta10, gamma10 = 11.72, 0.032, 1.9, 0.49
        m_11, shm_norm_11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        if scatter != 0.0:
            scatter = 0.1
    elif formula == "GryllsNoz":
        z_parameter = 1.0
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

    def get_sm_fixedz(sm, z_fixed):

        width = 0.05
        sm_range = np.arange(5, 15, width)


        file = "/Users/Chris/Documents/ProjectSigma/GalaxyTools/data/Catalog_{}_{}.npy".format(np.round(z_fixed, 3), mdef)
        means_file = "/Users/Chris/Documents/ProjectSigma/GalaxyTools/data/means_{}_{}.npy".format(np.round(z_fixed, 3), formula)

        if not os.path.isfile(means_file):
            print("Means does not exist, generating")

            if os.path.isfile(file):
                print("Catalog File Exists")
                catalog = np.load(file)
            else:
                print("Catalog File does not exist")
                z_domain = np.arange(0, 10, 0.1)

                #high_track = Mass_acc_history_VDB_FS(16., z_domain, cosmo.h, cosmo.Om0)
                #z2high = interp1d(z_domain, high_track)
                #mhigh = np.round(z2high(z_fixed), 1)

                #low_track = Mass_acc_history_VDB_FS(8, z_domain, cosmo.h, cosmo.Om0)
                #z2low = interp1d(z_domain, low_track)
                #mlow = np.round(z2low(z_fixed), 1)

                catalog = generate_parents_catalogue("tinker08", 500, (11, 16, 0.1), z_fixed, cosmo.h, mdef="200c")
                #np.save(file, catalog)

            catalog_sm = halo_mass_to_stellar_mass(catalog, z_fixed, formula=formula) # With scatter
            means = binned_statistic(catalog_sm, catalog, bins=sm_range)[0]

            np.save(means_file, means)

        else:
            means = np.load(means_file)

        sm_range = sm_range[:-1] + width/2

        safe_mask = ~np.isnan(means)

        """
        def similar_function(m, n, mn, beta, gamma, delta):
            res = 2. * n * m * ((10**(m-mn))**-beta + (10**(m-mn))**gamma)**-delta
            return np.log10(res)

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
            print(mass)
            return mass
        except Exception as e:
            print(e) # You get into trouble otherwise"""

        sm2hm = interp1d(sm_range, means, bounds_error=False, fill_value='extrapolate')

        res = sm2hm(sm)

        res[res > 15.5] = 15.5

        return res


    stellar_mass = np.array(stellar_mass)
    if hasattr(z, "__len__"):
        assert len(z) == len(stellar_mass), "Length of sm ({}) != length z ({})".format(len(haloes), len(z))

        binwidth = 0.05
        z_bins = np.arange(0, 10, binwidth)
        indexes = np.digitize(z, z_bins)-1

        haloes_store = np.ones_like(stellar_mass) * -1
        track = 0
        unique_steps = np.unique(indexes)

        for index in unique_steps:
            redshift = z_bins[index]
            sm = stellar_mass[indexes == index]
            hm = get_sm_fixedz(sm, redshift)
            haloes_store[indexes == index] = hm

        return haloes_store
    else:
        binwidth = 0.05
        z_bins = np.arange(0, 10, binwidth)
        index = np.digitize(z, z_bins)-1

        hm = get_sm_fixedz(stellar_mass, z_bins[index])
        return hm


if __name__ == "__main__":
    cosmo = cosmology.setCosmology("planck18")
    
    stellar_masses  = [11.5]
    z = 0.0
    haloes = stellar_mass_to_halo_mass(stellar_masses, z)

    print(haloes)
