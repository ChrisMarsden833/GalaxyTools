import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from colossus.cosmology import cosmology
from scipy.stats.stats import zscore
cosmo = cosmology.setCosmology("planck18")
from colossus.halo import profile_nfw
from colossus.halo import concentration

import HM2SM
import SDSSExtractor
import darkmatter
from Utility import binnedMean

def GetDefaultParameters(Stellar_mass, z=0, halo_mass="Generate", hmsm = "Grylls19", retmass = False, mdef = 'auto', get_vcirc = False):
    
    if mdef == 'auto':
        if(hmsm == "Moster"):
            mdef = "200c"
        else:
            mdef = "vir"

    # Do input validation to check that halo mass is a string
    if isinstance(halo_mass, str):
        if halo_mass == "Generate":
            halo_mass = HM2SM.stellar_mass_to_halo_mass(Stellar_mass, z, formula=hmsm, mdef=mdef)
        else:
            assert False, "GetDefaultParameters - Unrecognised halo_mass string {}. Did you mean Generate?".format(halo_mass)
    
    # Assume that halo masses are (now) an array of values (or single value)
    if hasattr(halo_mass, "__len__") and hasattr(Stellar_mass, "__len__"):
        assert len(halo_mass) == len(Stellar_mass), "Stellar Mass (len {}) and halo mass (len {}) are different lengths".format(len(Stellar_mass), len(halo_mass))
    
    # Start to assign parameters
    
    # Dark Matter Halo
    
    conc = concentration.concentration((10**halo_mass)*cosmo.h, mdef, z, model = 'ishiyama20')
    nfw  = profile_nfw.NFWProfile(M = (10**halo_mass)*cosmo.h, c = conc, z = z, mdef = mdef) # Just for obj - these parameters do not matter (a quirk of colossus)
    rho, rs = nfw.fundamentalParameters( (10**halo_mass)*cosmo.h, conc, z, mdef)
    rs /= cosmo.h
    rho *= cosmo.h**2

    # Radius and Sersic
    radius = SDSSExtractor.MANGa_Sizes_Fit(Stellar_mass, z, incGamma=True)
    n = SDSSExtractor.MANGa_Sersic_Fit(Stellar_mass) * (1. + z)**-1
    
    if retmass:
        return radius, n, halo_mass, conc

    if get_vcirc:
        vcirc = nfw.Vmax()
        return radius, n, rs, rho, vcirc

    return radius, n, rs, rho


"""

if __name__ == "__main__":

    example_sm = np.array([11., 11.5, 9.5])
    print("Example_sm: {}".format(example_sm))

    radius, n, rs, rho = GetDefaultParameters(example_sm)
    print("Radius: {} \nSersicIndex: {} \nrs: {} \nrho: {}".format(radius, n, rs, rho))

    conc = concentration.concentration((10**halo_mass)*cosmo.h, mdef, z, model = 'ishiyama20')
    catalog_sm = halo_mass_to_stellar_mass(catalog, z_fixed, formula=formula) # With scatter
    width = 0.05
    sm_range = np.arange(5, 15, width)

    means = binned_statistic(catalog_sm, catalog, bins=sm_range)[0]

    sm_range = sm_range[:-1] + width/2

    safe_mask = ~np.isnan(means)

    snip_low = 20

    p0 = [0.5, 12, 0.52, 0.02, 1] # Initial Guess - this speeds up the process 'considerably'
    params = curve_fit(similar_function, means[safe_mask][snip_low:],
                           sm_range[safe_mask][snip_low:], p0 = p0)
    params = params[0]

    halo_domain = np.arange(6, 20, 0.1)
    sm_domain = similar_function(halo_domain, params[0], params[1], params[2], params[3], params[4])
    sm2hm = interp1d(sm_domain, halo_domain, bounds_error = False, fill_value="extrapolate")
    mass = sm2hm(sm)
    return mass


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

        assert np.sum(haloes_store == -1) == 0, "Logical error: {} elements of {} are still unassigned inside haloes_store. Track recorded {}".format(np.sum(haloes_store == -1), len(stellar_mass), track) nfw  = profile_nfw.NFWProfile(M = 1E12, c = 10.0, z = 0.0, mdef = mdef) # Just for obj - these parameters do not matter (a quirk of colossus)
        rho, rs = nfw.fundamentalParameters( (10**halo_mass)*cosmo.h, conc, z, mdef)fw  = profile_nfw.NFWProfile(M = 1E12, c = 10.0, z = 0.0, mdef = mdef) # Just for obj - these parameters do not matter (a quirk of colossus)
        rho, rs = nfw.fundamentalParameters( (10**halo_mass)*cosmo.h, conc, z, mdef)


        
"""
