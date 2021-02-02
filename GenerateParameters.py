import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology("planck18")
from colossus.halo import profile_nfw
from colossus.halo import concentration

import HM2SM
import SDSSExtractor
import darkmatter
from Utility import binnedMean

def GetDefaultParameters(Stellar_mass, z=0, halo_mass="Generate", hmsm = "Moster"):

    if(hmsm == "Moster"):
        mdef = "200c"
    elif(hmsm == "Grylls19"):
        mdef = "mvir"
    else:
        assert False, "GetDefaultParameters - Unrecognised hmsm parameter {}".format(hmsm)

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
    nfw  = profile_nfw.NFWProfile(M = 1E12, c = 10.0, z = 0.0, mdef = mdef) # Just for obj - these parameters do not matter (a quirk of colossus)
    rho, rs = nfw.fundamentalParameters( (10**halo_mass)*cosmo.h, conc, z, mdef)
    rs /= cosmo.h
    rho *= cosmo.h**2

    # Radius and Sersic
    radius = SDSSExtractor.SDSS_Sizes_Fit(Stellar_mass, z)
    n = SDSSExtractor.SDSS_Sersic_Fit(Stellar_mass) * (1. + z)**-1
    
    return radius, n, rs, rho

if __name__ == "__main__":

    example_sm = np.array([11., 11.5, 9.5])
    print("Example_sm: {}".format(example_sm))

    radius, n, rs, rho = GetDefaultParameters(example_sm)
    print("Radius: {} \nSersicIndex: {} \nrs: {} \nrho: {}".format(radius, n, rs, rho))




        

