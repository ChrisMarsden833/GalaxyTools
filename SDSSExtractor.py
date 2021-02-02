import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt


def get_data(path="~/data/SDSS/Catalog_SDSS_complete.dat"):
    """ Internal function to read in (and filter) the SDSS catalog
    :params path: string, the path to the SDSS file.
    :returns StellarMass, SersicIndex, Size, VMax: numpy arrays of approprate
    propertues.
    """
    df = pd.read_csv(path, sep=' ')  # Read in the file

    # Grab useful fields
    SersicIndex = np.array(df['n_bulge'])
    StellarMass = np.array(df['MsMendSerExp'])
    VMax = np.array(df["Vmaxwt"])
    Re = np.array(df["logReSerExpCircNotTrunc"])

    # Create flags to remove offending galaxies
    flag_central = df["NewLCentSat"] == 1. # Only centrals
    flag_hasMass = StellarMass != -999. # Only valid stellar masses
    flag_is8 = SersicIndex != 8. # Cap on Sersic Indexes
    flag_is0 = SersicIndex != 0.1 # lower bound on sersic indexes
    flag_morph = np.array(df["TType"]) <= 0. # Ellipticals
    flag_combined = flag_hasMass & flag_is8 & flag_is0 & flag_central & flag_morph

    StellarMass = StellarMass[flag_combined]
    SersicIndex = SersicIndex[flag_combined]
    Size = 10**Re[flag_combined]
    VMax = VMax[flag_combined]

    return StellarMass, SersicIndex, Size, VMax


def Assign_Size(input_masses, scatter = False, mag = 1):
    """ Function to assign sizes based on empirical relations derived from the
    SDSS. Only valid for low redshifts.
    :param input_masses: numpy array of Stellar Masses [log10 M_sun]
    :param scatter: bool, if scatter should be included or not
    :param mag: float, multiplicative factor to modify the scatter (default is
    1, so no change). Often the large scatter can cause problems so reducing it
    can be useful.
    returns: numpy array of galaxy sizes [kpc]
    """
    StellarMass, SersicIndex, Size, VMax = get_data()

    # Bin by sm
    width = 0.05
    bins = np.arange(0., 30.0, width)

    # Weighted average by VMax
    array, edges, numbers = stats.binned_statistic(StellarMass, VMax*Size, statistic = 'sum', bins = bins)
    den = stats.binned_statistic(StellarMass, VMax, statistic = 'sum', bins = bins)[0]

    array[array != 0] = array[array != 0]/den[array != 0] # Calculate the numerator and denominator separately, to make the best use of binned statistic


    # For the purposes of the error, we need to do some housekeeping here
    means = np.zeros(len(numbers)) # Storage array for the means - the value of the mean at each element
    for i in range(len(means)):
        means[i] = array[numbers[i]-1] # There probably is a faster way to do this
    # Calculate the components needed for the standard deviation
    std_wrong = stats.binned_statistic(StellarMass, VMax*(Size - means)**2, statistic = 'sum', bins = bins)[0]
    binCounts = stats.binned_statistic(StellarMass, means, statistic = 'count', bins = bins)[0]

    # Calculate the standard deviation
    nonzero = binCounts != 0

    den = (((binCounts[nonzero]-1)/binCounts[nonzero])*den[nonzero])
    valid_den = den != 0
    dev = np.sqrt(std_wrong[nonzero][valid_den]/den[valid_den])

    get_Size = interpolate.interp1d((bins[0:-1]+width/2)[array != 0],
                                    array[array != 0], bounds_error=False,
                                    fill_value=(np.amin(array[array != 0]), np.amax(array[array != 0])))


    get_size_Error = interpolate.interp1d((bins[0:-1]+width/2)[nonzero][valid_den], dev,
                                         bounds_error=False, fill_value="extrapolate")

    result = get_Size(input_masses)

    # Set all values that are negative (or zero) to the minimum positive result
    min_pos = np.amin(result[result > 0])
    result[result <= 0] = min_pos

    if scatter:
        magnitude = get_size_Error(input_masses)
        magnitude[magnitude < 0] = 0
        scatter = np.random.normal(loc = 0, scale = mag*magnitude)
        result_temp = result + scatter
        result_temp[result_temp <= 0] = result[result_temp <= 0]
        result = result_temp

    return result

def AssignSersicIndex(input_masses, scatter=False, mag=1):
    """ Function to assign Sersic Indexes based on empirical relations derived from the
    SDSS. Only valid  at low redshifts.
    :param input_masses: numpy array of Stellar Masses [log10 M_sun]
    :param scatter: bool, if scatter should be included or not
    :param mag: float, multiplicative factor to modify the scatter (default is
    1, so no change). Often the large scatter can cause problems so reducing it
    can be useful.
    returns: numpy array of galaxy sersic indexes [dimensionless]
    """
    StellarMass, SersicIndex, Size, VMax = get_data()

    # Bin up the SDSS
    bins = np.arange(0.0, 30.0, 0.05)
    # Weighted average by VMax
    array, edges, numbers = stats.binned_statistic(StellarMass, VMax*SersicIndex, statistic = 'sum', bins = bins)
    den = stats.binned_statistic(StellarMass, VMax, statistic = 'sum', bins = bins)[0]
    array[array != 0] = array[array != 0]/den[array != 0] # Calculate the numerator and denominator separately, to make the best use of binned statistic


    # For the purposes of the error, we need to do some housekeeping here
    means = np.zeros(len(numbers)) # Storage array for the means - the value of the mean at each element
    for i in range(len(means)):
        means[i] = array[numbers[i]-1] # There probably is a faster way to do this
    # Calculate the components needed for the standard deviation
    std_wrong = stats.binned_statistic(StellarMass, VMax*(SersicIndex - means)**2, statistic = 'sum', bins = bins)[0]
    binCounts = stats.binned_statistic(StellarMass, means, statistic = 'count', bins = bins)[0]
    # Calculate the standard deviation
    nonzero = binCounts != 0

    den = (((binCounts[nonzero]-1)/binCounts[nonzero])*den[nonzero])
    valid_den = den != 0

    dev = np.sqrt(std_wrong[nonzero][valid_den]/den[valid_den])

    get_SersicIndex = interpolate.interp1d(bins[0:-1][array != 0],
                                           array[array!= 0],
                                           bounds_error=False,
                                           fill_value="extrapolate")

    get_Error = interpolate.interp1d(bins[0:-1][nonzero][valid_den], dev,
                                    bounds_error=False, fill_value="extrapolate")

    result = get_SersicIndex(input_masses)

    if scatter:
        magnitude = get_Error(input_masses)
        scatter = np.random.normal(loc=0, scale=mag*magnitude)
        result_temp = result + scatter
        result_temp[result_temp <= 2] = result[result_temp <= 2]
        result = result_temp

    return result


def SDSS_Sizes_Fit(sm, z=0, incGamma = "Marsden"):

    args = [3.04965733e-03, 9.67029240e-02, -3.91008421e+01, 2.04401878e-01, -4.29464785e+00]
    def gammaFunc(sm, a1, a2, a3, a4, a5):
        return a1 * sm * ((sm * a2) ** a3 + (sm * a4) ** a5) ** -1

    smp = 10**(sm)
    res = (10**-0.314) * (smp**0.042) * (1 + smp/(10**10.537))**0.76
    # This is now just a modified version of the RN/Mosleh fit...

    isarray_sm = True if hasattr(sm, "__len__") else  False
    isarray_z = True if hasattr(z, "__len__")  else  False

    if isarray_sm and isarray_z:
        assert len(sm) == len(z), "sm and z are unequal lengths: {} and {} respectively".format(len(sm), len(z))


    if incGamma == "Marsden" or bool(incGamma) == True:
        gamma = gammaFunc(sm, *args)
    elif incGamma == "RN":
        gamma = (1/0.85) * (sm - 10.75)
        gamma[gamma < 0] = 0
    else:
        assert False, "Unregonised type for incGamma {}. Value is: {}".format(str(type(incGamma)), incGamma)

    res = res * (1.+z)**-gamma

    return res


def SDSS_Sersic_Fit(sm, minsm = 8, natmin = 1.6):
    res = 10**(-0.01072867 * sm**3 +\
                0.32824192 * sm**2 +\
               -3.18145729 * sm +\
               10.14652827 )
    isarray_sm = True if hasattr(sm, "__len__") else  False

    if isarray_sm:
        res[sm < minsm] = natmin
    else:
        if sm < minsm:
            res = natmin
    return res

if __name__ == "__main__":
    stellar_masses = np.linspace(9, 12)

    plt.figure()
    plt.plot(stellar_masses, Assign_Size(stellar_masses))
    plt.plot(stellar_masses, Assign_Size(stellar_masses, scatter=True))
    plt.show()
