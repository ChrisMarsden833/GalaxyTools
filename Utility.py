import numpy as np
import scipy.stats as stats

def binnedMean(bins, data2bin,  data2mean, vmax):
    """ Function to do binned means (and errors) with vmax weighting.

    Saves a lot of time
    """
    assert len(data2bin) == len(data2mean) ==  len(vmax),\
        "The length of supplied data(s) (data2bin {}, data2mean {}, vmax {}) are not consistent".format(len(data2bin), len(data2mean), len(vmax))

    # Convert to numpy arrays
    bins = np.array(bins)
    data2bin = np.array(data2bin)
    data2mean = np.array(data2mean)
    vmax = np.array(vmax)

    # Remove values that are outside the range
    cut_mask = (data2bin <= np.amax(bins)) & (data2bin >= np.amin(bins))
    data2bin = data2bin[cut_mask]
    data2mean = data2mean[cut_mask]
    vmax = vmax[cut_mask]

    assert (len(data2bin) != 0) and (len(data2mean) != 0) and (len(vmax) != 0), "Length zero arrays - bins are probably not in range"

    # average
    array, edges, numbers = stats.binned_statistic(data2bin,  vmax * data2mean, statistic='sum', bins = bins)

    den = stats.binned_statistic(data2bin, vmax, statistic = 'sum', bins = bins)[0]
    nonzero_main = array != 0
    array[nonzero_main] = array[nonzero_main]/den[nonzero_main]

    # errors
    means = np.zeros(len(data2bin)) # Storage array for the means - the value of the mean at each element
    # numbers is an array holding the index of the bin at which each value of  data belongs
    # This loop simply finds the mean value within the bin WRT to the data point 

    for i in range(len(means)):
        means[i] = array[numbers[i]-1] # There probably is a faster way to do this

    residual = data2mean - means


    # positive error
    std_wrong_pos = stats.binned_statistic(data2bin[residual >= 0],  vmax[residual >= 0]*(data2mean[residual >= 0] - means[residual >= 0])**2,
            statistic = 'sum', bins = bins)[0]
    binCounts_pos = stats.binned_statistic(data2bin[residual >= 0], means[residual >= 0], statistic = 'count', bins = bins)[0]

    den_pos = stats.binned_statistic(data2bin[residual >= 0], vmax[residual >= 0], statistic = 'sum', bins = bins)[0]
    nonzero = (std_wrong_pos != 0) & (den_pos != 0) & (binCounts_pos > 1) # To prevent /0 errors
    dev_pos = np.zeros_like(array)
    dev_pos[nonzero] = np.sqrt(std_wrong_pos[nonzero]/(((binCounts_pos[nonzero]-1) /binCounts_pos[nonzero])*den_pos[nonzero]))

    # Negative error
    std_wrong_neg = stats.binned_statistic(data2bin[residual <= 0],  vmax[residual <= 0]*(data2mean[residual <= 0] - means[residual <= 0])**2,
            statistic = 'sum', bins = bins)[0]
    binCounts_neg = stats.binned_statistic(data2bin[residual <= 0], means[residual <= 0], statistic = 'count', bins = bins)[0]

    den_neg = stats.binned_statistic(data2bin[residual <= 0], vmax[residual <= 0], statistic = 'sum', bins = bins)[0]
    nonzero = (std_wrong_neg != 0) & (den_neg != 0) & (binCounts_neg > 1) # To prevent /0 errors
    dev_neg = np.zeros_like(array)
    dev_neg[nonzero] = np.sqrt(std_wrong_neg[nonzero]/(((binCounts_neg[nonzero]-1) /binCounts_neg[nonzero])*den_neg[nonzero]))

    return array, dev_pos, dev_neg

