from os import error
import numpy as np
from numpy.core.fromnumeric import cumsum
from numpy.lib.function_base import interp
import scipy.stats as stats
from scipy.special import gamma, gammainc, gammaincc
from scipy.interpolate import interp1d
from scipy import integrate
import pandas as pd
import sys
sys.path.insert(0, "/Users/chris/Documents/ProjectSigma/VelocityDispersion")
from SigmaLib import Sigma

def binnedMean(bins, data2bin,  data2mean, vmax, error_on_mean = False):
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

    if error_on_mean:
        dev_pos[nonzero] /= np.sqrt(binCounts_pos[nonzero])

    # Negative error
    std_wrong_neg = stats.binned_statistic(data2bin[residual <= 0],  vmax[residual <= 0]*(data2mean[residual <= 0] - means[residual <= 0])**2,
            statistic = 'sum', bins = bins)[0]
    binCounts_neg = stats.binned_statistic(data2bin[residual <= 0], means[residual <= 0], statistic = 'count', bins = bins)[0]

    den_neg = stats.binned_statistic(data2bin[residual <= 0], vmax[residual <= 0], statistic = 'sum', bins = bins)[0]
    nonzero = (std_wrong_neg != 0) & (den_neg != 0) & (binCounts_neg > 1) # To prevent /0 errors
    dev_neg = np.zeros_like(array)
    dev_neg[nonzero] = np.sqrt(std_wrong_neg[nonzero]/(((binCounts_neg[nonzero]-1) /binCounts_neg[nonzero])*den_neg[nonzero]))

    if error_on_mean:
        dev_neg[nonzero] /= np.sqrt(binCounts_neg[nonzero])

    return array, dev_pos, dev_neg

from scipy.special import erf
def anisotropy(mstar):
    low = 0,
    high = 0.4
    return 0.4/2 * (erf( (mstar - 10.5)/0.5 ) + 1)

def gamma_lower(a, z):
    """Lower gamma function, used in sersic profiles"""
    top = gammainc(a, z)
    cancel = gamma(a)
    return top * cancel
    
def tgamma(a, z):
    """gamma function used in sersic profiles"""
    top = gammaincc(a, z)
    cancel = gamma(a)
    return top * cancel

def sersicMassWithin(r, bulge_mass, re, n):
    """Get stellar mass within r based on a sersic profile"""
    b_n = 2.*n - (1./3.) + (.009876/n)
    p_n = 1. - .6097/n  + .00563/(n*n)
    a_s = re/(b_n**n)
    x = r/a_s
    threemp_term = (3.-p_n) * n
    res = 10.**bulge_mass 
    gamma1 = gamma_lower(threemp_term, x**(1./n))
    gamma2 = tgamma(threemp_term, 0.0)
    res *= gamma1/gamma2
    return res
    
def b_n(n):
    return 2.*n - (1./3.) + (.009876/n)

def Ie(mstar, re, n):
    return mstar / (2. * np.pi * re**2 * n * np.exp(b_n(n)) * gamma(2.*n) / b_n(n)**(2.*n))

def sersicProfile(r, mstar, re, n):
    return Ie(mstar, re, n) * np.exp(-b_n(n) * ( (r/re)**(1./n) - 1 ))
                             # np.exp(-b_n * ( (r/re)**(1./n) - 1 ))

def PrugDensity(r, mstar, re, n, ML = 1.0):
    p_n = 1. - .6097/n  + .00563/(n*n)
    rho0 = ( Ie(mstar, re, n) * ML * np.exp(b_n(n)) * b_n(n)**(n*(1-p_n)) * (gamma(2.*n)/(2.*re*gamma(n * (3 - p_n))))) 
    return rho0 * (r/re)**(-p_n) * np.exp(-b_n(n) * (r/re)**(1/n))

def NFW_massWithin(Rs, rho, Re, profile = "NFW"):
    print("Profile = {}".format(profile))
    if profile == "NFW":
        """Mass within Re of halo with NFW profile"""
        x = Re/Rs
        res = 4 * np.pi * Rs**3 * rho * (np.log(1 + x) - x/(1+x))
        return res
    elif profile == "BUR":
        x = Re/Rs
        tanterm = -2. * np.arctan(x)
        term2 = 2.* np.log(1+x)
        term3 = np.log(1 + x*x)
        res = np.pi * rho * Rs**3 * (tanterm + term2 + term3)
        return res
    else:
        assert False, "Profile {} not recognised".format(profile)

def disk_massWithin(R, disk_mass, h):
    mfactor = (1 - np.exp(-R/h)*(1+R/h))
    sigma_0 = mfactor * (10**disk_mass)/(2. * np.pi * h**2)
    res = 2 * np.pi * sigma_0 * h**2 * mfactor
    return res

def get_values(Mr, log_sig, ellipticals = True):
    if ellipticals:
        if  -22.5 > Mr:
            if log_sig > 2.4:
                R0, R8, R1 = 8, 4, 3.5
            else:
                R0, R8, R1 = 7, 3, 3
        elif -21.5 > Mr > -22.5:
            if log_sig > 2.3:
                R0, R8, R1 = 5, 4, 3
            else:
                R0, R8, R1 = 5, 3, 2.5
        elif Mr > -21.5:
            if log_sig > 2.2:
                R0, R8, R1 = 5, 4, 3
            else:
                R0, R8, R1 = 3, 2.5, 2
        else:
            assert False, "Unknown value of Mr ({})".format(Mr)
    else:
        if -21.5 > Mr:
            if log_sig > 2.3:
                R0, R8, R1 = 6.5, 4.5, 3.0
            elif 2.2 < log_sig <= 2.3:
                R0, R8, R1 = 5, 2.5, 2.0
            else:
                R0, R8, R1 = 2., 1., 1.
        elif -20.5 > Mr >= -21.5:
            if log_sig > 2.2:
                R0, R8, R1 = 5.5, 4., 3.
            elif  2.1 < log_sig < 2.2:
                R0, R8, R1 = 3., 1.5, 1.5
            else:
                R0, R8, R1 = 2., 1.5, 1.5
        elif Mr > -20.5:
            if log_sig > 2.1:
                R0, R8, R1 = 4., 3.5, 3.
            elif 2 < log_sig < 2.1:
                R0, R8, R1 = 4., 3., 3.
            else:
                R0, R8, R1 = 1.5, 1.5, 1.5
        else:
            assert False, "Unknown value of Mr ({})".format(Mr)

    return R0, R8, R1

class IMF_Galaxy:
    def __init__(self, Lum, Re, n, R0, R1):
        self.Lum = Lum
        self.Re = Re
        self.n = n
        self.R0 = R0
        self.R1 = R1

    def alpha(self, Y):
        if Y >= 1:
            return 0.    
        else:
            alpha = (self.R0 - self.R1)/self.R1
        return alpha

    def ML(self, r):
        Y = r/self.Re
        alpha = self.alpha(Y)
        if hasattr(r, "__len__"):
            res = np.zeros_like(r)
            mask = Y < 1.0
            res[mask] = self.R1 * (1. + alpha - alpha * Y)
            res[Y >= 1] = self.R1
            return res
        else:
            if Y < 1:
                return self.R1 * (1. + alpha - alpha * Y)
            else:
                return self.R1

    def rhoX_integrand_simple(self, Y, y):
        alpha = self.alpha(Y)
        res  = ( 1/(Y * np.sqrt(Y**2 - y**2)) ) * np.exp(-b_n(self.n) * ( (Y)**(1./self.n) - 1 )) \
            * ( (self.n * alpha * Y) + b_n(self.n) * (alpha - alpha * Y) * Y**(1./self.n))
        return res

    def rhoX_simple(self, r):
        y = r/self.Re
        Ymax = 1.0

        if hasattr(r, "__len__"):
            res = np.zeros_like(r)
            for i, yi in enumerate(y):
                if yi < 1:
                    res[i] = integrate.quad(self.rhoX_integrand_simple, yi, Ymax, args = (y), full_output = 1)[0]
        else:
            if y < 1:
                res = integrate.quad(self.rhoX_integrand_simple, y, Ymax, args = (y), full_output = 1)[0]
            else:
                res = 0.

        I0 = Ie(self.Lum, self.Re, self.n)
        res *= (self.R1 * I0) / (np.pi * self.Re * self.n)
        return res

    def rho_IMF_simple(self, r):
    
        basic_density = PrugDensity(r, self.Lum, self.Re, self.n, ML=self.R1)
        extra_density = self.rhoX_simple(r)

        return basic_density + extra_density
        
    def projected_density_IMF_simple(self, r):
        return self.ML(r) * sersicProfile(r, self.Lum, self.Re, self.n)

    def get_total_mass(self):

        def proj_integrand(r):
            return 2. * np.pi * r * self.projected_density_IMF_simple(r)

        res = integrate.quad(proj_integrand, 0, 10.*self.Re, full_output = 1)[0]
        return res

    def get_mass_within_deproj(self, r):
        def deprojected_integrand(r):
            return 4. * np.pi * r**2 * self.rho_IMF_simple(r)
        res = integrate.quad(deprojected_integrand, 0, r, full_output = 1)[0]
        return res
    
"""

def ML(r, Mr, log_sig, Re, gradient = False):

    R0, R8, R1 = get_values(Mr, log_sig)
    ratio = r/Re

    grad1 = (R8 - R0) / (0.8)
    intercept1 = R0
    grad2 = (R1-R8) / (0.2)
    intercept2 = R8 - grad2 * (0.8)

    if hasattr(r, "__len__"):
        if gradient:
            res = np.zeros_like(r)
            mask = ratio < 0.8
            res[mask] = grad1
            mask = (ratio >= 0.8) & (ratio < 1.0) 
            res[mask] = grad2
            res[ratio >= 1] = 0.0
            return res
        else:
            res = np.zeros_like(r)
            # First interval
            mask = ratio < 0.8
            res[mask] = grad1 * ratio[mask] + intercept1
            mask = (ratio >= 0.8) & (ratio < 1.0) 
            res[mask] = grad2 * ratio[mask] + intercept2
            res[ratio >= 1] = R1
            return res
    else:
        if ratio < 0.8:
            grad= (R8 - R0) / (0.8)
            if gradient:
                return grad
            intercept = R0
            return grad * ratio + intercept
        elif ratio < 1:
            grad = (R1-R8) / (0.2)
            if gradient:
                return grad
            intercept = R8 - grad * (0.8)
            return grad * ratio + intercept
        else:
            if gradient:
                return 0.0
            return R1

def Jprime(R, L, re, n, Mr, sig):
    # Chain rule
    return sersicProfileL(R, L, re, n) * ML(R, Mr, sig, re, gradient = True) + \
        sersicProfileL(R, L, re, n, gradient = True)  + ML(R, Mr, sig, re)

def integrand(R, r, observerL, re, n):
    return sersicProfileL(R, observerL, re, n, gradient=True)/(np.sqrt(R**2 - r**2))

def integrandIMF(R, r, L, Mr, sig, re, n):
    Jp = Jprime(R, L, re, n, Mr, sig)
    den = (np.sqrt(R**2 - r**2))
    res = Jp/den
    return res

def rho_IMF(r_domain, L, re, n, Mr, log_sig):
    rho_IMF = np.zeros_like(r_domain)
    for i, r in enumerate(r_domain):

        if r < re:
            res = integrate.quad(integrandIMF, r, np.inf, args = (r, L, Mr, log_sig, re, n), full_output = 1 )
            rho_IMF[i] = - (1./np.pi) * res[0]
        else:
            R0, R8, R1 = get_values(Mr, log_sig)
            rho_IMF[i] = PrugDensityM(r, np.log10(R1 * L), re, n)

    return rho_IMF

def rho_IMF_num(r_domain, L, re, n, Mr, log_sig):
    rho_IMF = np.zeros_like(r_domain)
    for i, r in enumerate(r_domain):
        integral_domain, step = np.linspace(r+0.001, 500, 10000, retstep=True)

        integrand = integrandIMF(integral_domain, r, L, Mr, log_sig, re, n)

        integral = np.sum(integrand * step)
        rho_IMF[i] = - (1./np.pi) * integral

    return rho_IMF

def projected_mass_integrand(r, L, re, n, Mr, log_sig):
    return 2.*np.pi*r * sersicProfileL(r, L, re, n) * ML(r, Mr, log_sig, re)



def total_mass_IMF(L, re, n, Mr, log_sig):
    r_domain, spacing = np.linspace(0.001, 20*re, 1000, retstep=True)
    density = rho_IMF(r_domain, L, re, n, Mr, log_sig)
    mass = np.sum( density * 4*np.pi*r_domain**2 * spacing )
    return mass

def mass_within_IMF(r, L, re, n, Mr, log_sig):
    r_domain, spacing = np.linspace(0.001, r, 100, retstep=True)
    density = rho_IMF(r_domain, L, re, n, Mr, log_sig)
    mass = np.sum( density * 4*np.pi*r_domain**2 * spacing )
    return mass
""" 

def mstar2LumandMag(mstar):
    df = pd.read_csv("/Users/chris/data/MANGa/MANGa.csv")
    step = 0.1
    mstar_bins = np.arange(8, 13, step)
    res, _, _ = binnedMean(mstar_bins, df["IMF_mass"],  df['Luminosity_r'], df['Vmax'] )

    mask = res > 0
    mstar2Luminosity = interp1d( (mstar_bins[:-1]+step/2)[mask], res[mask], bounds_error=False, 
        fill_value=(res[mask][0], res[mask][-1]) )
    
    res2, _, _ = binnedMean(mstar_bins, df["IMF_mass"],  df['mr'], df['Vmax'] )
    mask2 = res2 < 0
    mstar2mag = interp1d( (mstar_bins[:-1]+step/2)[mask2], res2[mask2], bounds_error=False, 
        fill_value=(res2[mask2][0], res2[mask2][-1]) )

    log_lum = 2.5924279257970944 + 0.7236937315796155 * mstar

    return 10**log_lum, mstar2mag(mstar)

def lum2mstar(lum):
    loglum = np.log10(lum)
    mstar = (loglum - 2.5924279257970944)/0.7236937315796155
    return mstar


def lum2R1R0(lum):
    loglum = np.log10(lum)

    fixed = 1.5
    stop = 9.5

    ### R1 ###
    gradient = 0.2 #0.9
    intercept = fixed - gradient * stop
    R1 = gradient * loglum + intercept

    #R1[loglum > 11] += 0.1*loglum[loglum > 11]

    ### R0 ###
    gradient = 0.5 #3.2
    intercept = fixed - gradient * stop
    R0 = gradient * loglum + intercept

    #R0[loglum > 11] += 0.1*loglum[loglum > 11]

    

    R1[loglum < stop] = fixed
    R0[loglum < stop] = fixed

    return R1, R0


