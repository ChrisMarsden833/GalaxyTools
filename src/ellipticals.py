import numpy as np
from scipy.special import gamma, gammaincc

def sersic_profile(R, stellar_mass, hlr, n):
    b_n = 2*n - (1/3) - (0.09876/n)
    gam = gamma(2*n)*gammaincc(2*n, b_n)
    sigma_e = 10**stellar_mass / (hlr**2 * np.pi*2*2*n*np.exp(b_n)*gam/b_n**(2*n))
    #sigma_e = (10**stellar_mass)/(2*np.pi * hlr**2)
    internal_term = R/hlr
    return sigma_e * np.exp(-b_n * (internal_term**(1/n) - 1))

def de_projected_sersic(r, stellar_mass, hlr, n):

    p_n = 1. - .6097/n  + .00563/(n*n)
    b_n = 2*n - (1/3) - (0.09876/n)

    Sigma0 = sersic_profile(0, stellar_mass, hlr, n)
    left = (Sigma0 * b_n**(n * (1-p_n))) / (2*hlr)
    right = gamma(2*n)/gamma(n*(3. - p_n))
    rho0 = left * right;

    ratio = r/hlr
    power_term = ratio**-p_n
    inverse_n = 1/n
    exp_power_term = ratio**inverse_n
    exp_term = np.exp(-b_n * exp_power_term)
    res = rho0 * power_term * exp_term

    return res


if __name__ == "__main__":
    print(de_projected_sersic(5, 12, 6, 4))
