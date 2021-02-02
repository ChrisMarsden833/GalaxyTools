import numpy as np
from BlackHoles import *
import matplotlib.pyplot as plt

stellar_mass_range = np.arange(10, 12.0, 0.5)

plt.figure(dpi=130)

for stellar_mass in stellar_mass_range:

    ratio_range = np.linspace(0.01, 1, 20)
    sats = np.log10(ratio_range * 10**stellar_mass)

    central_black_hole = stellar_mass_to_black_hole_mass(np.array([stellar_mass]), scatter=0.)
    sat_black_holes = stellar_mass_to_black_hole_mass(sats, scatter=0.)
    post_black_holes = np.log10(10**central_black_hole + 10**sat_black_holes)

    def K(BH):
        return 10**((BH-7.8)/5.7)

    ratio = K(post_black_holes)/K(central_black_hole)

    plt.plot(ratio_range, ratio, label="{}".format(stellar_mass))

plt.xlim([0, 1])
plt.legend()
plt.xlabel(r"$\Delta M_*/M_*$")
plt.ylabel(r"$\sigma_{new}/\sigma_{central}$")
plt.savefig("/Users/chris/Desktop/test.png")

