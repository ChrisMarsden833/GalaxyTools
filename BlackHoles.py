import numpy as np

def stellar_mass_to_black_hole_mass(stellar_mass, method="Shankar16", scatter="Intrinsic"):
    """ Function to assign black hole mass from the stellar mass.
    :param stellar_mass: array, of stellar masses [log10 M_sun]
    :param method: string, specifying the method to be used, current options are "Shankar16",  "KormondyHo" and "Eq4".
    :param scatter: string or float, string should be "Intrinsic", float value specifies the (fixed) scatter magnitude
    :return: array, of the black hole masses [log10 M_sun].
    """

    # Main values
    if method == "Shankar16":
        log_black_hole_mass = 7.574 + 1.946 * (stellar_mass - 11) - 0.306 * (stellar_mass - 11)**2. \
                              - 0.011 * (stellar_mass - 11)**3
    elif method == "KormondyHo":
        log_black_hole_mass = 8.54 + 1.18 * (stellar_mass - 11)
    elif method == 'Eq4':
        log_black_hole_mass = 8.35 + 1.31 * (stellar_mass - 11)

    # Scatter
    if scatter == "Intrinsic" or scatter == "intrinsic":
        if method == "Shankar16":
            # The intrinsic formula for scatter given in FS's relation
            scatter = (0.32 - 0.1 * (stellar_mass - 12.)) * np.random.normal(0., 1., len(stellar_mass))
        if method == "KormondyHo":
            scatter = np.random.normal(0, 0.5, len(stellar_mass))
        if method == "Eq4":
            scatter = np.random.normal(0, 0.5, len(stellar_mass))
    elif isinstance(type(scatter), float):
        scatter = np.random.normal(0., scatter, len(stellar_mass))
    elif isinstance(type(scatter), list) or isinstance(type(scatter), np.ndarray):
        scatter = np.random.normal(0, 1, len(stellar_mass)) * scatter
    elif scatter == False or scatter == None:  # Not sketchy, as we actually want to compare this with a boolean,
        # ...despite what my IDE thinks.
        scatter = np.zeros_like(stellar_mass)
    else:
        raise ValueError("Unknown Scatter argument {}".format(scatter))

    log_black_hole_mass += scatter
    return log_black_hole_mass
