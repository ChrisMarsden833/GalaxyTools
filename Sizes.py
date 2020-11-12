import numpy as np
import scipy.interpolate as spi


def sizes_ric_nat(sm):
    return (10**-0.314) * ((10*sm)*0.042) * (1 + (10**sm)/(10**10.537))**0.76

def sizes_FS(sm):
    return 0.1 * ((10**sm)**0.14) * ((1 + (10**sm)/(3.98e10))) ** (0.39-0.14)

def ReRN(sm, z = 0):
    smp = 10**sm
    Re = (10**-0.314) * (smp**0.042) * (1 + smp/(10**10.537))**0.76
    
    gamma = (1/0.85) * (sm - 10.75)
    gamma[gamma < 0] = 0
    Re *= (1+z)**-gamma
    return Re                                    
                


def sizes_sersic(sm):

    data = np.array([[10.928755783198959, 0.5926430517711172],
                [11.010819276530802, 0.625340599455041],
                [11.098763853708041, 0.6471389645776566],
                [11.172029693080939, 0.6798365122615804],
                [11.245263570041633, 0.73433242506812],
                [11.318537400017581, 0.7615803814713897],
                [11.39615811805317, 0.8242506811989101],
                [11.470878247181316, 0.8651226158038148],
                [11.544092147634384, 0.9332425068119892],
                [11.617334015198129, 0.9822888283378748],
                [11.696421008893543, 1.0449591280653951],
                [11.775527979096584, 1.094005449591281],
                [11.834075127649886, 1.164850136239782],
                [11.904356476783304, 1.2329700272479565],
                [11.970219022429625, 1.3147138964577656],
                [12.047823759259114, 1.388283378746594],
                [12.109327430941214, 1.4427792915531334],
                [12.264548890504768, 1.5817438692098094],
                [12.32605256218687, 1.6362397820163488],
                [12.419790326575948, 1.70708446866485]])

    """
    data = np.array([[10.92182660788968, 0.6693121693121693],
                    [10.990198811901083, 0.664021164021164],
                    [11.052763648694473, 0.716931216931217],
                    [11.135287542544239, 0.7592592592592593],
                    [11.200720897228187, 0.8015873015873016],
                    [11.277568039385605, 0.8333333333333335],
                    [11.354400110520828, 0.873015873015873],
                    [11.419833465204777, 0.9153439153439153],
                    [11.493822136819762, 0.9523809523809523],
                    [11.562113962046142, 0.9894179894179895],
                    [11.638910867462918, 1.047619047619048],
                    [11.718581314444318, 1.0925925925925926],
                    [11.789706491842809, 1.1375661375661377],
                    [11.860811574545043, 1.1931216931216932],
                    [11.931921680921342, 1.2460317460317463],
                    [12.003041834645767, 1.2936507936507937],
                    [12.07415696469613, 1.3439153439153437],
                    [12.131040026123104, 1.388888888888889],
                    [12.188058726749809, 1.3624338624338628],
                    [12.22498775479447, 1.4153439153439156],
                    [12.264745111337175, 1.478835978835979],
                    [12.307340843726058, 1.547619047619048],
                    [12.367298393680217, 1.4735449735449737],
                    [12.406955276741645, 1.5899470899470902],
                    [12.426783718272358, 1.6481481481481484],
                    [12.486560415960213, 1.6693121693121695]])
    """
    data_sm = data[:, 0]
    data_size = data[:, 1]

    sm2size = spi.interp1d(data_sm, data_size, bounds_error=False, fill_value="extrapolate")

    return 10**sm2size(sm)

