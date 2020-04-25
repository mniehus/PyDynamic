# -*- coding: utf-8 -*-

"""
Code from Jupyter notebook created by
Martin Weber, Volker Wilkens (Physikalisch-Technische Bundesanstalt, Ultrasonics Working Group)
"""

import numpy as np

def calcfreqscale(timeseries, sign2=1):
    """Calculates the appropriate time scale for a given (equidistant) time series

    Parameters
    ----------
        timeseries: ndarray
            strictly increasing steps of the sampling time, max- and min-value is used for calculation
        sign2: integer (optional)
            sign for sin part frequencies, allows to be negative for ploting
            1 gives positive frequencies fitting to GUM_DFT data (default)
            0 gives only first half
            -1 sin part is negative, usable for easy plotting

    Returns
    -------
        f: ndarray
            frequencies matching to the result of GUM_DFT, if sign2 has the default value of 1

    """
    n = np.size(timeseries)
    if sign2 < 0:
        sign = -1
    else:
        sign = 1

    fmax = 1 / ((np.max(timeseries) - np.min(timeseries)) * n / (n - 1)) * (n - 1)

    f = np.linspace(0, fmax, n)

    f2 = np.hstack((f[0:int(n / 2. + 1)], sign * f[0:int(n / 2 + 1)]))
    if sign2 == 0:
        f2 = f[0:int(n / 2. + 1)]

    return f2

# Return the amplitude of a PyDynamic-style vector
def amplitude(data):
    n=data.size
    return np.sqrt(data[:n//2]**2+data[n//2:]**2)

# Return the phase (in rad) of a PyDynamic-style vector
def phase(data):
    n=data.size
    return np.arctan2(data[n//2:],data[:n//2])

# Return the real part of a PyDynamic-style vector
def realpart(data):
    n=data.size
    return data[:n//2]

# Return the imaginary part of a PyDynamic-style vector
def imagpart(data):
    n=data.size
    return data[n//2:]


def pulseparamter(time, pressure, deltapressure):
    """Calculates the pulse parameter of a given time series

    Parameters
    ----------
        time: ndarray
            strictly increasing steps of the sampling time, max- and min-value is used for calculation
        pressure: ndarray
            time series of pressure data
        deltapressure: ndarray
            time series of uncertainty of pressure data
        All data sets must be same length


    Returns
    -------
        result: dict
            returns the calculated pulse parameters
            pc: peak compression pressure
            pr: peak rarefaction pressure
            ppsi: pulse pressure squared integral
            _index: index of peak location in list
            _value: value of the paramter
            _uncertainty: uncertainty of the parameter
            _time: position of peak on time scale


    """
    assert len(time) == len(pressure), "Length of data sets do not match."
    assert len(time) == len(deltapressure), "Length of data sets do not match."

    dt = (max(time) - min(time)) / (len(time) - 1)
    result = {"dt": dt}
    pc_index = np.argmax(pressure)
    result["pc_index"] = pc_index
    result["pc_value"] = pressure[pc_index]
    result["pc_uncertainty"] = deltapressure[pc_index]
    result["pc_time"] = time[pc_index]

    pr_index = np.argmin(pressure)
    result["pr_index"] = pr_index
    result["pr_value"] = -pressure[pr_index]
    result["pr_uncertainty"] = deltapressure[pr_index]
    result["pr_time"] = time[pr_index]

    result["ppsi_value"] = np.sum(np.square(pressure)) * dt
    # result["ppsi_uncertainty"] = np.sqrt(np.sum(np.square(pressure*2)*np.square(deltapressure)))*dt #Without correlation, may give to small uncertainty

    # with correlation (+1). Note that the absolute value is considdered, to ensure the additive contributions of the uncertainties

    c = 2 * np.abs(pressure) * dt
    result["ppsi_uncertainty"] = np.double(np.sqrt(
        np.dot(deltapressure,c)*np.dot(c,deltapressure)
        ))
    return result

def get_file_info(infos):
    i = infos["i"]
    # M mode 3 MHz
    if i == 1:
        measurementfile = "MeasuredSignals/M-Mode 3 MHz/M3_MH44.DAT"
        noisefile = "MeasuredSignals/M-Mode 3 MHz/M3_MH44r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_MH44ReIm.csv"
        infos["hydrophonname"] = "GAMPT MH44"
        infos["measurementtype"] = "M-Mode 3 MHz"

    if i == 2:
        measurementfile = "MeasuredSignals/M-Mode 3 MHz/M3_MH46.DAT"
        noisefile = "MeasuredSignals/M-Mode 3 MHz/M3_MH46r.DAT"
        hydfilename = "HydrophonCalibrationData/MH46_MWReIm.csv"
        infos["hydrophonname"] = "GAMPT MH46"
        infos["measurementtype"] = "M-Mode 3 MHz"

    if i == 3:
        measurementfile = "MeasuredSignals/M-Mode 3 MHz/M3_ON1704.DAT"
        noisefile = "MeasuredSignals/M-Mode 3 MHz/M3_ON1704r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_ONDA1704_SECReIm.csv"
        infos["hydrophonname"] = "ONDA1704"
        infos["measurementtype"] = "M-Mode 3 MHz"

    if i == 4:
        measurementfile = "MeasuredSignals/M-Mode 3 MHz/M3_PA1434.DAT"
        noisefile = "MeasuredSignals/M-Mode 3 MHz/M3_PA1434r.DAT"
        hydfilename = "HydrophonCalibrationData/MW PA1434 ReIm.csv"
        infos["hydrophonname"] = "Precision Acoustics 1434"
        infos["measurementtype"] = "M-Mode 3 MHz"

    ### pD3-Mode 3 MHz

    if i == 5:
        measurementfile = "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44.DAT"
        noisefile = "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_MH44ReIm.csv"
        infos["hydrophonname"] = "GAMPT MH44"
        infos["measurementtype"] = "Pulse-Doppler-Mode 3 MHz"

    if i == 6:
        measurementfile = "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46.DAT"
        noisefile = "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46r.DAT"
        hydfilename = "HydrophonCalibrationData/MH46_MWReIm.csv"
        infos["hydrophonname"] = "GAMPT MH46"
        infos["measurementtype"] = "Pulse-Doppler-Mode 3 MHz"

    if i == 7:
        measurementfile = "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704.DAT"
        noisefile = "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_ONDA1704_SECReIm.csv"
        infos["hydrophonname"] = "ONDA1704"
        infos["measurementtype"] = "Pulse-Doppler-Mode 3 MHz"

    if i == 8:
        measurementfile = "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434.DAT"
        noisefile = "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434r.DAT"
        hydfilename = "HydrophonCalibrationData/MW PA1434 ReIm.csv"
        infos["hydrophonname"] = "Precision Acoustics 1434"
        infos["measurementtype"] = "Pulse-Doppler-Mode 3 MHz"

    # M mode 6 MHz
    if i == 9:
        measurementfile = "MeasuredSignals/M-Mode 6 MHz/M6_MH44.DAT"
        noisefile = "MeasuredSignals/M-Mode 6 MHz/M6_MH44r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_MH44ReIm.csv"
        infos["hydrophonname"] = "GAMPT MH44"
        infos["measurementtype"] = "M-Mode 6 MHz"

    if i == 10:
        measurementfile = "MeasuredSignals/M-Mode 6 MHz/M6_MH46.DAT"
        noisefile = "MeasuredSignals/M-Mode 6 MHz/M6_MH46r.DAT"
        hydfilename = "HydrophonCalibrationData/MH46_MWReIm.csv"
        infos["hydrophonname"] = "GAMPT MH46"
        infos["measurementtype"] = "M-Mode 6 MHz"

    if i == 11:
        measurementfile = "MeasuredSignals/M-Mode 6 MHz/M6_ON1704.DAT"
        noisefile = "MeasuredSignals/M-Mode 6 MHz/M6_ON1704r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_ONDA1704_SECReIm.csv"
        infos["hydrophonname"] = "ONDA1704"
        infos["measurementtype"] = "M-Mode 6 MHz"

    if i == 12:
        measurementfile = "MeasuredSignals/M-Mode 6 MHz/M6_PA1434.DAT"
        noisefile = "MeasuredSignals/M-Mode 6 MHz/M6_PA1434r.DAT"
        hydfilename = "HydrophonCalibrationData/MW PA1434 ReIm.csv"
        infos["hydrophonname"] = "Precision Acoustics 1434"
        infos["measurementtype"] = "M-Mode 6 MHz"

    # pD mode 7 MHz
    if i == 13:
        measurementfile = "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44.DAT"
        noisefile = "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_MH44ReIm.csv"
        infos["hydrophonname"] = "GAMPT MH44"
        infos["measurementtype"] = "Pulse-Doppler-Mode 7 MHz"

    if i == 14:
        measurementfile = "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46.DAT"
        noisefile = "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46r.DAT"
        hydfilename = "HydrophonCalibrationData/MH46_MWReIm.csv"
        infos["hydrophonname"] = "GAMPT MH46"
        infos["measurementtype"] = "Pulse-Doppler-Mode 7 MHz"

    if i == 15:
        measurementfile = "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704.DAT"
        noisefile = "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704r.DAT"
        hydfilename = "HydrophonCalibrationData/MW_ONDA1704_SECReIm.csv"
        infos["hydrophonname"] = "ONDA1704"
        infos["measurementtype"] = "Pulse-Doppler-Mode 7 MHz"

    if i == 16:
        measurementfile = "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434.DAT"
        noisefile = "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434r.DAT"
        hydfilename = "HydrophonCalibrationData/MW PA1434 ReIm.csv"
        infos["hydrophonname"] = "Precision Acoustics 1434"
        infos["measurementtype"] = "Pulse-Doppler-Mode 7 MHz"

    return infos, measurementfile, noisefile, hydfilename


def findnearestmatch(liste,value):
    """This is a help function to to find the closest value in a list
    Its purpose is to find list indices fast
    For example when list is [1,2,3] and value = 2.2
    then the return value for the index will 1 as the number 2 in the list is the closest value to 2.2
    """
    return np.argmin(abs(liste-value))

