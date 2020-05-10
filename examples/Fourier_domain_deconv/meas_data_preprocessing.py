# -*- coding: utf-8 -*-

from matplotlib.pyplot import *
from helper_methods import *
from PyDynamic.uncertainty.propagate_DFT import GUM_DFT

def read_data(meas_scenario = 13, verbose = True):
    """
    :param meas_scenario: index of measurement scenario
    :param verbose: Boolean; if true write info to command line
    :return: infos, measurement_data as dict
    """
    infos = {"i": meas_scenario}
    infos, measurementfile, noisefile, hydfilename = get_file_info(infos) # file names and such
    infos["measurementfile"] = measurementfile
    infos["noisefile"] = noisefile
    infos["hydfilename"] = hydfilename
    measurement_data = {"name": measurementfile} # store information in dict
    rawdata = np.loadtxt(measurementfile) # load raw data from file
    measurement_data["voltage"] = rawdata[4:] # read voltage data as list
    measurement_data["time"] = np.array(range(0, int(rawdata[0]))) * rawdata[
        1]  # build time scale from header information
    if verbose:
        print("The file {0} was read and it contains {1} data points.".format(measurement_data["name"],measurement_data["voltage"].size)) #Give some datails
        print("The time increment is {0} s".format(measurement_data["time"][1]))
    return infos, measurement_data

def data_preprocess(measurement_data):
    """
    :param measurement_data: voltage data, dict
    :return: measurement_data
    """
    # remove any DC component
    measurement_data["voltage"] = measurement_data["voltage"] - np.mean(measurement_data["voltage"])
    return measurement_data

def uncertainty_from_noisefile(infos, measurement_data, verbose = True, do_plot = True):
    """
    :param infos: dict containing file names and such
    :param measurement_data: dict containing measurement data read from file
    :param verbose: Boolean, if true write info to command line
    :param do_plot: Boolean, if true plot noise data
    :return: measurement data
    """
    noise_data = np.loadtxt(infos["noisefile"])[4:] # read data as list
    if verbose:
        print("The file \"{0}\" was read and it contains {1} data points".format(
            infos["noisefile"], noise_data.size))
    # Add uncertainty information to data
    uncertainty = np.ones(np.shape(measurement_data["voltage"]))*noise_data.std()
    measurement_data["uncertainty"] = uncertainty
    if do_plot:
        # Plot noise data
        mean = noise_data.mean()
        std = noise_data.std()
        figure(1)
        hist(noise_data, bins=100)
        axvline(mean, color="k")
        axvline(mean + 2*std, color="g")
        axvline(mean - 2*std, color="g")
        xlabel("Voltage U / V")

        # Plot measured data
        figure(2)
        subplot(211)
        plot(measurement_data["time"] / 1e-6, measurement_data["voltage"])
        plot(measurement_data["time"] / 1e-6, measurement_data["voltage"] + 2 * measurement_data["uncertainty"], "g")
        plot(measurement_data["time"] / 1e-6, measurement_data["voltage"] - 2 * measurement_data["uncertainty"], "g")
        legend(["signal", "signal + 2*uncertainty", "signal - 2*uncertainty"])
        xlabel("time t / µs")
        ylabel("Signal voltage U / V")
        title("Filename: {}".format(measurement_data["name"]))
        subplot(212)
        plot(measurement_data["time"] / 1e-6, measurement_data["uncertainty"])
    return measurement_data

def calculate_spectrum(measurement_data, normalize = True, do_plot = True):
    """Fourier transformation of measured signal
    :param measurement_data: dict containing measured data
    :param normalize: Boolean, if true scale spectrum and uncertainty for compatibility with continuous Fourier
    :param do_plot: Boolean, if true plot spectrum with uncertainties
    :return: measurement_data dict
    """
    measurement_data["frequency"] = calcfreqscale(measurement_data["time"])
    measurement_data["spectrum"], measurement_data["varspec"] = \
        GUM_DFT(measurement_data["voltage"], measurement_data["uncertainty"]**2)
    if normalize:
        # normalisation of signals
        measurement_data["spectrum"] = 2 * measurement_data["spectrum"]/np.size(measurement_data["time"])
        measurement_data["varspec"] = 4 * measurement_data["varspec"]/(np.size(measurement_data["time"]))**2
    if do_plot:
        # Plot of spectra
        figure(3)
        # Use method ´realpart´ to get first half of frequency vector
        plot(realpart(measurement_data["frequency"]), amplitude(measurement_data["spectrum"]))
        xlabel("frequency f / Hz")
        ylabel("Spectral amplitude")
        title("Filename: {}".format(measurement_data["name"]))

if __name__ == "__main__":
    infos, measurement_data = read_data()
    measurement_data = data_preprocess(measurement_data)
    measurement_data = uncertainty_from_noisefile(infos, measurement_data)
    calculate_spectrum(measurement_data)
    show()
