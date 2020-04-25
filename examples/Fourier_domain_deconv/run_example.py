# -*- coding: utf-8 -*-

"""
Example for deconvolution of hydrophone measurements in the Fourier domain.

Code from Jupyter notebook created by
Martin Weber, Volker Wilkens (Physikalisch-Technische Bundesanstalt, Ultrasonics Working Group)

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from helper_methods import *
from PyDynamic.uncertainty.propagate_DFT import *
from PyDynamic.uncertainty.interpolation import interp1d_unc

i=13 # i can have a value between 1 and 16 and determines the measurement scenario

# Prepare files
infos = {"i":i}
infos["resultpathname"]="Results/" # path to save data
infos, measurementfile, noisefile, hydfilename = get_file_info(infos)
measurement_data = {"name":measurementfile} #store information in dict
rawdata = np.loadtxt(measurementfile)
measurement_data["voltage"] = rawdata[4:] #read data as list

#  Preprocessing of measurement data
measurement_data["voltage"] = measurement_data["voltage"] - np.mean(measurement_data["voltage"]) # remove any DC component
print("The file {0} was read and it contains {1} data points.".format(measurement_data["name"],measurement_data["voltage"].size)) #Give some datails
measurement_data["time"] = np.array(range(0,int(rawdata[0])))*rawdata[1] #build time scale from header information
print("The time increment is {0} s".format(measurement_data["time"][1]))

## Uncertainty calculated from noise data
### Load files
noise_data = {"name":noisefile}                   # store information in dict
noise_data["voltage"] = np.loadtxt(noisefile)[4:] # read data as list
print("The file \"{0}\" was read and it contains {1} data points".format(noise_data["name"],noise_data["voltage"].size)) #Give some details
#### Calculate mean and std
mean = np.mean(noise_data["voltage"])
stdev = np.std(noise_data["voltage"])
### Plot noise data
plt.figure(1)
plt.plot(noise_data["voltage"] )
plt.plot([0,len(noise_data["voltage"])],[mean,mean])
plt.plot([0,len(noise_data["voltage"])],[mean+2*stdev,mean+2*stdev])
plt.plot([0,len(noise_data["voltage"])],[mean-2*stdev,mean-2*stdev])
plt.title("Plot of noise signal")
plt.xlabel("Datapoint")
plt.ylabel("Voltage U / V")
plt.legend(["s(i) signal",r'$\overline{s}$ mean',r'$\overline{s}+2\times\sigma$',r'$\overline{s}-2\times\sigma$'])

### Add uncertainty information to data
uncertainty = np.ones(np.shape(measurement_data["voltage"]))*stdev
print("Elements in vector: ",len(uncertainty))
measurement_data["uncertainty"] = uncertainty

## Plot measured data
plt.figure(2)
plt.subplot(211)
plt.plot(measurement_data["time"]/1e-6,measurement_data["voltage"])
plt.plot(measurement_data["time"]/1e-6,measurement_data["voltage"]+2*measurement_data["uncertainty"],"g")
plt.plot(measurement_data["time"]/1e-6,measurement_data["voltage"]-2*measurement_data["uncertainty"],"g")
plt.legend(["signal","signal + 2*uncertainty","signal - 2*uncertainty"])
plt.xlabel("time t / µs")
plt.ylabel("Signal voltage U / V")
plt.title("Filename: {}".format(measurement_data["name"]))
plt.subplot(212)
plt.plot(measurement_data["time"]/1e-6, measurement_data["uncertainty"])

## Fourier transformation of measured signal
measurement_data["frequency"] = calcfreqscale(measurement_data["time"])
measurement_data["spectrum"], measurement_data["varspec"] = \
    GUM_DFT(measurement_data["voltage"], measurement_data["uncertainty"]**2)

### normalisation of signals
measurement_data["spectrum"] = 2 * measurement_data["spectrum"]/np.size(measurement_data["time"])
measurement_data["varspec"] = 4 * measurement_data["varspec"]/(np.size(measurement_data["time"]))**2

## Plot of spectra
plt.figure(3)
# Use method ´realpart´ to get first half of frequency vector
plt.plot(realpart(measurement_data["frequency"]), amplitude(measurement_data["spectrum"]))
plt.xlabel("frequency f / Hz")
plt.ylabel("Signal")
plt.title("Filename: {}".format(measurement_data["name"]))

# Preprocessing of the hydrophone calibration data
## Reading data files
imax = None
hydcal_data = np.loadtxt(hydfilename,skiprows=1,delimiter=",")
hyd_data = {"name":hydfilename}
hyd_data["frequency"] = hydcal_data[:imax,0]*1E6
hyd_data["real"]= hydcal_data[:imax,1]
hyd_data["imag"] = hydcal_data[:imax,2]
hyd_data["varreal"] = hydcal_data[:imax,3]
hyd_data["varimag"] = hydcal_data[:imax,4]
hyd_data["kovar"] = hydcal_data[:imax,5]

## Plot hydrophone calibration data
plt.figure(4)
plt.plot(hyd_data["frequency"]/1E6,np.sqrt(hyd_data["real"]**2+hyd_data["imag"]**2))
plt.xlabel("Frequency f / MHz")
plt.ylabel("Sensitivity M / V/Pa")
plt.title("Filename: {}".format(hyd_data["name"]))

plt.figure(5)
plt.plot(hyd_data["frequency"]/1E6,np.arctan2(hyd_data["imag"],hyd_data["real"]))
plt.xlabel("Frequency f / MHz")
plt.ylabel(r"Phase $\varphi$ / rad")
plt.title("Filename: {}".format(hyd_data["name"]))


## select frequency range
fmin = 1E6
fmax = 100E6
infos["fmin_cal"]=fmin
infos["fmax_cal"]=fmax
imin = findnearestmatch(hyd_data["frequency"],fmin)
imax = findnearestmatch(hyd_data["frequency"],fmax)

## Summarise all information
print("Measurement data")
dt=(measurement_data["time"][2]-measurement_data["time"][1])
print("Points time: {} dt: {} s fs: {} MHz".format(len(measurement_data["time"]),dt*len(measurement_data["time"]),round(1/dt)/1E6))
df=(measurement_data["frequency"][2]-measurement_data["frequency"][1])
print("Points frequency: {} df: {} MHz fmax: {} MHz".format(len(measurement_data["frequency"]),df/1E6,max(measurement_data["frequency"])/1e6))
print("Hydrophone calibration data")
print("Points: {} fmin: {} MHz fmax: {} MHz df {} Hz".format(len(hyd_data["frequency"]),hyd_data["frequency"][1]/1E6,hyd_data["frequency"][-1]/1E6,hyd_data["frequency"][2]-hyd_data["frequency"][1]))
print("Selected range: {} - {} MHz".format(hyd_data["frequency"][imin]/1e6,hyd_data["frequency"][imax]/1e6))

# Interpolation and extrapolation of calibration data
## Interpolation using PyDynamic
f = measurement_data["frequency"].round()
hyd_interpolated = {"frequency": f}
imin_ = findnearestmatch(f, fmin+1)
imax_ = findnearestmatch(f, fmax)
N = len(f)//2
### interpolate in selected frequency range
hyd_interpolated["real"] = np.zeros((N,))
hyd_interpolated["varreal"] = np.zeros((N,))
_, hyd_interpolated["real"][imin_:imax_+1], hyd_interpolated["varreal"][imin_:imax_+1] = interp1d_unc(
    f[imin_:imax_+1],
    hyd_data["frequency"][imin:imax+1],
    hyd_data["real"][imin:imax+1],
    hyd_data["varreal"][imin:imax+1])
## extrapolate left and right hand sides with the nearest existing value within the selected range (like default in np.interp)
hyd_interpolated["real"][:imin_] = hyd_interpolated["real"][imin_]
hyd_interpolated["varreal"][:imin_] = hyd_interpolated["varreal"][imin_]
hyd_interpolated["real"][imax_+1:] = hyd_interpolated["real"][imax_]
hyd_interpolated["varreal"][imax_+1:] = hyd_interpolated["varreal"][imax_]


hyd_interpolated["imag"] = np.zeros((N,))
hyd_interpolated["varimag"] = np.zeros((N,))
_, hyd_interpolated["imag"][imin_:imax_+1], hyd_interpolated["varimag"][imin_:imax_+1] = interp1d_unc(
    f[imin_:imax_+1],
    hyd_data["frequency"][imin:imax+1],
    hyd_data["imag"][imin:imax+1],
    hyd_data["varimag"][imin:imax+1])
# extrapolate left and right hand sides with the last existing value within the selected range
hyd_interpolated["imag"][:imin_] = hyd_interpolated["imag"][imin_]
hyd_interpolated["varimag"][:imin_] = hyd_interpolated["varimag"][imin_]
hyd_interpolated["imag"][imax_+1:] = hyd_interpolated["imag"][imax_]
hyd_interpolated["varimag"][imax_+1:] = hyd_interpolated["varimag"][imax_]
hyd_interpolated["imag"][0] = 0  # Must be 0 by definition
hyd_interpolated["imag"][-1]= 0
hyd_interpolated["varimag"][0]=0 # Must be 0 by definition
hyd_interpolated["varimag"][-1]=0

## Use pseudo-interpolation for the covariances between real and imaginary parts
hyd_interpolated["kovar"] = np.interp(f[:N], hyd_data["frequency"][imin:imax+1], hyd_data["kovar"][imin:imax+1])
hyd_interpolated["kovar"][0] = 0 # Must be 0 by definition
hyd_interpolated["kovar"][-1] = 0

## Plot interpolated data
plt.figure(6)
plt.plot(hyd_data["frequency"], amplitude(np.r_[hyd_data["real"], hyd_data["imag"]]))
plt.plot(f[:N], amplitude(np.r_[hyd_interpolated["real"],hyd_interpolated["imag"]]))
plt.title("Amplitude")
plt.xlabel("Frequency f / MHz")
plt.ylabel(r"Amplitude M / a.u.")
plt.legend(["from file","interpolated"])
#
plt.figure(7)
plt.plot(hyd_data["frequency"], phase(np.r_[hyd_data["real"],hyd_data["imag"]]))
plt.plot(f[:N], phase(np.r_[hyd_interpolated["real"],hyd_interpolated["imag"]]))
plt.title("Phase")
plt.xlabel("Frequency f / MHz")
plt.ylabel(r"Phase $\varphi$ / rad")
plt.legend(["From file","Interpolated"])

# Transform to time domain to obtain impulse response
H_RI = np.r_[hyd_interpolated["real"],hyd_interpolated["imag"]]
U_HRI = np.r_[
    np.c_[np.diag(hyd_interpolated["varreal"]), np.diag(hyd_interpolated["kovar"])],
    np.c_[np.diag(hyd_interpolated["kovar"]), np.diag(hyd_interpolated["varimag"])]]

imp, Uimp = GUM_iDFT(H_RI, U_HRI)

## Plot centralised impulse response
dt = 1/(hyd_interpolated["frequency"][1]-hyd_interpolated["frequency"][0])
c_time = np.linspace(-dt/2,dt/2,np.size(imp))
c_imp = np.fft.fftshift(imp)
plt.figure(8)
plt.plot(c_time, c_imp)

# Deconvolution in the frequency domain
deconv = {"frequency": measurement_data["frequency"]}
deconv["P"], deconv["U_P"] = DFT_deconv(H_RI , measurement_data["spectrum"], U_HRI, measurement_data["varspec"])

## Plot result of deconvolution
plt.figure(9)
plt.subplot(2,1,1)
plt.plot(f[:N]/1E6, amplitude(measurement_data["spectrum"]))
plt.title("Meausered voltage signal spektrum")
plt.ylabel("Voltage U / V")
plt.subplot(2,1,2)
plt.plot(f[:N]/1e6, amplitude(deconv["P"]))
plt.xlabel("Frequency f / MHz")
plt.ylabel("Pressure p / Pa")

# Transformation to the time domain
deconvtime = {"t": measurement_data["time"]}
deconvtime["p"], deconvtime["Up"] = GUM_iDFT(deconv["P"], deconv["U_P"])
## correct for above normalisation
deconvtime["p"] = deconvtime["p"]/2*np.size(deconvtime["t"])
deconvtime["Up"] = deconvtime["Up"]/4*np.size(deconvtime["t"])**2

## Plot result in the time domain
plt.figure(10)
plt.plot(deconvtime["t"]/1E-6, deconvtime["p"]/1E6)
plt.plot(deconvtime["t"]/1E-6, deconvtime["p"]/1E6 - 2*np.sqrt(np.diag(deconvtime["Up"]))/1E6, color="r")
plt.plot(deconvtime["t"]/1E-6, deconvtime["p"]/1E6 + 2*np.sqrt(np.diag(deconvtime["Up"]))/1E6, color="r")
plt.xlabel("Time t / µs")
plt.ylabel("Pressure p / MPa")

plt.show()