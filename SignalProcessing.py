# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:21 2016

@author: Maxime
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import sys
sys.path.append('C:\Codes\pyHegel\pyHegel')
import derivative as dv


def Reshape(x, y, z):
    """
    Takes the arrays in the format output by the NanonisTramea and returns an array for the x and y axis and an array of arrays for z.
    """
    N = z.shape[0]
    Nx = list(y).count(y[0])
    Ny = N/Nx
    zz = np.copy(z)
    zz.shape = (Ny,Nx)
    xx = x[:Nx]
    yy = np.zeros(Ny)
    for u in range(Ny):
        yy[u] = y[Nx*u]
    return xx,yy,zz

def NormalisedSignal(data):
    """
    Returns the normalised data using a Hilbert transform.
    """
    N = float(data.shape[0])
    env = signal.hilbert(data, N)
    return data/np.abs(env)

def Envelope(data):
    """
    Extracts the envelope of a signal by performing a Hilbert transform and taking the norm (np.abs()) of the Hilbert signal
    """
    N = float(data.shape[0])
    hil = signal.hilbert(data, N)
    return np.abs(hil)

def AdjacentAveraging(data, nPoints=10):
    """
    Removes the DC component of a data set by substracting each data point with the average of that point and the nPoints before and after.
    
    For the edges, the averaging uses the same ammount of points before and after, meaning that the first and last point are always 0.
    """
    N = data.shape[0]
    avdata = np.zeros_like(data)
    for u, i in enumerate(data):
        if u<nPoints:
            avdata[u] = np.average(data[:(u*2+1)])
        elif ((N-u)<nPoints):
            avdata[u] = np.average(data[-(N*2-u*2-1):])
        else:
            temp = data[:(u+1+nPoints)]
            temp = temp[-(1+2*nPoints):]
            avdata[u] = np.average(temp)
    return data-avdata
    
def AdjacentAveraging2D(data, nPoints=10):
    """
    Applies 1 dimentional AdjacentAveraging on the given data along the 'x' axis.
    """
    zz = np.zeros_like(data)
    for u, i in enumerate(data):
        zz[u] = AdjacentAveraging(i, nPoints=nPoints)
    return zz

def Cutoff(x, y):
    """
    Cutoff calculates the cutoff frequency for the high pass filter in order to keep most of the desired signal but remove a maximum ammount of the DC and low frequency components.
    
    The function starts by applying a high pass filter with a low cutoff frequency of 5 Hz, calculates the Fourier transform of the signal and fits it with 2 Lorentzians (symetrical in positive and negative frequencies) to extract the "main signal" and its width.
    The cutoff frequency is calculated using the peak and gamma of the Lorentzian (x0-gamma)
    """
    fs = (x.shape[0]-1)/(x.max()-x.min())
    y2 = HighPassFilter(x, y, fs, order=5, cutoff=5.)
    freq = FourierFrequency(x, x.shape[0])
    tdf = FourierTransform(y2, y2.shape[0])
    tdf = abs(tdf)
    def lor(x, A, x0, gamma):
        return A*(1/np.pi)*(gamma/2)/((x-x0)**2+(gamma/2)**2)+A*(1/np.pi)*(gamma/2)/((x+x0)**2+(gamma/2)**2)
    p0 = ([10.,30.,5.])
    ret = curve_fit(lor, freq, tdf, p0)
    p0 = ret[0]
    return abs(p0[1])-p0[2]

def butter_highpass(cutoff, fs, order=5):
    """
    Generates the filter coefficients (numerator and denominator) of a butterworth digital filter design.
    
    "cutoff" is the "-3 dB point" of the filter
    "fs" is the sample rate
    "order" is how steep the slope of the filter is
    """
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def HighPassFilter(x, data, fs, order=5, cutoff=None):
    """
    Applies a high pass filter on the data using a butterworth digital filter design where the cutoff frequency is "the -3dB point".  
    
    "cutoff" is the "-3 dB point" of the filter.  If cutoff is None, the cutoff frequency is calculated using the Cutoff function.
    "fs" is the sampling rate
    "order" is how steep the slope of the filter is
    """
    if cutoff==None:
        cutoff=Cutoff(x, data)
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data, padtype='even')
    return y

def HighPassFilter2D(x, z, fs, order=5, cutoff=None):
    """
    Applies 1 dimentional HighPassFilter function along the 'x' axis.
    """
    zz = np.zeros_like(z)
    for u, i in enumerate(z):
        zz[u] = HighPassFilter(x, i, fs, order=order, cutoff=cutoff)
    return zz

def FourierFrequency(x, nPoints):
    """
    Calculates the frequency array for the Fourier transform of nPoints given the x axis array.
    """
    freq = np.fft.fftfreq(nPoints, d=(x.max()-x.min())/x.shape[0])
    return freq

def FourierTransform(z, nPoints):
    """
    Calculates the Fourier transform of the given data.
    """
    tdf = np.fft.fft(z, nPoints)
    return tdf

def FourierTransform2D(x, z, nPoints):
    """
    Calculates the 1 dimentional Fourier transform along the x axis.
    """
    freq = FourierFrequency(x, nPoints)
    tdf = np.zeros_like(z, dtype=complex)
    for u, i in enumerate(z):
        tdf[u] = FourierTransform(x, i, nPoints)
    return freq, tdf

def Phase(data):
    """
    Extracts the phase of each point of the data set given as an argument to the function.
    
    Performs a Hilbert transform of the signal and returns the phase=arctan(Im/Re).
    
    The function also unwraps the phase.
    """
    hil = signal.hilbert(data)
    return np.unwrap(np.arctan2(hil.imag, hil.real))

def Phase2D(data):
    """
    Extracts the phase of each data point along the x axis.  Before applying, the DC and low frequency components must be removed from the signal.
    
    Performs an adjacent averaging on each point before doing a 1D Hilbert transform along x and returning the phase=arctan(Im/Re).
    
    The function also unwraps the phase.
    """
    ph = np.zeros_like(data)
    for u, i in enumerate(data):
        ph[u] = Phase(i)
    return ph

def Derivate(x, y, k=3, sigma=None, s=None, n=1):
    d = dv.Dspline(x, y, sigma=sigma, s=s, k=k, n=n)
    return d[1]

def Derivate2D(x, z, k=3, sigma=None, s=None, n=1):
    """
    Calculates the first derivative along the x axis using splines.
    
    k is the spline order (1 <= k <= 5)
    """
    der = np.zeros_like(z)
    for u, i in enumerate(z):
        der[u] = Derivate(x, i, k=k, sigma=sigma, s=s, n=n)
    return der

def Threshold(z, sigma=2.0):
    """
    Given a standard deviation, sigma, extracts the treshold under which points are "out of noise".
    """
    return np.mean(z)-sigma*np.sqrt(np.var(z))

def Transition(z, tr=None):
    """
    Given a set of data and a treshold, this function gives the value 0 to any point above treshold and -1 to any point below treshold.
    """
    if tr==None:
        tr = Threshold(z)
    temp = np.where(z<=tr, -1,0)
    return temp

def Transition2D(x, y, z, tresh_axis='x', tresh='all', sigma1=2.0, sigma2=2.0):
    """
    Gives the value 0 to any point not detected as a transition and -1 to any point detected as one.
    
    A data point is considered a transition point if it is more negative than a number of standard deviations to the average.
    
    tresh_axis: possible values are 'x' and 'xy'.  If tresh_axis=='x', the treshold is determined for every trace along the x axis.  If tresh_axis=='xy', a global treshold is determined for the entire 2D diagram
    
    tresh: possible values are 'all' and 'background'.  If tresh=='all', all data points are considered for the calculation of both the average and standard deviation.  If tresh=='background', the function determines a first approximation of the transition points using sigma1 and removes those points from the calculation of the average and standard deviation, allowing to get statistics on the background without the transitions being considered. 
    
    """
    tre = np.zeros_like(z)
    if tresh_axis=='xy':
        if tresh=='all':
            tr = Threshold(np.ravel(z), sigma=sigma1)
        elif tresh=='background':
            zz = np.ravel(z)
            tr = Threshold(zz, sigma=sigma1)
            zi = Transition(zz, tr=tr)
            index = np.where(zi==-1.0)
            zz = np.delete(zz, index)
            tr = Threshold(zz, sigma=sigma2)
        print tr
        for u, i in enumerate(z):
            tre[u] = Transition(i, tr=tr)
    elif tresh_axis=='x':
        if tresh=='all':
            for u, i in enumerate(z):
                tr = Threshold(i, sigma=sigma1)
                tre[u] = Transition(i, tr=tr)
        elif tresh=='background':
            tr=np.zeros_like(y)
            for u, i in enumerate(z):
                tr[u] = Threshold(i, sigma=sigma1)
                temp = Transition(i, tr=tr[u])
                index = np.where(temp==-1.0)
                temp = np.delete(i, index)
                tr[u] = Threshold(temp, sigma=sigma2)
            for i in range (y.shape[0]):
                tre[i] = Transition(z[i], tr=tr[i])
    return tre

def AdjacentTransition(der,tt,tr,nPoints):
    """
    Looks for transition points with a less severe threshold around the preselected transition points
    
    "der" is the derivative that will be searched for additional transitions
    "tt" is the actual transition points
    "tr" is the new threshold below which a point will now be considered a transition point
    "nPoints" is how far away from the actual transition points you will look for new transition points.
    """
    xSize=tt.shape[1]
    ySize=tt.shape[0]
    index=np.where(tt!=0)
    index=np.array(index).T
    for i in index:
        for j in range(-nPoints,nPoints+1):
            for k in range(-nPoints,nPoints+1):
                if der[(i[0]+j)%ySize][(i[1]+k)%xSize]<tr:
                    tt[(i[0]+j)%ySize][(i[1]+k)%xSize]=-1.0
    return tt

def Borders(data,nPoints):
    """
    Zeros the nPoints closer to all the edges of the image.
    """
    for u in range(data.shape[0]):
        if u<nPoints:
            for i in range(data.shape[1]):
                data[u][i]=0
        else:
            for i in range(nPoints):
                data[u][i]=0
                data[u][-(i+1)]=0
    return data








#def FullAnalysis(x,y,z,aa,s1,s2):
#    ph=Phase2D(x,y,z,aa)
#    der=Derivate2D(x,y,ph)
#    tt=Transition2D(x,y,der,tresh_axis='xy',tresh='background',sigma1=s1,sigma2=s2)
#    tt=Borders(tt,6)
#    return tt
#
#def Transition2D(x,y,z,tresh_axis='x',tresh='all',sigma1=2.0,sigma2=2.0):
#    """
#    Gives the value 0 to any point not detected as a transition and -1 to any point detected as one.
#    
#    A data point is considered a transition point if it is more negative than a number of standard deviations to the average.
#    
#    tresh_axis: possible values are 'x' and 'xy'.  If tresh_axis=='x', the treshold is determined for every trace along the x axis.  If tresh_axis=='xy', a global treshold is determined for the entire 2D diagram
#    
##    tresh: possible values are 'all' and 'background'.  If tresh=='all', all data points are considered for the calculation of both the average and standard deviation.  If tresh=='background', the function determines a first approximation of the transition points using sigma1 and removes those points from the calculation of the average and standard deviation, allowing to get statistics on the background without the transitions being considered. 
#    
#    """
#    tre=np.zeros(z.shape)
#    if tresh_axis=='xy':
#        if tresh=='all':
#            tr=Threshold(np.ravel(z),sigma=sigma1)
#        elif tresh=='background':
#            zz=np.ravel(z)
#            tr=Threshold(zz,sigma=sigma1)
#            tre=Transition(zz,tr=tr)
#            index=np.where(tre==-1.0)
#            zz=np.delete(zz,index)
#            tr=Threshold(zz,sigma=sigma2)
#        print tr
#        tre=Transition(z,tr=tr)
#    elif tresh_axis=='x':
#        temp=np.zeros(x.shape[0])
#        if tresh=='all':
#            for i in range (y.shape[0]):
#                temp=np.copy(z[i])
#                tr=Threshold(temp,sigma=sigma1)
#                temp=Transition(temp,tr=tr)
#                tre[i]=np.copy(temp)
#        elif tresh=='background':
#            tr=Threshold(np.ravel(z),sigma=sigma1)
#            for i in range (y.shape[0]):
#                temp=np.copy(z[i])
#                temp=Transition(temp,tr=tr)
#                tre[i]=np.copy(temp)
#            zz=np.ravel(tre)
#            index=np.where(zz==-1.0)
#            zz=np.delete(zz,index)
#            tr=Threshold(zz,sigma=sigma2)
#            for i in range (y.shape[0]):
#                temp=np.copy(z[i])
#                temp=Transition(temp,tr=tr)
#                tre[i]=np.copy(temp)
#                ##  for i,j in zip(z,tre):
#                ##        temp = np.copy(i)
#                ##        temp = Transition(temp, tr=tr)
#                ##        j = cp.copy(temp)
#    return tre
#
#def Hanning(data):
#    """"
#    Applies a Hanning window on the data set given as an argument to the function.
#    
#    w(n)=0.5*(1-cos(2 pi*n/(N-1)))
#    """
#    N=float(data.shape[0])
#    temp=np.zeros(N)
#    for u, i in enumerate(data):
#        temp[u]=(0.5-0.5*np.cos(2*np.pi*(u/N)))*i
#    return temp
#
#def Hamming(data):
#    """"
#    Applies a Hamming window on the data set given as an argument to the function.
#    
#    The Hamming window is similar to the Hann window except the constants (0.5) are replaced by a=0.54 and b=1-a
#    
#    w(n)=0.54-0.46*cos(2 pi*n/(N-1))
#    """
#    N=float(data.shape[0])
#    temp=np.zeros(N)
#    for u, i in enumerate(data):
#        temp[u]=(0.54-0.46*np.cos(2*np.pi*(u/N)))*i
#    return temp
#
#
#








