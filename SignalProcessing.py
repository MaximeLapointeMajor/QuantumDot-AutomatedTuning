# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:21 2016
@author: Maxime
"""

import numpy as np
#from scipy.optimize import curve_fit
from scipy import signal
import sys
sys.path.append('C:\Codes\pyHegel')
import derivative as dv
import peakdetect as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lmfit as lmf
from copy import deepcopy


class ProcessedSignal:
    def __init__(self, xData, yData, zData, Nanofile=None, acqIndex=None, freq=None, _dummycheck=None):
        if Nanofile == None:
            self.xData = xData
            self.yData = yData
            self._Nanofile = None
            self.yName = 'yName'
            self.yUnit = 'yUnit'
            self.yStart = min(yData)
            self.yStop = max(yData)
            self.yNPoints = yData.shape[0]
            self.yResol = (self.yStop-self.yStart)/(self.yNPoints-1)
            self.xName = 'xName'
            self.xUnit = 'xUnit'
            self.xStart = min(xData)
            self.xStop = max(xData)
            self.xNPoints = xData.shape[0]
            self.xResol = (self.xStop-self.xStart)/(self.xNPoints-1)
            self.filename = 'filename'
            self.extension = 'extension'
            self.nAcqChan = 'nAcqChan'
            self.acqName = 'acqName'
            self.acqUnit = 'acqUnit'
            self.data = []
            self.data = self.data.append(zData)
            self.measType = 'measType'
            self._yFlip = False
            self.acqIndex = 0

        if Nanofile != None:
            if acqIndex == None:
                self.acqIndex = 0
            else:
                self.acqIndex = acqIndex
            self._Nanofile = Nanofile
            yData = Nanofile.yData
            xData = Nanofile.xData
            self.yName = Nanofile.yName
            self.yUnit = Nanofile.yUnit
            self.yStart = Nanofile.yStart
            self.yStop = Nanofile.yStop
            self.yNPoints = Nanofile.yNPoints
            self.yResol = abs(self.yStop-self.yStart)/(self.yNPoints-1)
            self.yData = yData
            self.xName = Nanofile.xName
            self.xUnit = Nanofile.xUnit
            self.xStart = Nanofile.xStart
            self.xStop = Nanofile.xStop
            self.xNPoints = Nanofile.xNPoints
            self.xResol = abs(self.xStop-self.xStart)/(self.xNPoints-1)
            self.xData = xData
            self.filename = Nanofile.filename
            self.extension = Nanofile.extension
            self.nAcqChan = Nanofile.nAcqChan
            self.acqName = Nanofile.acqName
            self.acqUnit = Nanofile.acqUnit
            self.data = Nanofile.data
            self.measType = Nanofile.measType
            self._yFlip = Nanofile._yFlip

        def _createDiagram(self, zData, dataType):
            return ProcessedSignal._Diagram(zData, dataType, self)
        
        if _dummycheck == None:
            self.zData = _createDiagram(self, zData, "original")
            filtered = PassFilter2D(xData, zData, freq=freq)
            self.filtered = _createDiagram(self, filtered, "filtered")
            phase = Phase2D(filtered)
            self.phase = _createDiagram(self, phase, "phase")
            derivative = Derivate2D(xData, phase)
            self.derivative = _createDiagram(self, derivative, "derivative")
            self.derivative = self.derivative.maxmin()
            transition = Transition2D(xData, yData, self.derivative.zData, tresh_axis='x', tresh='background', sigma1=1., sigma2=2.8)
            transition = Borders(transition, 15)
            self.transition = _createDiagram(self, transition, "transition")

        if _dummycheck == True:
            self.zData = _createDiagram(self, zData, "original")
            self.filtered = _createDiagram(self, zData*0, "filtered")
            self.phase = _createDiagram(self, zData*0, "phase")
            self.derivative = _createDiagram(self, zData*0, "derivative")
            self.transition = _createDiagram(self, zData*0, "transition")
            
            
            
            
            
            
            
    class _Diagram:
        def __init__(self, zData, dataType, ProSignal):
            self.zData = zData
            self._dataType = dataType
            self._ProSignal = ProSignal
        
        def plot(self, color = cm.bone, colorbar = False):
            plt.clf()
            plt.imshow(self.zData, aspect = 'auto', cmap = color, extent=(self._ProSignal.xData.min(), self._ProSignal.xData.max(), self._ProSignal.yData.min(), self._ProSignal.yData.max()))
            plt.ylabel("%(yname)s (%(yunit)s)" %{"yname":self._ProSignal.yName, "yunit":self._ProSignal.yUnit})
            plt.xlabel("%(xname)s (%(xunit)s)" %{"xname":self._ProSignal.xName, "xunit":self._ProSignal.xUnit})
            plt.title("%(filen)s -%(dset)s - %(zname)s (%(zunit)s)" %{"filen":self._ProSignal.filename, "dset":self._dataType, "zname":self._ProSignal.acqName[0][self._ProSignal.acqIndex], "zunit":self._ProSignal.acqUnit[0][self._ProSignal.acqIndex]})
            if colorbar == True:
                plt.colorbar()

        def totxt(self, xAxis=False, fmt = '%.5e'):
            if xAxis == True:
                pass
            if self._dataType == "transition":
                fmt = '%.1i'
            np.savetxt("%(fname)s - %(dtype)s.txt" %{"fname":self._ProSignal.filename, "dtype":self._dataType}, self.zData, fmt = fmt)
            
        def copy(self):
            return deepcopy(self)           
            
        def maxmin(self, plot=False, autorange=True, minimum=None, maximum=None):
            diag = self.copy()
            if autorange == True:
                aa = diag.zData.copy()
                aa = np.sort(np.ravel(aa))
                size = aa.shape[0]
                minimum = aa[int(size*.001)]
                maximum = aa[int(size*.98)]
            for u, i in enumerate(diag.zData):
                for j, k in enumerate(i):
                    if minimum != None and k < minimum:
                        diag.zData[u][j] = minimum
                    if maximum != None and k > maximum:
                        diag.zData[u][j] = maximum
            if plot == True:
                plt.clf()
                diag.plot(colorbar=True)
            return diag

    def datacutter(self, plot=False, xstart=None, xstop=None, ystart=None, ystop=None, freq=None):
        cut = deepcopy(self._Nanofile)
#        if self._yFlip == True:
#            data = np.zeros_like(cut.data)
#            for u, i in enumerate(data):
#                for j, k in enumerate(i):
#                    data[u][-j-1] = cut.data[u][j]
#            self.data = deepcopy(data)
        if self.xStart < self.xStop:
            xmax_ind = sum(1 for i in abs(cut.xData) if round(i, 6) > abs(xstop))
            xmin_ind = sum(1 for i in abs(cut.xData) if round(i, 6) < abs(xstart))
            cut.xData = cut.xData[xmin_ind:][:cut.xNPoints-xmin_ind-xmax_ind]
            cut.data = np.zeros((cut.nAcqChan, cut.yNPoints, cut.xNPoints-xmax_ind-xmin_ind))
            for u, i in enumerate(self.data):
                cut.data[u] = i.T[xmin_ind:][:cut.xNPoints-xmin_ind-xmax_ind].T
        else:
            xmax_ind = sum(1 for i in abs(cut.xData) if round(i, 6) > abs(xstop))
            xmin_ind = sum(1 for i in abs(cut.xData) if round(i, 6) < abs(xstart))
            cut.xData = cut.xData[xmax_ind:][:cut.xNPoints-xmin_ind-xmax_ind]
            cut.data = np.zeros((cut.nAcqChan, cut.yNPoints, cut.xNPoints-xmax_ind-xmin_ind))
            for u, i in enumerate(self.data):
                cut.data[u] = i.T[xmax_ind:][:cut.xNPoints-xmin_ind-xmax_ind].T
        cut.xNPoints, cut.xStart, cut.xStop = cut.xData.size, cut.xData[0], cut.xData[-1]
        if self.yStart < self.yStop:
            ymax_ind = sum(1 for i in abs(cut.yData) if round(i, 6) > abs(ystop))
            ymin_ind = sum(1 for i in abs(cut.yData) if round(i, 6) < abs(ystart))
            cut.yData = cut.yData[ymin_ind:][:cut.yNPoints-ymin_ind-ymax_ind]
            cut2 = deepcopy(cut)
            cut.data = np.zeros((cut2.nAcqChan, cut2.yNPoints-ymax_ind-ymin_ind, cut2.xNPoints))
            if self._yFlip == True:
                for u, i in enumerate(cut2.data):
                    cut.data[u] = i[ymax_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
            else:
                for u, i in enumerate(cut2.data):
                    cut.data[u] = i[ymin_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
        else:
            ymax_ind = sum(1 for i in abs(cut.yData) if round(i, 6) > abs(ystop))
            ymin_ind = sum(1 for i in abs(cut.yData) if round(i, 6) < abs(ystart))
            cut.yData = cut.yData[ymin_ind:][:cut.yNPoints-ymin_ind-ymax_ind]
            cut2 = deepcopy(cut)
            cut.data = np.zeros((cut2.nAcqChan, cut2.yNPoints-ymax_ind-ymin_ind, cut2.xNPoints))
            if self._yFlip == True:
                for u, i in enumerate(cut2.data):
                    cut.data[u] = i[ymax_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
            else:
                for u, i in enumerate(cut2.data):
                    cut.data[u] = i[ymin_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
        cut.yNPoints, cut.yStart, cut.yStop = cut.yData.size, cut.yData[0], cut.yData[-1]
#        if self._yFlip == True:
#            data = np.zeros_like(cut.data)
#            for u, i in enumerate(data):
#                for j, k in enumerate(i):
#                    data[u][-j-1] = cut.data[u][j]
#            cut.data = deepcopy(data)
        if plot == True:
            plt.figure()
            cut.plot()
        return ProcessedSignal(cut.xData, cut.yData, cut.data[self.acqIndex], cut, self.acqIndex, freq=freq)

    def _datacutter_nocomputation(self, xstart, xstop, ystart, ystop):
        cut = deepcopy(self._Nanofile)
        if self.xStart < self.xStop:
            xmax_ind = sum(1 for i in abs(cut.xData) if i > abs(xstop))
            xmin_ind = sum(1 for i in abs(cut.xData) if i < abs(xstart))
            cut.xData = cut.xData[xmin_ind:][:cut.xNPoints-xmin_ind-xmax_ind]
            cut.data = np.zeros((cut.nAcqChan, cut.yNPoints, cut.xNPoints-xmax_ind-xmin_ind))
            for u, i in enumerate(self.data):
                cut.data[u] = i.T[xmin_ind:][:cut.xNPoints-xmin_ind-xmax_ind].T
        else:
            xmax_ind = sum(1 for i in abs(cut.xData) if i > abs(xstop))
            xmin_ind = sum(1 for i in abs(cut.xData) if i < abs(xstart))
            cut.xData = cut.xData[xmax_ind:][:cut.xNPoints-xmin_ind-xmax_ind]
            cut.data = np.zeros((cut.nAcqChan, cut.yNPoints, cut.xNPoints-xmax_ind-xmin_ind))
            for u, i in enumerate(self.data):
                cut.data[u] = i.T[xmax_ind:][:cut.xNPoints-xmin_ind-xmax_ind].T
        cut.xNPoints, cut.xStart, cut.xStop = cut.xData.size, cut.xData[0], cut.xData[-1]
        if self.yStart < self.yStop:
            ymax_ind = sum(1 for i in abs(cut.yData) if i > abs(ystop))
            ymin_ind = sum(1 for i in abs(cut.yData) if i < abs(ystart))
            cut.yData = cut.yData[ymin_ind:][:cut.yNPoints-ymin_ind-ymax_ind]
            cut2 = deepcopy(cut)
            cut.data = np.zeros((cut2.nAcqChan, cut2.yNPoints-ymax_ind-ymin_ind, cut2.xNPoints))
            for u, i in enumerate(cut2.data):
                cut.data[u] = i[ymin_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
        else:
            ymax_ind = sum(1 for i in abs(cut.yData) if i > abs(ystop))
            ymin_ind = sum(1 for i in abs(cut.yData) if i < abs(ystart))
            cut.yData = cut.yData[ymax_ind:][:cut.yNPoints-ymin_ind-ymax_ind]
            cut2 = deepcopy(cut)
            cut.data = np.zeros((cut2.nAcqChan, cut2.yNPoints-ymax_ind-ymin_ind, cut2.xNPoints))
            for u, i in enumerate(cut2.data):
                cut.data[u] = i[ymax_ind:][:cut2.yNPoints-ymin_ind-ymax_ind]
        cut.yNPoints, cut.yStart, cut.yStop = cut.yData.size, cut.yData[0], cut.yData[-1]
        pp = ProcessedSignal(cut.xData, cut.yData, cut.data[self.acqIndex], cut, self.acqIndex, _dummycheck = True)
        xInd = np.where((self.xData <= xstop) & (self.xData >= xstart))[0]
        yInd = np.where((self.yData <= ystop) & (self.yData >= ystart))[0]
        if cut._yFlip == False:
            for u, i in enumerate(yInd):
                for j, k in enumerate(xInd):
                    pp.transition.zData[u,j] = self.transition.zData[i,k]
                    pp.filtered.zData[u,j] = self.filtered.zData[i,k]
                    pp.phase.zData[u,j] = self.phase.zData[i,k]
                    pp.derivative.zData[u,j] = self.derivative.zData[i,k]
        else:
            for u, i in enumerate(yInd):
                for j, k in enumerate(xInd):
                    ind = (np.size(self.yData)-1)/2
                    pp.transition.zData[-u-1,j] = self.transition.zData[(-(i-ind))+ind,k]
                    pp.filtered.zData[-u-1,j] = self.filtered.zData[(-(i-ind))+ind,k]
                    pp.phase.zData[-u-1,j] = self.phase.zData[(-(i-ind))+ind,k]
                    pp.derivative.zData[-u-1,j] = self.derivative.zData[(-(i-ind))+ind,k]
        return pp

    def _plot_box(self, color='b', ls='-'):
        plt.plot([self.xStart, self.xStart], [self.yStart, self.yStop], color=color, ls=ls)
        plt.plot([self.xStop, self.xStop], [self.yStart, self.yStop], color=color, ls=ls)
        plt.plot([self.xStart, self.xStop], [self.yStart, self.yStart], color=color, ls=ls)
        plt.plot([self.xStart, self.xStop], [self.yStop, self.yStop], color=color, ls=ls)

    def copy(self):
        return deepcopy(self)                       




            
def Reshape(xdata, ydata, zdata):
    """
    Takes the arrays in the format output by the NanonisTramea and returns an array for the x and y axis and an array of arrays for z.
    """
    N = zdata.shape[0]
    Nx = list(ydata).count(ydata[0])
    Ny = N/Nx
    zz = np.copy(zdata)
    zz.shape = (Ny,Nx)
    xx = xdata[:Nx]
    yy = np.zeros(Ny)
    for u in range(Ny):
        yy[u] = ydata[Nx*u]
    return xx,yy,zz

def _envelope(data):
    """
    Extracts the envelope of a signal by performing a Hilbert transform and taking the norm (np.abs()) of the Hilbert signal
    """
    N = data.shape[0]
    hil = signal.hilbert(data, N)
    return np.abs(hil)

def Sensitivity(xdata, ydata, fs):
    """
    Extracts the envelope of the derivative of ydata and applies a low pass filter to remove the noise in the envelope.
    
    This function is mostly useful to identify the range of operation of a SET in which it has the best sensitivity.
    """
    dd = dv.Dspline(xdata, ydata, s=.0001, k=3, n=1)
    env = _envelope(dd[1])
    env = PassFilter(xdata, env, order=5, btype='low', fs=fs, cutoff=10)
    return env

def Sensitivity2D(xdata, zdata):
    """
    Applies 1 dimentional Sensitivity function along the 'x' axis.
    """
    fs = (xdata.shape[0]-1)/(xdata.max()-xdata.min())
    zz = np.zeros_like(zdata)
    for u, i in enumerate(zdata):
        zz[u] = Sensitivity(xdata, i, fs)
    return zz

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
    
def AdjacentAveraging2D(zdata, nPoints=10):
    """
    Applies 1 dimentional AdjacentAveraging on the given data along the 'x' axis.
    """
    zz = np.zeros_like(zdata)
    for u, i in enumerate(zdata):
        zz[u] = AdjacentAveraging(i, nPoints=nPoints)
    return zz

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

def _maxima(ydata, xdata=None, lookahead=1):
    """
    
    """
    
    xdata, ydata = _datacheck_peakdetect(xdata, ydata)
#    length = len(ydata)
    peaks = []
    
    for u, (xx, yy) in enumerate(zip(xdata[lookahead:][:-lookahead], ydata[lookahead:][:-lookahead])):
        if yy == max(ydata[u:][:(2*lookahead)+1]):
            peaks.append([xx, yy])
    
    return np.array(peaks)
    
def _minima(ydata, xdata=None, lookahead=1):    
    """
    
    """
    peaks = _maxima(ydata = -ydata, xdata = xdata, lookahead=lookahead)
    peaks.T[1] = -peaks.T[1]
    return peaks

def _peak_decay_checkup(peaks, mid=None):
    """
    
    """
    if mid == None:
        mid = peaks.shape[0]/2
    
    peaks = list(peaks)
    
    for u, (xx, yy) in enumerate(peaks):
        if u < mid:
            pass#look points before
        elif u > mid:
            pass#look points after
    return np.array(peaks)
    
def _frequency_estimation(xdata, ydata):
    """
    
    """
    mmax = _maxima(ydata, xdata, lookahead = 20)
    mmin = _minima(ydata, xdata, lookahead = 20)
    delta = []
    ind = []
    
    for u, i in enumerate(mmax[:-1]):
        delta.append(mmax[u+1,0]-mmax[u,0])
    for u, i in enumerate(mmin[:-1]):
        delta.append(mmin[u+1,0]-mmin[u,0])

    avg = np.mean(delta)
    std = np.sqrt(np.var(delta))

    for u, i in enumerate(delta):
        if i < avg-2*std or i > avg+2*std:
            ind.append(u)

    index = np.argsort(-np.array(ind))
    for u in index:
        delta.pop(ind[u])

    ff = 1./np.mean(delta)

    # verify the length of measure > (mmax.shape[0]-1) * ff -- si on trouve plein de maximums qu'on finit par discarter, on doit augmenter lookahead
    # verify the resolution*2. < np.mean(delta) -- si on sonde moins de 2 points par cycle, on a un prob
    # verify resolution * lookahead < np.mean(delta)*2 -- le *2 car on veut faire sur qu'on est pas pogné sur une harmonique -- on doit donc diminuer lookahead

    #verify if ff makes sense with resolution, lookahead, length of measure
    
    return ff
    

def _cutoff(xdata, ydata, btype, fs, ff):
    """
    _cutoff calculates the cutoff frequency for a low or high pass filter in order to keep most of the desired signal but remove a maximum ammount of the DC and low frequency components or high frequency components depending on the filter type.
    
    The function calculates the Fourier transform of the signal and fits it with 3 Lorentzians (2 symetrical in positive and negative frequencies + 1 centered at 0) to extract the "main signal" and its width.
    The cutoff frequency is calculated using the peak and gamma of the double-Lorentzians.  (x0-gamma) or (x0+gamma) depending on if you are applying a high or low pass filter.
    """
    try:
#        print ff
        nPts = int(1./(((xdata.max()-xdata.min())/xdata.shape[0])*(ff/10.)))
        if nPts%2 == 0:
            nPts = nPts + 1
        if nPts < xdata.shape[0]:
            nPts = xdata.shape[0]
#        print nPts
        window = np.hanning(ydata.shape[0])
        freq = FourierFrequency(xdata, nPts)
        index = np.argsort(freq)
        tdf = FourierTransform(ydata*window, nPts)
        tdf = abs(tdf)
        pp = _maxima(tdf[index], freq[index], lookahead = 1)
#        mm = _minima(tdf[index], freq[index], lookahead=1)
        pp, hh = np.array(np.array(pp).T[0]), np.array(np.array(pp).T[1])
#        mm = np.array(np.array(mm).T[0])#, np.array(np.array(mm).T[1])
        ind = np.where(pp == min(abs(pp)))[0][0]
        ind2 = np.where(hh == max(hh[(ind+1):]))[0][0]
        for u, i in enumerate(freq):
            if i > abs(pp[ind2])*1.5 or i < -abs(pp[ind2])*1.5 or (i < abs(pp[ind2])/2. and i > -abs(pp[ind2])/2.) or (tdf[u] > hh[ind2]*1.05): #(abs(i) < abs(mm[indmin])) or 
                tdf[u] = 0.
        def lor2(x, A0, x0, gamma0):
            return A0*(1/np.pi)*(gamma0/2)/((x-x0)**2+(gamma0/2)**2)+A0*(1/np.pi)*(gamma0/2)/((x+x0)**2+(gamma0/2)**2)
        lmod2 = lmf.Model(lor2)
        lmod2.make_params()
        lmod2.set_param_hint('A0', value=max(tdf), min=max(tdf)/1000.)
        lmod2.set_param_hint('x0', value=abs(pp[ind2]), min=0.)
        lmod2.set_param_hint('gamma0', value=1., min=0.)
        result2 = lmod2.fit(tdf[index], x=freq[index])
#        print result2.values.get('x0'), result2.values.get('gamma0')
        if btype=='high':
            if result2.values.get('x0')-result2.values.get('gamma0') > 0.:
#                print "frequency: ", result2.values.get('x0')-result2.values.get('gamma0')
                if hh[ind2] != max(hh[(ind+1):]):
                    print "False", " maximum", "\n", "\n", "\n"
                return result2.values.get('x0')-result2.values.get('gamma0')
            else:
#                print "failed: 0"
                return 0.
        elif btype=='low':
            return result2.values.get('x0')+result2.values.get('gamma0')
    except Exception:
        pass
    finally:
        pass

def _butter_pass(cutoff, fs, order=5, btype='high'):
    """
    Generates the filter coefficients (numerator and denominator) of a butterworth digital filter design.
    
    "cutoff" is the "-3 dB point" of the filter
    "fs" is the sample rate
    "order" is how steep the slope of the filter is
    """
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
#    print "normalized cutoff: ", normal_cutoff
    if normal_cutoff >= 1.:
        normal_cutoff = 1.
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def PassFilter(xdata, ydata, fs, order=5, btype='high', freq = None, cutoff=None):
    """
    Applies a high or low pass filter on the data using a butterworth digital filter design where the cutoff frequency is "the -3dB point".  
    
    
    "cutoff" is the "-3 dB point" of the filter.  If cutoff is None, the cutoff frequency is calculated using the Cutoff function.
    "fs" is the sampling rate
    "order" is how steep the slope of the filter is
    "btype" is the filter type.  Can be 'high' or 'low'
    """
    if freq == None:
        freq = _frequency_estimation(xdata, ydata)
    if cutoff==None:
        cutoff = _cutoff(xdata, ydata, btype, fs, ff = freq)
    b, a = _butter_pass(cutoff, fs, order=order, btype=btype)
    y = signal.filtfilt(b, a, ydata, padtype='even')
    return y

def PassFilter2D(xdata, zdata, order=5, btype='high', freq = None, cutoff=None):
    """
    Applies 1 dimentional PassFilter function along the 'x' axis.
    """
    fs = (xdata.shape[0]-1)/abs(xdata.max()-xdata.min())
    zz = np.zeros_like(zdata)
    for u, i in enumerate(zdata):
#        print u
        zz[u] = PassFilter(xdata, i, fs=fs, order=order, btype=btype, freq = freq, cutoff=cutoff)
    return zz

def FourierFrequency(xdata, nPoints):
    """
    Calculates the frequency array for the Fourier transform of nPoints given the x axis array.
    """
    freq = np.fft.fftfreq(nPoints, d=(xdata.max()-xdata.min())/xdata.shape[0])
    return freq

def FourierTransform(data, nPoints):
    """
    Calculates the Fourier transform of the given data.
    """
    tdf = np.fft.fft(data, nPoints)
    return tdf

def FourierTransform2D(xdata, zdata, nPoints):
    """
    Calculates the 1 dimentional Fourier transform along the x axis.
    """
    freq = FourierFrequency(xdata, nPoints)
    tdf = np.zeros_like(zdata, dtype=complex)
    for u, i in enumerate(zdata):
        tdf[u] = FourierTransform(xdata, i, nPoints)
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

def Derivate(xdata, ydata, k=3, sigma=None, s=None, n=1):
    d = dv.Dspline(xdata, ydata, sigma=sigma, s=s, k=k, n=n)
    return d[1]

def Derivate2D(xdata, zdata, k=3, sigma=None, s=None, n=1):
    """
    Calculates the first derivative along the x axis using splines.
    
    k is the spline order (1 <= k <= 5)
    """
    der = np.zeros_like(zdata)
    for u, i in enumerate(zdata):
        der[u] = Derivate(xdata, i, k=k, sigma=sigma, s=s, n=n)
    return der

def _threshold(data, sigma=2.0):
    """
    Given a standard deviation, sigma, extracts the treshold under which points are "out of noise".
    """
    return np.mean(data)-sigma*np.sqrt(np.var(data))

def Transition(data, tr=None):
    """
    Given a set of data and a treshold, this function gives the value 0 to any point above treshold and -1 to any point below treshold.
    """
    if tr==None:
        tr = _threshold(data)
    temp = np.where(data<=tr, -1,0)
    return temp

def Transition2D(xdata, ydata, zdata, tresh_axis='x', tresh='all', sigma1=.5, sigma2=1.0):
    """
    Gives the value 0 to any point not detected as a transition and -1 to any point detected as one.
    
    A data point is considered a transition point if it is more negative than a number of standard deviations to the average.
    
    tresh_axis: possible values are 'x' and 'xy'.  If tresh_axis=='x', the treshold is determined for every trace along the x axis.  If tresh_axis=='xy', a global treshold is determined for the entire 2D diagram
    
    tresh: possible values are 'all' and 'background'.  If tresh=='all', all data points are considered for the calculation of both the average and standard deviation.  If tresh=='background', the function determines a first approximation of the transition points using sigma1 and removes those points from the calculation of the average and standard deviation, allowing to get statistics on the background without the transitions being considered. 
    
    """
    tre = np.zeros_like(zdata)
    if tresh_axis=='xy':
        if tresh=='all':
            tr = _threshold(np.ravel(zdata), sigma=sigma1)
        elif tresh=='background':
            zz = np.ravel(zdata)
            tr = _threshold(zz, sigma=sigma1)
            zi = Transition(zz, tr=tr)
            index = np.where(zi==-1.0)
            zz = np.delete(zz, index)
            tr = _threshold(zz, sigma=sigma2)
#        print "Threshold: %s"%tr
        for u, i in enumerate(zdata):
            tre[u] = Transition(i, tr=tr)
    elif tresh_axis=='x':
        if tresh=='all':
            for u, i in enumerate(zdata):
                tr = _threshold(i, sigma=sigma1)
                tre[u] = Transition(i, tr=tr)
        elif tresh=='background':
            tr=np.zeros_like(ydata)
            for u, i in enumerate(zdata):
                tr[u] = _threshold(i, sigma=sigma1)
                temp = Transition(i, tr=tr[u])
                index = np.where(temp==-1.0)
                temp = np.delete(i, index)
                tr[u] = _threshold(temp, sigma=sigma2)
            for i in range (ydata.shape[0]):
                tre[i] = Transition(zdata[i], tr=tr[i])
    return tre

def AdjacentTransition(der,tt,tr,nPoints):
    """
    Looks for transition points with a less severe threshold around the preselected transition points
    
    "der" is the derivative that will be searched for additional transitions
    "tt" is the actual transition points
    "tr" is the new threshold below which a point will now be considered a transition point
    "nPoints" is how far away from the actual transition points you will look for new transition points.
    """
    if nPoints < 1:
        raise ValueError("nPoints must be '1' or above")
    
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
    if nPoints < 1:
        raise ValueError("nPoints must be '1' or above")
    
    for u in range(data.shape[0]):
        for i in range(nPoints):
            data[u][i]=0
            data[u][-(i+1)]=0
    return data

def PeakSpacing(xdata, ydata, lookahead=20, delta=0, sigma=None, smooth=None, k=3, n=0, plot=True):
    """
    This function calculates the distance between the peaks of a signal by first smoothing the signal and then using the peakdetect library.
    The peakdetect library can be found at https://gist.github.com/sixtenbe/1178136
    The smoothing is done using Dspline from the derivative library that can be found in pyHegel.
    
    sigma is the standard error of y (needed for smoothing)
    s is the smoothing factor (chi^2 <=s).  The bigger s, the more the smoothing
    k is the spline order.  Default is 3 for cubic.  (1 <= k <= 5)
    n is the derivative.  Default is to fit the raw data, therefore no derivative is calculated.  (n <= k)
    
    (len(y)-sqrt(2*len(y)))*std**2 <= s <= (len(y)+sqrt(2*len(y)))*std**2
    where std is the standard error of the noise you want to smooth out of your data.
    
    The peak detection uses the peakdetect.peakdetect function.  It requires the x and y axis arrays.
    lookahead is the distance to look ahead from a peak candidate to determine if it is an actual peak
    delta specifies a minimum difference between a peak and the following points, before a peak may be considered a peak.  May be usefull for noisy signals
    The function returns the max, min peaks and the peak spacing calculated using the maximums.
    """
    spl = dv.Dspline(xdata, ydata, sigma=sigma, s=smooth, k=k, n=n)
    spl = spl[1]
    peak = pd.peakdetect(spl, xdata, lookahead=lookahead, delta=delta)
    pup = np.array(peak[0]).T
    pdown = np.array(peak[1]).T
    xspacing = np.zeros(pup[0].shape[0]-1)
    spacing = np.zeros(pup[0].shape[0]-1)
    for u, i in enumerate(xspacing):
        xspacing[u] = (pup[0,u]+pup[0,u+1])/2.
        spacing[u] = pup[0,u+1]-pup[0,u]
    ss = np.array((xspacing, spacing))
    if plot==True:
        plt.figure(100)
        plt.plot(xdata,ydata,'-b')
        plt.plot(xdata,spl,'-g')
        plt.plot(pup[0],pup[1],'or')
        plt.plot(pdown[0],pdown[1],'om')
        plt.figure(101)
        plt.plot(ss[0],ss[1],'o-b')
    return pup, pdown, ss

def Analysis(xdata, ydata, zdata):
    print "Applying high pass filters"
    zz = PassFilter2D(xdata, zdata, order=5, btype='high')
    print "Calculating the Hilbert transform and extrating the phase of the signal"
    zz = Phase2D(zz)
    print "Calculatin derivative"
    zz = Derivate2D(xdata, zz)
    print "Calculating thresholds and transitions"
    zz = Transition2D(xdata, ydata, zz, tresh_axis='xy', tresh='background', sigma1=.6, sigma2=1.)
    zz = Borders(zz, 5)
    return zz



def Hanning(data):
    """"
    Applies a Hanning window on the data set given as an argument to the function.
    
    w(n)=0.5*(1-cos(2 pi*n/(N-1)))
    """
    N=float(data.shape[0])
    temp=np.zeros(data.shape[0])
    for u, i in enumerate(data):
        temp[u]=(0.5-0.5*np.cos(2*np.pi*(u/N)))*i
    return temp

def Hamming(data):
    """"
    Applies a Hamming window on the data set given as an argument to the function.
    
    The Hamming window is similar to the Hann window except the constants (0.5) are replaced by a=0.54 and b=1-a
    
    w(n)=0.54-0.46*cos(2 pi*n/(N-1))
    """
    N=float(data.shape[0])
    temp=np.zeros(data.shape[0])
    for u, i in enumerate(data):
        temp[u]=(0.54-0.46*np.cos(2*np.pi*(u/N)))*i
    return temp

#def _smoothfactor(ydata, nPoints=40):
#    """
#    Calculates the smoothing factor 's' required for interpolation in order to remove noise.
#    
#    (len(y)-sqrt(2*len(y)))*std**2 <= s <= (len(y)+sqrt(2*len(y)))*std**2
#    where std is the standard error.
#    """
#    N=ydata.shape[0]
#    vardata = _adjacentvariance(ydata, nPoints)
#    vardata = np.mean(vardata[(int(.05*N)):(int(.9*N))])
#    return N*vardata
#
def _adjacentvariance(data,nPoints=40):
    """
    Calculates the variance of each point of an array with the nPoints on both sides of each point.
    """
    N = data.shape[0]
    vardata = np.zeros_like(data)
    for u, i in enumerate(data):
        if u<nPoints:
           vardata[u] = np.var(data[:(u*2+1)])
        elif ((N-u)<nPoints):
            vardata[u] = np.var(data[-(N*2-u*2-1):])
        else:
            temp = data[:(u+1+nPoints)]
            temp = temp[-(1+2*nPoints):]
            vardata[u] = np.var(temp)
    return vardata
#
#def Sensitivity(xdata, ydata):
#    nPoints = int(ydata.shape[0])*.1
#    dd = dv.Dspline(xdata, ydata, s=.0001, k=3, n=1)
#    env = _envelope(dd[1])
#    s = _smoothfactor(env, nPoints=nPoints)
#    env = dv.Dspline(xdata, env, s=s, k=3, n=0)
#    return env[1]
#
#def Sensitivity2D(xdata, data):
#    """
#    
#    """
#    zz = np.zeros_like(data)
#    for u, i in enumerate(data):
#        zz[u] = Sensitivity(xdata, i)
#    return zz
#