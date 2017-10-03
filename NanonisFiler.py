# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:21 2016

@author: Maxime
"""

import numpy as np
import string
import sys
sys.path.append('C:\Codes\pyHegel')
import pyHegel.commands as ph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy

class Nanofile:
    def __init__(self, filename, extension, path=None):
        if path != None:
            ph.make_dir("%s"%path)
        self.filename = filename
        self.extension = extension
        f = open("%(fname)s.%(ext)s"%{"fname":filename, "ext":extension})
        try:
            for line in f:
                if np.any("Sweep channel" in line):
                    mType = "1D sweep"
                if np.any("Step channel 1" in line):
                    mType = "2D sweep"
                if np.any("Step channel 2" in line):
                    mType = "3D sweep"
        finally:
            f.close()
        self.measType = mType
        f = open("%(fname)s.%(ext)s"%{"fname":filename, "ext":extension})
        try:
            for line in f:
                if np.any("[DATA]" in line):
                    break
                elif np.any("Sweep channel: Name" in line):
                    xAxis = string.lstrip(line,"Sweep channel: Name\t")
                    xAxis = string.rstrip(xAxis,"\t\n")
                    xName, xUnit = string.split(xAxis, "(")
                    xName = string.rstrip(xName, " ")
                    xUnit = string.rstrip(xUnit, ")")
                    self.xName = xName
                    self.xUnit = xUnit
                elif np.any("Sweep channel: Start" in line):
                    xStart = string.lstrip(line, "Sweep channel: Start\t")
                    xStart = string.rstrip(xStart,"\t\n")
                    xStart = float(xStart)
                    self.xStart = xStart
                elif np.any("Sweep channel: Stop" in line):
                    xStop = string.lstrip(line, "Sweep channel: Stop\t")
                    xStop = string.rstrip(xStop,"\t\n")
                    xStop = float(xStop)
                    self.xStop = xStop
                elif np.any("Sweep channel: Points" in line):
                    xNPoints = string.lstrip(line, "Sweep channel: Points\t")
                    xNPoints = string.rstrip(xNPoints,"\t\n")
                    xNPoints = int(float(xNPoints))
                    self.xNPoints = xNPoints
                    self.xData = np.linspace(xStart, xStop, xNPoints)
                elif np.any("Acquire channels" in line):
                    Chans = string.lstrip(line,"Acquire channels\t")
                    Chans = string.rstrip(Chans,"\t\n")
                    name, unit = [], []
                    for u in string.split(Chans, ";"):
                        name.append(string.split(u, "(")[0])
                        unit.append(string.split(u, "(")[1])
                    for u, i in enumerate(name):
                        name[u] = string.rstrip(i, " ")
                    for u, i in enumerate(unit):
                        unit[u] = string.rstrip(i, ")")
                    nAcqChan = np.array(name).shape[0]
                    self.nAcqChan = nAcqChan
                    self.acqName = np.array(name)
                    self.acqUnit = np.array(unit)
                if mType == "1D sweep":
                    raise NotImplementedError #need to fill the yAxis data + name + unit
                elif mType != "1D sweep":
                    if np.any("Step channel 1: Name" in line):
                        yAxis = string.lstrip(line,"Step channel 1: Name\t")
                        yAxis = string.rstrip(yAxis,"\t\n")
                        yName, yUnit = string.split(yAxis, "(")
                        yName = string.rstrip(yName, " ")
                        yUnit = string.rstrip(yUnit, ")")
                        self.yName = yName
                        self.yUnit = yUnit
                    elif np.any("Step channel 1: Start" in line):
                        yStart = string.lstrip(string.lstrip(string.lstrip(line, "Step channel "), "1"), ": Start\t")
                        yStart = string.rstrip(yStart,"\t\n")
                        yStart = float(yStart)
                        self.yStart = yStart
                    elif np.any("Step channel 1: Stop" in line):
                        yStop = string.lstrip(string.lstrip(string.lstrip(line, "Step channel "), "1"), ": Stop\t")
                        yStop = string.rstrip(yStop,"\t\n")
                        yStop = float(yStop)
                        self.yStop = yStop
                    elif np.any("Step channel 1: Points" in line):
                        yNPoints = string.lstrip(string.lstrip(string.lstrip(line, "Step channel "), "1"), ": Points\t")
                        yNPoints = string.rstrip(yNPoints,"\t\n")
                        yNPoints = int(float(yNPoints))
                        self.yNPoints = yNPoints
                        self.yData = np.linspace(yStart, yStop, yNPoints)
                    if mType != "2D sweep":
                        raise NotImplementedError
        finally:
            f.close()
        if mType == "2D sweep":
            if nAcqChan == 1:
                data = []
                data.append(Lecture(filename, extension)[1:])
            elif nAcqChan > 1:
                tempdata = Lecture(filename, extension)[1:]
                data = np.zeros((nAcqChan, xNPoints, yNPoints))
                for i in range(nAcqChan):
                    for u in range(xNPoints):
                       data[i,u] = tempdata[u*nAcqChan+i]
            data = np.array(data)
            if yStart < yStop:
                data2 = np.zeros_like(data)
                for u, i in enumerate(data):
                    for j, k in enumerate(i):
                        data2[u][-j-1] = data[u][j]
                data = deepcopy(data2)
                self._yFlip = True
            else:
                self._yFlip = False
            self.data = data
    
    def plot(self, acqIndex=0,  color = cm.bone, colorbar = False):
        plt.clf()
        plt.imshow(self.data[acqIndex], aspect = 'auto', cmap = color, extent=(self.xData.min(), self.xData.max(), self.yData.min(), self.yData.max()))
        plt.ylabel("%(yname)s (%(yunit)s)" %{"yname":self.yName, "yunit":self.yUnit})
        plt.xlabel("%(xname)s (%(xunit)s)" %{"xname":self.xName, "xunit":self.xUnit})
        plt.title("%(filen)s -%(dset)s - %(zname)s (%(zunit)s)" %{"filen":self.filename, "dset":"original", "zname":self.acqName[0][acqIndex], "zunit":self.acqUnit[0][acqIndex]})
        if colorbar == True:
            plt.colorbar()

    def datacutter(self, plot=False, xstart=None, xstop=None, ystart=None, ystop=None):
        cut = deepcopy(self)
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
        if plot == True:
            plt.figure()
            cut.plot()
        return cut
        
    def copy(self):
        return deepcopy(self)   

def Lecture(filename, extension):
    """
    Takes an output data file from the NanonisTramea and converts it to a txt file that can be read using the readfile function.
    """
    data=False
    aa=[]
    for line in file("%(fname)s.%(ext)s" %{"fname":filename, "ext":extension}):
        if data==True:
            aa.append(line)
        if np.any("[DATA]" in line):
            data=True
    Datatxt="%(name)s%(ext)s"%{"name":filename,"ext":".txt"}
    with open("%s" %Datatxt,'w') as f:
        f.write("#")
        for line in aa:
            f.write("%s" %line)
    a=ph.readfile("%s"%Datatxt)
    return a
