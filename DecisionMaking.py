# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:21 2016
@author: Maxime
"""

import numpy as np
#from itertools import product, izip
from copy import deepcopy
#import lmfit as lmf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SignalProcessing as sp
import ImageDetection as ID

class Gate:
    """
    
    """
    
    def __init__(self, Gname, Vmin, Vmax, Vstart):
        self.name = Gname
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Voltage = Vstart
        self._Vstart = Vstart

    def copy(self):
        return deepcopy(self)

class Diagram:
    """
    
    """
    
    def __init__(self, xGate, xResol, yGate, yResol):
        self.xMin = xGate.Vmin
        self.xMax = xGate.Vmax
        self.yMin = yGate.Vmin
        self.yMax = yGate.Vmax
        self.xResol = xResol
        self.yResol = yResol    
    
        xNPoints = int(round((self.xMax-self.xMin)/self.xResol)+1)
        yNPoints = int(round((self.yMax-self.yMin)/self.yResol)+1)
        self.grid = np.zeros((yNPoints, xNPoints))
        self.grid[0,0], self.grid[-2,1], self.grid[1,-2], self.grid[-1,-1] = -1, -1, -1, -1 #done to always have leftover and avoid fitting mistake from an empty leftover cluster in ImageDetection loop
        self.xData = np.linspace(self.xMin, self.xMax, xNPoints)
        self.yData = np.linspace(self.yMin, self.yMax, yNPoints)

        self.MeasWindows = []
        self.clist = []
        self.leftover = []
        self.tlist = []
        self._steps = []
        self.MeasWindows, self.clist, self.leftover, self.tlist, self._steps = np.array(self.MeasWindows), np.array(self.clist), np.array(self.leftover), np.array(self.tlist), np.array(self._steps)
        
        self._proSignal = None

    def new_measurement(self, MeasWindow):
        self.MeasWindows, self.clist, self.leftover, self.tlist, self._steps = list(self.MeasWindows), list(self.clist), list(self.leftover), list(self.tlist), list(self._steps)
        self.MeasWindows.append(MeasWindow)
        self._steps.append(MeasWindow.mstate._step)
        for u in MeasWindow.ProcessedImage.clusters[:-1]:
            self.clist.append(u)
        for u in MeasWindow.ProcessedImage.clusters[-1].cluster:
            self.leftover.append(u)
        for u in MeasWindow.ProcessedImage.transitions:
            self.tlist.append(u)
        delta_y = int(round((self.yMax-MeasWindow.ProcessedImage._proSignal.yStop)/self.yResol))
        delta_x = int(round((MeasWindow.ProcessedImage._proSignal.xStart-self.xMin)/self.xResol))
        for u, i in enumerate(MeasWindow.ProcessedImage.clusters):
            for j, k in enumerate(i.cluster):
                self.grid[delta_y+k[0], delta_x+k[1]]=-1
        self.MeasWindows, self.clist, self.leftover, self.tlist, self._steps = np.array(self.MeasWindows), np.array(self.clist), np.array(self.leftover), np.array(self.tlist), np.array(self._steps)

    def plot_grid(self):
        plt.imshow(self.grid, aspect='auto', cmap=cm.bone, extent=(self.xMin, self.xMax, self.yMin, self.yMax))

    def plot_boxes(self):
        colormap = cm.RdYlBu_r
        num_plots = np.size(self.MeasWindows)
        colormap = ([colormap(i) for i in np.linspace(0.0, 1., num_plots)])
        for u, i in enumerate(self.MeasWindows):
            i.ProcessedImage._proSignal._plot_box(color=colormap[u])

    def _gen_pro_signal(self, _yFlip=False):
        if self._proSignal == None:
            self._proSignal = sp.ProcessedSignal(self.xData, self.yData, self.grid*0.)
        self._proSignal.transition.zData = deepcopy(self.grid)
        if _yFlip == True:
            self._proSignal._yFlip = True
        return self._proSignal

    def _gen_pro_image(self, _yFlip=False):
        proSignal = self._gen_pro_signal(_yFlip = _yFlip)
        proImage = ID.ProcessedImage(proSignal)
        return proImage

    def update_lists(self, _yFlip = False):
        proImage = self._gen_pro_image(_yFlip = _yFlip)
        self.clist, self.leftover, self.tlist = [], [], []
        for u in proImage.clusters[:-1]:
            self.clist.append(u)
        for u in proImage.clusters[-1].cluster:
            self.leftover.append(u)
        for u in proImage.transitions:
            self.tlist.append(u)
            if u.length < .35:
                u._update_test_flag(False)
            else:
                u._update_test_flag(True)
        self.clist, self.leftover, self.tlist = np.array(self.clist), np.array(self.leftover), np.array(self.tlist)            

    def copy(self):
        return deepcopy(self)

class MeasuredWindow:
    """
    
    """
    
    def __init__(self, ProcessedImage, mstate):
        self.ProcessedImage = ProcessedImage
        self.mstate = mstate


        if np.size(self.ProcessedImage.transitions) != 0:
            self._trans_found = True
        else:
            self._trans_found = False

        if self.mstate._step == 'confirmtrans':
            for u, i in enumerate(self.ProcessedImage.transitions):
                if self.ProcessedImage.transitions[u].length < .35:
                    self.ProcessedImage.transitions[u]._update_test_flag(False)
                else:
                    self.ProcessedImage.transitions[u]._update_test_flag(True)

    def copy(self):
        return deepcopy(self)
    
class MeasState:
    """
    
    """
    def __init__(self, diag, *args, **kwargs):
        self.xRange = 0.13
        self.yRange = 0.10
        self.xResol = diag.xResol
        self.yResol = diag.yResol
        self._step = ''
        for u in kwargs.items():
            if u[0] == 'xmin':
                self.xx = u[0]
                self.xst = u[1]
                self.xmin = u[1]
            elif u[0] == 'xmid':
                self.xx = u[0]
                self.xst = u[1]
                self.xmid = u[1]
            elif u[0] == 'xmax':
                self.xx = u[0]
                self.xst = u[1]
                self.xmax = u[1]
            if u[0] == 'ymin':
                self.yy = u[0]
                self.yst = u[1]
                self.ymin = u[1]
            elif u[0] == 'ymid':
                self.yy = u[0]
                self.yst = u[1]
                self.ymid = u[1]
            elif u[0] == 'ymax':
                self.yy = u[0]
                self.yst = u[1]
                self.ymax = u[1]

    def _update_step(self, step):
        self._step = step

    def _update_xy(self):
        if self.xx == 'xmin':
            self.xmax = self.xmin + self.xRange
            self.xmid = self.xmin + self.xRange/2.
        elif self.xx == 'xmid':
            self.xmin = self.xmid - self.xRange/2.
            self.xmax = self.xmid + self.xRange/2.
        elif self.xx == 'xmax':
            self.xmin = self.xmax - self.xRange
            self.xmid = self.xmax - self.xRange/2.
        if self.yy == 'ymin':
            self.ymax = self.ymin + self.yRange
            self.ymid = self.ymin + self.yRange/2.
        elif self.yy == 'ymid':
            self.ymin = self.ymid - self.yRange/2.
            self.ymax = self.ymid + self.yRange/2.
        elif self.yy == 'ymax':
            self.ymin = self.ymax - self.yRange
            self.ymid = self.ymax - self.yRange/2.

    def _update_xRange(self, xRange):
        self.xRange = xRange
        self._update_xy()

    def _update_yRange(self, yRange):
        self.yRange = yRange
        self._update_xy()

    def copy(self):
        return deepcopy(self)

def _extract_xRange(diag, proSignal, xx, yy):
    maxrange = diag.xMax-diag.xMin
    i = 0
    s = .1
    while i < 100: 
        if s > maxrange:
            return maxrange
        try:
            xData, xarray = _extract_xaxis(diag, proSignal, xx, yy, s)
            ff = sp._frequency_estimation(xData, xarray)
            if ff == None:
                raise
            else:
                return 2./ff + 40*diag.xResol
        except:
            s = s+.02
            i = i+1

def _extract_yRange(diag, proSignal, xx, yy):

    return 24.*diag.yResol
##This part had to be removed because we dont sweep along the y axis & telegraphic noise messes the entire yFreq estimation
#    maxrange = diag.yMax - diag.yMin
#    i = 0
#    s = .1
#    while i < 100: 
#        if s > maxrange:
#            return maxrange
#        try:
#            yData, yarray = _extract_xaxis(diag, proSignal, xx, yy, s)
#            index = np.argsort(yData)
#            ff = sp._frequency_estimation(yData[index], yarray[index])
#            if ff == None:
#                raise
#            else:
#                if .5/ff > 24.*diag.yResol:
#                    return .5/ff
#                else:
#                    return 24.*diag.yResol     
#        except:
#            s = s+.02
#            i = i+1

def _extract_xaxis(diag, proSignal, xx, yy, xRange):
    xmin = xx-xRange/2
    xmax = xx+xRange/2
    if xmax > diag.xMax:
        delta = xmax-diag.xMax
        xmax = diag.xMax
        xmin = xmin-delta
    if xmin < diag.xMin:
        delta = diag.xMin-xmin
        xmin = diag.xMin
        xmax = xmax+delta
    yy = _find_nearest(proSignal.yData, yy)
    indxmin = int(round((xmin-diag.xMin)/diag.xResol))
    indxmax = int(round((xmax-diag.xMin)/diag.xResol))+1
    indy = int(round((diag.yMax-yy)/diag.yResol))
    xarray = proSignal.zData.zData[indy][:indxmax][indxmin:]
    xx = np.linspace(xmin, xmax, int(round((xmax-xmin)/diag.xResol))+1)
    return xx, xarray

def _extract_yaxis(diag, proSignal, xx, yy, yRange):
    xx = _find_nearest(proSignal.xData, xx)
    ymin = yy-yRange/2
    ymax = yy+yRange/2
    if ymax > diag.yMax:
        delta = ymax-diag.yMax
        ymax = diag.yMax
        ymin = ymin-delta
    if ymin < diag.yMin:
        delta = diag.yMin-ymin
        ymin = diag.yMin
        ymax = ymax+delta
    indymin = int(round((diag.yMax-ymin)/diag.yResol))+1
    indymax = int(round((diag.yMax-ymax)/diag.yResol))
    indx = int(round((xx-diag.xMin)/diag.xResol))
    yarray = []
    for u in range(indymax, indymin):
        yarray.append(proSignal.zData.zData[u][indx])
    yy = np.linspace(ymax, ymin, int(round((ymax-ymin)/diag.yResol))+1)
    return yy, np.array(yarray)

def _find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def _meas_printer(mstate):
    mstate._update_xy()
    print "Next measurement fromx = %(xmin)s to x = %(xmax)s and y = %(ymin)s to y = %(ymax)s" %{"xmin":mstate.xmin, "xmax":mstate.xmax, "ymin":mstate.ymin, "ymax":mstate.ymax}

def _meas_extraction(mstate, proSignal, nocomp=True):
    if nocomp == True:
        c0 = proSignal._datacutter_nocomputation(xstart=mstate.xmin, xstop=mstate.xmax, ystart=mstate.ymin, ystop=mstate.ymax)
        try:
            i0 = ID.ProcessedImage(c0)
        except(IndexError, TypeError):
            c0.transition.zData[1][1], c0.transition.zData[-2][-2] = -1, -1
            i0 = ID.ProcessedImage(c0)
        mw0 = MeasuredWindow(i0, mstate)
        return mw0
    elif nocomp == False:
        c0 = proSignal.datacutter(xstart=mstate.xmin, xstop=mstate.xmax, ystart=mstate.ymin, ystop=mstate.ymax)
        try:
            i0 = ID.ProcessedImage(c0)
        except(IndexError, TypeError):
            c0.transition.zData[1][1], c0.transition.zData[-2][-2] = -1, -1
            i0 = ID.ProcessedImage(c0)
        mw0 = MeasuredWindow(i0, mstate)
        return mw0


def _wrapper(diag, proSignal, _yFlip = False, nocomp=True, numCalls = 100):
    mstate = 1
    ind = 0
    while ind < numCalls:
        print ind
        mstate = next_step(diag, proSignal, _yFlip = _yFlip)
        if mstate == None:
            break
        mw = _meas_extraction(mstate, proSignal, nocomp=nocomp)
        diag.new_measurement(mw)
        ind = ind+1
    return diag

def next_step(diag, proSignal, _yFlip = False):
    if np.size(diag.tlist) == 0:
        if np.size(diag.MeasWindows) == 0:
            mstate = _first_measurement(diag, proSignal)
            return mstate
        else:
            mstate = _find_any_transition(diag, proSignal, ind=-1)
            return mstate
    else:
        if diag.MeasWindows[-1].mstate._step == 'confirmtrans':
            diag.update_lists(_yFlip=_yFlip)
        verif = []
        for u in diag.tlist:
            verif.append(u._tested_flag)
        verif = np.array(verif)
        if np.all(verif==False):
            if np.any(diag._steps == 'findanytrans'):
                ind = max(np.where(diag._steps == 'findanytrans')[0])
            else:
                ind = np.where(diag._steps == 'init')[0][0]
            mstate = _find_any_transition(diag, proSignal, ind=ind)
            return mstate
        elif np.any(verif==True):
            if diag.MeasWindows[-1].mstate._step == 'last':
                diag.update_lists(_yFlip = _yFlip)
                print "last transition was found.  all that's needed is to perform voltage addition on last slice of meas."
                return None
            if (diag.MeasWindows[-1].mstate._step == 'goup' and np.size(diag.MeasWindows[-1].ProcessedImage.transitions) == 0):
                diag.update_lists(_yFlip = _yFlip)
            if diag.MeasWindows[-1]._trans_found == True:
                if diag.MeasWindows[-1].mstate.ymax < diag.yMax:
                    mstate = _goup(diag, proSignal)
                    return mstate
                else:
                    if diag.MeasWindows[-1].mstate.xmin > diag.xMin and (diag.MeasWindows[-1].mstate._step == 'goup' or diag.MeasWindows[-1].mstate._step == 'goleft'):
                        mstate = _goleft(diag, proSignal)
                        return mstate
                    else:
                        if diag.MeasWindows[-1].mstate.xmax < diag.xMax:
                            mstate = _goright(diag, proSignal)
                            return mstate
                        else:
                            diag.update_lists(_yFlip = _yFlip)
                            mstate = _last_measurement(diag, proSignal)
                            return mstate
            elif diag.MeasWindows[-1]._trans_found == False:
                if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goup':
                    mstate = _goleftup(diag, proSignal)
                    return mstate
                if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goleft':
                    mstate = _goleft(diag, proSignal)
                    return mstate
                else:
                    if diag.MeasWindows[-1].mstate.xmax < diag.xMax:
                        mstate = _goright(diag, proSignal)
                        return mstate
                    else:
                        if diag.MeasWindows[-1].mstate._step == 'last':
                            diag.update_lists(_yFlip = _yFlip)
                            print "last transition was found.  all that's needed is to perform voltage addition on last slice of meas."
                            return None
                        else:
                            diag.update_lists(_yFlip = _yFlip)
                            mstate = _last_measurement(diag, proSignal)
                            return mstate
        elif np.any(verif==None):
            ind = np.where(verif == None)[0][0]
            diag.tlist[ind]._update_test_flag(False)
            mstate = _confirm_transition(diag, proSignal, ind)
            return mstate

def _first_measurement(diag, proSignal):
    ymid = (diag.yMin+diag.yMax)/2.
    xmax = diag.xMax
    setting = {"ymid":ymid, "xmax":xmax}
    mstate = MeasState(diag, **setting)
    mstate._update_step('init')
    mstate._update_xy()
    xRange = _extract_xRange(diag, proSignal, mstate.xmid, mstate.ymid)
    yRange = _extract_yRange(diag, proSignal, mstate.xmid, mstate.ymid)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    return mstate

def _find_any_transition(diag, proSignal, ind=-1):
    xmax = diag.MeasWindows[ind].mstate.xmin
    ymin = diag.MeasWindows[ind].mstate.ymax
    setting = {"ymin":ymin, "xmax":xmax} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('findanytrans')
    xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymin)
    yRange = _extract_yRange(diag, proSignal, mstate.xmax, mstate.ymin)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    if mstate.ymax > diag.yMax or mstate.xmin < diag.xMin:
        ind = []
        for u, i in enumerate(diag.MeasWindows):
            if i.ProcessedImage._proSignal.xStop == diag.xMax:
                ind.append(u)
        xmax = diag.xMax
        if np.size(ind) == 1:
            ymin = diag.MeasWindows[ind[-1]].ProcessedImage._proSignal.yStop + 1.5*diag.MeasWindows[ind[-1]].mstate.yRange
            setting = {"ymin":ymin, "xmax":xmax} 
            mstate = MeasState(diag, **setting)
            xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymin)
            yRange = _extract_yRange(diag, proSignal, mstate.xmax, mstate.ymin)
        elif np.size(ind)%2 == 0:
            ymax = diag.MeasWindows[ind[-2]].ProcessedImage._proSignal.yStart - 1.5*diag.MeasWindows[ind[-2]].mstate.yRange
            setting = {"ymax":ymax, "xmax":xmax} 
            mstate = MeasState(diag, **setting)
            xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymax)
            yRange = _extract_yRange(diag, proSignal, mstate.xmax, mstate.ymax)
        else:
            ymin = diag.MeasWindows[ind[-2]].ProcessedImage._proSignal.yStop + 1.5*diag.MeasWindows[ind[-2]].mstate.yRange
            setting = {"ymin":ymin, "xmax":xmax} 
            mstate = MeasState(diag, **setting)
            xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymin)
            yRange = _extract_yRange(diag, proSignal, mstate.xmax, mstate.ymin)
        mstate._update_xRange(xRange)
        mstate._update_yRange(yRange)
        mstate._update_step('findanytrans')
        mstate._update_xy()
        mstate = _check_ymax(diag, mstate)
        if mstate.ymin < diag.yMin:
            ind = []
            for u, i in enumerate(diag.MeasWindows):
                if i.ProcessedImage._proSignal.yStart == diag.yMin:
                    ind.append(u)
            mstate = _check_ymin(diag, mstate)
            if np.size(ind) != 0:
                ind = max(ind)
                mstate.xmax, mstate.xx = diag.MeasWindows[ind].mstate.xmin-diag.MeasWindows[ind].mstate.xRange, 'xmax'
                mstate._update_xy()
                mstate = _check_xmin(diag, mstate)
        return mstate
    else:
        return mstate

def _confirm_transition(diag, proSignal, ind):
    xmid = (diag.tlist[ind].vStart_x + diag.tlist[ind].vStop_x)/2.
    ymid = (diag.tlist[ind].vStart_y + diag.tlist[ind].vStop_y)/2.
    setting = {"xmid":xmid, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('confirmtrans')
    xRange = _extract_xRange(diag, proSignal, mstate.xmid, mstate.ymid)
    yRange = _extract_yRange(diag, proSignal, mstate.xmid, mstate.ymid)
    if diag.MeasWindows[-1].mstate.yRange < .4:
        mstate._update_yRange(.4)
    else:
        mstate._update_yRange(yRange)
    xRange_trans = 2*max([(mstate.ymax-diag.tlist[ind].intercept)/diag.tlist[ind].slope-mstate.xmid, (mstate.ymin-diag.tlist[ind].intercept)/diag.tlist[ind].slope-xmid])
    if xRange_trans > mstate.xRange:
        mstate._update_xRange(diag.xResol*40 + xRange_trans)
    else:
        mstate._update_xRange(xRange)
    mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    return mstate   

def _goup(diag, proSignal):
    ymin = diag.MeasWindows[-1].mstate.ymax
    xmid = (ymin-diag.MeasWindows[-1].ProcessedImage.transitions[0].intercept)/diag.MeasWindows[-1].ProcessedImage.transitions[0].slope
    setting = {"xmid":xmid, "ymin":ymin} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goup')
    xRange = _extract_xRange(diag, proSignal, mstate.xmid, mstate.ymin)
    yRange = _extract_yRange(diag, proSignal, mstate.xmid, mstate.ymin)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    return mstate

def _goleftup(diag, proSignal):
    xmax = diag.MeasWindows[-1].mstate.xmid
    ymid = diag.MeasWindows[-1].mstate.ymax
    setting = {"xmax":xmax, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goleft')
    xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymid)
    yRange = 2*_extract_yRange(diag, proSignal, mstate.xmax, mstate.ymid)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    return mstate

def _goleft(diag, proSignal):
    xmax = diag.MeasWindows[-1].mstate.xmin
    ymid = diag.MeasWindows[-1].mstate.ymid
    setting = {"xmax":xmax, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goleft')
    xRange = _extract_xRange(diag, proSignal, mstate.xmax, mstate.ymid)
    yRange = _extract_yRange(diag, proSignal, mstate.xmax, mstate.ymid)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    return mstate

def _goright(diag, proSignal):
    xmin = diag.MeasWindows[-1].mstate.xmax
    ymid = diag.MeasWindows[-1].mstate.ymid
    setting = {"xmin":xmin, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goright')
    xRange = _extract_xRange(diag, proSignal, mstate.xmin, mstate.ymid)
    yRange = _extract_yRange(diag, proSignal, mstate.xmin, mstate.ymid)
    mstate._update_xRange(xRange)
    mstate._update_yRange(yRange)
    mstate._update_xy()
    mstate = _check_xmax(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    return mstate

def _last_measurement(diag, proSignal):
    ymid = (diag.tlist[0].vStart_y+diag.tlist[0].vStop_y)/2.
    xmid = (diag.tlist[0].vStart_x+diag.tlist[0].vStop_x)/2.
    setting = {"xmid":xmid, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    xRange = _extract_xRange(diag, proSignal, mstate.xmid, mstate.ymid)
    yRange = _extract_yRange(diag, proSignal, mstate.xmid, mstate.ymid)
    mstate._update_xRange(xRange)
    if yRange < .4:
        mstate._update_yRange(.4)
        mstate._update_xy()
    else:
        mstate._update_yRange(yRange)
        mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    xRange_trans = 2*max([(mstate.ymax-diag.tlist[0].intercept)/diag.tlist[0].slope-mstate.xmid, (mstate.ymin-diag.tlist[0].intercept)/diag.tlist[0].slope-xmid])
    if xRange_trans > mstate.xRange:
        mstate._update_xRange(diag.xResol*40 + xRange_trans)
        mstate._update_xy()
    mstate = _check_xmin(diag, mstate)
    mstate = _check_ymax(diag, mstate)
    mstate._update_step('last')
    return mstate

def _check_xmax(diag, mstate):
    if mstate.xmax > diag.xMax:
        mstate.xmax, mstate.xx = diag.xMax, 'xmax'
        mstate._update_xy()
    return mstate

def _check_xmin(diag, mstate):
    if mstate.xmin < diag.xMin:
        mstate.xmin, mstate.xx = diag.xMin, 'xmin'
        mstate._update_xy()
    return mstate

def _check_ymax(diag, mstate):
    if mstate.ymax > diag.yMax:
        mstate.ymax, mstate.yy = diag.yMax, 'ymax'
        mstate._update_xy()
    return mstate

def _check_ymin(diag, mstate):
    if mstate.ymin < diag.yMin:
        mstate.ymin, mstate.yy = diag.yMin, 'ymin'
        mstate._update_xy()
    return mstate

def _range_x(xdata, ydata):
    freq = sp._frequency_estimation(xdata, ydata)
    return 2./(freq)

def _range_y(xdata, ydata):
    freq = sp._frequency_estimation(xdata, ydata)
    return .5/freq

def _resolution_x(diag):
    return diag.xResol

def _resolution_y(diag):
    return diag.yResol
