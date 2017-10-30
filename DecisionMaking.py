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

class DotFeature:
    """

    """
    
    def __init__(self, name):
        self.name = name
        self.glist = []

    def dot_gate(self, gate):
        self.dGate = gate
#        self.glist.append(gate)

    def reservoir_gate(self, gate):
        self.rGate = gate
#        self.glist.append(gate)

    def add_gate(self, gate):
        self.glist.append(gate)

    def build_diagram(self, xResol, yResol):
        self.Diagram = None

class SETFeature:
    """

    """
    
    def __init__(self, name):
        self.name = name
        self.glist = []

    def acc_gate(self, gate):
        self.accGate = gate
#        self.glist.append(gate)

    def add_gate(self, gate):
        self.glist.append(gate)


class Device:
    """
    
    """
    
    def __init__(self):
        self.glist = []
        self.flist = []
        self.state = None
#        self.DotResDiagram = None

    def update_state(self):
        pass #update the state of the device for voltages-tracking purposes

###################################################################################################################################

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
    
        xNPoints = int((self.xMax-self.xMin)/self.xResol+1)
        yNPoints = int((self.yMax-self.yMin)/self.yResol+1)
        self.grid = np.zeros((yNPoints, xNPoints))
        self.grid[0,0], self.grid[-1,0], self.grid[0,-1], self.grid[-1,-1] = -1, -1, -1, -1 #done to always have leftover and avoid fitting mistake from an empty leftover cluster in ImageDetection loop
        self.xData = np.linspace(self.xMin, self.xMax, xNPoints)
        self.yData = np.linspace(self.yMin, self.yMax, yNPoints)

        self.MeasWindows = []
        self.clist = []
        self.leftover = []
        self.tlist = []
        self._steps = []
        self.MeasWindows, self.clist, self.leftover, self.tlist, self._steps = np.array(self.MeasWindows), np.array(self.clist), np.array(self.leftover), np.array(self.tlist), np.array(self._steps)
        

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
        for u in self.MeasWindows:
            u.ProcessedImage._proSignal._plot_box()

    def _gen_pro_signal(self, _yFlip=False):
        proSignal = sp.ProcessedSignal(self.xData, self.yData, self.grid*0.)
        proSignal.transition.zData = deepcopy(self.grid)
        if _yFlip == True:
            proSignal._yFlip = True
        return proSignal

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
        self.xRange = 0.05
        self.yRange = 0.1
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

def _meas_printer(mstate):
    mstate._update_xy()
    print "Next measurement fromx = %(xmin)s to x = %(xmax)s and y = %(ymin)s to y = %(ymax)s" %{"xmin":mstate.xmin, "xmax":mstate.xmax, "ymin":mstate.ymin, "ymax":mstate.ymax}

def next_step(diag, _yFlip = False):
    if np.size(diag.tlist) == 0:
        if np.size(diag.MeasWindows) == 0:
            mstate = _first_measurement(diag)
            return mstate
        else:
            mstate = _find_any_transition(diag, ind=-1)
            return mstate
    else:
        verif = []
        for u in diag.tlist:
            verif.append(u._tested_flag)
        verif = np.array(verif)
        if np.all(verif==False):
            if np.any(diag._steps == 'findanytrans'):
                ind = max(np.where(diag._steps == 'findanytrans')[0])
            else:
                ind = np.where(diag._steps == 'init')
            mstate = _find_any_transition(diag, ind=ind)
            return mstate
        elif np.any(verif==True):
            if diag.MeasWindows[-1].mstate._step == 'confirmtrans' or (diag.MeasWindows[-1].mstate._step == 'goup' and np.size(diag.MeasWindows[-1].ProcessedImage.transitions) == 0):
                diag.update_lists(_yFlip = _yFlip)
            if diag.MeasWindows[-1]._trans_found == True:
                if diag.MeasWindows[-1].mstate.ymax < diag.yMax:
                    mstate = _goup(diag)
                    return mstate
                else:
                    if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goup':
                        mstate = _goleftup(diag)
                        return mstate
                    if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goleft':
                        mstate = _goleft(diag)
                        return mstate
                    else:
                        if diag.MeasWindows[-1].mstate.xmax < diag.xMax:
                            mstate = _goright(diag)
                            return mstate
                        else:
                            print "last transition was found.  all that's needed is to perform voltage addition on last slice of meas."
            elif diag.MeasWindows[-1]._trans_found == False:
                if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goup':
                    mstate = _goleftup(diag)
                    return mstate
                if diag.MeasWindows[-1].mstate.xmin > diag.xMin and diag.MeasWindows[-1].mstate._step == 'goleft':
                    mstate = _goleft(diag)
                    return mstate
                else:
                    if diag.MeasWindows[-1].mstate.xmax < diag.xMax:
                        mstate = _goright(diag)
                        return mstate
                    else:
                        diag.update_lists(_yFlip = _yFlip)
                        print "last transition was found.  all that's needed is to perform voltage addition on last slice of meas."
        elif np.any(verif==None):
            ind = np.where(verif == None)[0][0]
            diag.tlist[ind]._update_test_flag(False)
            mstate = _confirm_transition(diag, ind)
            return mstate




def _first_measurement(diag):
    ymid = (diag.yMin+diag.yMax)/2.
    xmax = diag.xMax
    setting = {"ymid":ymid, "xmax":xmax}
    mstate = MeasState(diag, **setting)
    mstate._update_step('init')
    return mstate

def _find_any_transition(diag, ind=-1):
    xmax = diag.MeasWindows[-1].mstate.xmin
    ymin = diag.MeasWindows[-1].mstate.ymax
    setting = {"ymin":ymin, "xmax":xmax} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('findanytrans')
    mstate._update_xRange(diag.MeasWindows[-1].mstate.xRange)
    mstate._update_yRange(diag.MeasWindows[-1].mstate.yRange)
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
        elif np.size(ind)%2 == 0:
            ymax = diag.MeasWindows[ind[-2]].ProcessedImage._proSignal.yStart - 1.5*diag.MeasWindows[ind[-2]].mstate.yRange
            setting = {"ymax":ymax, "xmax":xmax} 
        else:
            ymin = diag.MeasWindows[ind[-2]].ProcessedImage._proSignal.yStop + 1.5*diag.MeasWindows[ind[-2]].mstate.yRange
            setting = {"ymin":ymin, "xmax":xmax} 
        mstate = MeasState(diag, **setting)
        return mstate
    else:
        return mstate

def _confirm_transition(diag, ind):
    xmid = (diag.tlist[0].vStart_x + diag.tlist[0].vStop_x)/2.
    ymid = (diag.tlist[0].vStart_y + diag.tlist[0].vStop_y)/2.
    setting = {"xmid":xmid, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('confirmtrans')
    mstate._update_xRange(diag.MeasWindows[-1].mstate.xRange)
    mstate._update_yRange(diag.MeasWindows[-1].mstate.yRange)
    if diag.MeasWindows[-1].mstate.yRange < .4:
        mstate._update_yRange(.4)
    else:
        mstate._update_yRange(diag.MeasWindows[-1].mstate.yRange)
    xRange = 2*max([(mstate.ymax-diag.tlist[ind].intercept)/diag.tlist[ind].slope-mstate.xmid, (mstate.ymin-diag.tlist[ind].intercept)/diag.tlist[ind].slope-xmid])
    if xRange > mstate.xRange:
        mstate._update_xRange(diag.xResol*40 + xRange)
    mstate._update_xy()
    if diag.xMin > mstate.xmin:
        mstate.xmin, mstate.xx = diag.xMin, 'xmin'
        mstate._update_xy()
    if diag.yMax < mstate.ymax:
        mstate.ymax, mstate.yy = diag.yMax, 'ymax'
        mstate._update_xy()
    return mstate   

def _goup(diag):
    ymin = diag.MeasWindows[-1].mstate.ymax
    xmid = (ymin-diag.MeasWindows[-1].ProcessedImage.transitions[0].intercept)/diag.MeasWindows[-1].ProcessedImage.transitions[0].slope
    setting = {"xmid":xmid, "ymin":ymin} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goup')
    mstate._update_xy()
    if mstate.ymax > diag.yMax:
        mstate.ymax = diag.yMax
        mstate.yy = 'ymax'
        mstate._update_xy()
    if mstate.xmin < diag.xMin:
        mstate.xmin = diag.xMin
        mstate.xx = 'xmin'
        mstate._update_xy()
    return mstate

def _goleftup(diag):
    xmax = diag.MeasWindows[-1].mstate.xmin
    ymid = diag.MeasWindows[-1].mstate.ymax
    setting = {"xmax":xmax, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goleft')
    mstate._update_yRange(mstate.yRange*2)
    mstate._update_xy()
    if mstate.xmin < diag.xMin:
        mstate.xmin = diag.xMin
        mstate.xx = 'xmin'
        mstate._update_xy()
    return mstate

def _goleft(diag):
    xmax = diag.MeasWindows[-1].mstate.xmin
    ymid = diag.MeasWindows[-1].mstate.ymid
    setting = {"xmax":xmax, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goleft')
    mstate._update_xy()
    if mstate.xmin < diag.xMin:
        mstate.xmin = diag.xMin
        mstate.xx = 'xmin'
        mstate._update_xy()
    return mstate

def _goright(diag):
    xmin = diag.MeasWindows[-1].mstate.xmax
    ymid = diag.MeasWindows[-1].mstate.ymid
    setting = {"xmin":xmin, "ymid":ymid} 
    mstate = MeasState(diag, **setting)
    mstate._update_step('goright')
    mstate._update_xy()
    if mstate.xmax > diag.xMax:
        mstate.xmax = diag.xMax
        mstate.xx = 'xmax'
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

def _confirm_if_last():
    pass































































