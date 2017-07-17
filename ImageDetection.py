# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:21 2016
@author: Maxime
"""

import numpy as np
#from skimage import transform
#from skimage import measure
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from itertools import product, izip
from numpy.linalg import eig
from copy import deepcopy
import lmfit as lmf
import matplotlib.pyplot as plt

#from scipy.optimize import curve_fit

class Segment:
    """
    Segment obtained from the Hough Transform of an ensemble of points.
    
    Attributes
    -----------
    p0, p1  \t Start and end points of the segment -- each point is an size 2 array for (x,y)
    x0, x1  \t X position of the start and end points of the segment
    y0, y1  \t Y position of the start and end points of the segment
    a       \t Slope of the segment
    b       \t Intersect of the segment at x=0 -- calculated even if the segment does not cross the y axis
    theta_x \t Angle between the segment and the x axis (in degrees).  theta_x goes from 0 to 180 degrees.  Positive slope will always give theta_x < 90 and negative slope will give theta_x > 90
    
    
    """
    def __init__(self, p0=None, p1=None, a=None, b=None):
        if a!=None:
            self.a = a
        else:
            self.a = float(p1[1]-p0[1])/float(p1[0]-p0[0])
        
        if b!=None:
            self.b = b
        else:
            self.b = float(p0[1])-self.a*float(p0[0])
        
        if p1[0]<p0[0]:
            p0, p1 = p1, p0
        self.p0 = p0
        if p0!=None:
            self.x0 = p0[0]
            self.y0 = p0[1]
        self.p1 = p1
        if p1!=None:
            self.x1 = p1[0]
            self.y1 = p1[1]
        
        self.theta_x = AngleLineXAxis(self.a)

class ProcessedImage:
    """
    
    """
    def __init__(self, ProcessedSignal, min_cluster_size = 5, max_gap_size = 0):
        self.min_cluster_size = min_cluster_size
        self.max_gap_size = max_gap_size
        self._proSignal = ProcessedSignal

        cc = Linkage(ProcessedSignal.transition.zData, gap_size = max_gap_size, min_cluster_size = min_cluster_size)
        cc = Initialization(cc, max_gap = max_gap_size, _proImage = self)

        self.clusters = cc

        self.transitions, self.leftover_clusters = Transitions(cc, (self._proSignal.yNPoints, self._proSignal.xNPoints), self)

    def PlotClusters(self):
        for u in self.clusters:
            u.plot()



    def PlotTransitions(self):
        for u in self.transitions:
            u.plot()
        pass


class Cluster:
    """
    The Cluster class is used to extract useful information of a cluster for further processing
    The __init__ takes the y-x coordinates of all the points of the cluster (in the form (:,2)) and the maximum gap size separating points apart of that cluster
    
    Attributes
    -----------
    cluster  . . . . .\t coordinates in the shape (:,2) of all points that are part of the cluster.
    ccx, ccy . . . . .\t x and y coordinates of the cluster.  Array-like.
    nPoints  . . . . .\t number of points
    ratio  . . . . . .\t ratio of height/length of the cluster in the rotated frame
    length . . . . . .\t length of the cluster
    p0 . . . . . . . .\t center position of the cluster (mean of y coordinates and x coordinates)
    u  . . . . . . . .\t unitary directional vector of the cluster.  u_x always > 0
    v  . . . . . . . .\t unitary vector perpendicular to u.  v_y always > 0

    theta_y  . . . . .\t angle between the negative y axis and the directional vector of the cluster, u.
    rho  . . . . . . .\t shortest distance between the origin (top left) of the diagram and the line passing through p0 along u.
    slope  . . . . . .\t slope of the line passing through p0 along the directional vector u in a pixel-pixel axis diagram.
    intercept  . . . .\t intercept of the line passing through p0 along the directional vector u in a pixel-pixel axis diagram.
    
    sigma_theta  . . .\t standard deviation on the theta_y variable as calculated in [1]
    sigma_rho  . . . .\t standard deviation on the rho variable as calculated in [1]
    sigma_rho_theta  .\t covariance of the rho and theta variables as calculated in [1]

    corr_sigma_theta .\t corrected sigma theta based on the non-collinearity of the points of the cluster
    corr_sigma_rho . .\t -- unavailable --
    corr_sigma_rt  . .\t -- unavailable --
    corr_x . . . . . .\t x axis coordinates of the points of the cluster in the rotated frame (nul-slope)
    corr_y . . . . . .\t y axis coordinates of the points of the cluster in the rotated frame (nul-slope)
    
    start  . . . . . .\t Corner of the smallest window containing the cluster with the smallest x and y coordinates.
    stop . . . . . . .\t Corner of the smallest window containing the cluster with the highest x and y coordinates.
    seg_start  . . . .\t start coordinates of the best segment identified in the cluster.  (y,x) shaped.  The start is the smallest x-coordinate
    seg_stop . . . . .\t end coordinates of the best segment identified in the cluster.  (y,x) shaped.  The end is the highest x-coordinate

    xSegment . . . . .\t x coordinates of the best segment identified in the cluster.  array-like
    ySegment . . . . .\t y coordinates of the best segment identified in the cluster.  array-like

    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.
    """
    def __init__(self, cluster, max_gap, _proImage = None):
        if _proImage != None:
            self._proSignal = _proImage._proSignal
        
        p0, u, v = u_v_(cluster)
        if v[0]<0.:
            v = -v
        if u[1]<0.:
            u = -u
        p0, u, v = _correction_uv(cluster, u, v, p0)
        if v[0]<0.:
            v = -v
        if u[1]<0.:
            u = -u
        cp = _rotate(cluster, u, v, p0)
        length, ratio = _ratio(cp)
        nPoints = cluster.T[0].shape[0]
        

        self.cluster = cluster
        self.ccx = cluster.T[1]
        self.ccy = cluster.T[0]
        self.nPoints = cluster.T[0].shape[0]
        self.ratio = ratio
        self.length = length
        self.p0 = p0
        self.u = u
        self.v = v
        self.p0_y = p0[0]
        self.p0_x = p0[1]
        self.u_y = u[0]
        self.u_x = u[1]
        self.v_y = v[0]
        self.v_x = v[1]
        
        rho = _rho(v, p0)
        theta_y = _theta_y(v)
        
        self.rho = rho
        self.theta_y = theta_y
        
        sigma_rho_sq, sigma_theta_sq, sigma_rho_theta = Sigmas(cluster, p0, u, v)
        
        self.sigma_mp_sq = _sigma_mp_square(cluster, p0, u)
        self.sigma_bp_sq = _sigma_bp_square(cluster.shape[0])
        self.sigma_rho_sq = sigma_rho_sq
        self.sigma_theta_sq = sigma_theta_sq
        self.sigma_rho_theta = sigma_rho_theta
        self.sigma_theta = np.sqrt(sigma_theta_sq)
        self.sigma_rho = np.sqrt(sigma_rho_sq)

        correl_coeff = sigma_rho_theta/np.sqrt(sigma_theta_sq*sigma_rho_sq)
        correction = nPoints*ratio/(1.+max_gap)
        cp = _rotate(cluster, u, v, p0)

        corr_sigma_theta = np.sqrt(sigma_theta_sq)*correction
        if corr_sigma_theta > np.pi/2:
            corr_sigma_theta= np.pi/2

        self.correl_coeff = correl_coeff
        self.correction_coeff = correction
        self.corr_sigma_theta = corr_sigma_theta
        self.corr_sigma_rho = np.sqrt(sigma_rho_sq)*correction
        self.corr_sigma_rt = sigma_rho_theta*correction*correction
        self.corr_x = cp[1]
        self.corr_y = cp[0]
        


        start, stop = _start_stop(cp[1], cp[0], u, v, p0)
        self.start = start
        self.start_x = start[1]
        self.start_y = start[0]
        self.stop = stop
        self.stop_x = stop[1]
        self.stop_y = stop[0]

        aa = -Slope(90-theta_y*180./np.pi)
        bb = -aa*p0[1]+p0[0]
        self.slope = aa
        self.intercept = bb

        segment_start, segment_stop = _segment_start_stop(cluster.T[1], cluster.T[0], aa, bb)
        self.seg_start = segment_start
        self.seg_stop = segment_stop
        self.seg_start_x = segment_start[1]
        self.seg_start_y = segment_start[0]
        self.seg_stop_x = segment_stop[1]
        self.seg_stop_y = segment_stop[0]

        xSegment = np.array((segment_start[1], segment_stop[1]))
        ySegment = np.array((segment_start[0], segment_stop[0]))
        self.xSegment = xSegment
        self.ySegment = ySegment

    def copy(self):
        return deepcopy(self)

    def plot(self):
        if self._proSignal == None:
            raise NotImplementedError
        else:
            if self._proSignal._yFlip == False:
                plt.plot(self.xSegment/float(self._proSignal.xNPoints-1)*(self._proSignal.xStop-self._proSignal.xStart)+self._proSignal.xStart, self.ySegment/float(self._proSignal.yNPoints-1)*(self._proSignal.yStop-self._proSignal.yStart)+self._proSignal.yStart, '-')
            else:
                plt.plot(self.xSegment/float(self._proSignal.xNPoints-1)*(self._proSignal.xStop-self._proSignal.xStart)+self._proSignal.xStart, self.ySegment/float(self._proSignal.yNPoints-1)*(self._proSignal.yStart-self._proSignal.yStop)+self._proSignal.yStop, '-')


class Transition:
    """
    The transition class is used to regroup Clusters together and identify the position of lines (therefore transitions)
    
    The __init__ takes a Cluster-class object and a Transition-class object as an input.
    If the cluster you are adding to the transition is the first, transition must be None.
    Afterwards, you can expand the transition by adding clusters that belong to that transition by passing the transition object that already exists in addition to the cluster that's being added.
    
    Attributes    
    -----------
    clusters . . .\t list of Cluster objects that belong to the Transition object
    coord  . . . .\t array-like. (y, x) coordinates of all the points that belong to this transition
    ccx, ccy . . .\t array-likes.  x and y coordinates of all the points that belong to this transition
  
    p0s  . . . . .\t array-like.  List of all center points of clusters that belong to this transition
    sStart . . . .\t (y, x) coordinate of the start point of the first cluster
    sStop  . . . .\t (y, x) coordinate of the end point of the last cluster
    
    xSegment . . .\t array-like.  List of x-coordinate of the points that approximate the best the transition.  It is built with (sStart_x, p0s_x, sStop_x)
    ySegment . . .\t array-like.  List of y-coordinate of the points that approximate the best the transition.  It is built with (sStart_y, p0s_y, sStop_y)
    """
    
    def __init__(self, cluster, transition=None, _proImage = None):
        if _proImage != None:
            self._proSignal = _proImage._proSignal
        
        clist = []
        if transition == None:
            clist.append(cluster)
        else:
            dist0 = (cluster.p0_x-transition.sStart_x)*(cluster.p0_x-transition.sStart_x)+(cluster.p0_y-transition.sStart_y)*(cluster.p0_y-transition.sStart_y)
            dist1 = (cluster.p0_x-transition.sStop_x)*(cluster.p0_x-transition.sStop_x)+(cluster.p0_y-transition.sStop_y)*(cluster.p0_y-transition.sStop_y)
            if dist0 < dist1:
                clist.append(cluster)
                for u in transition.clusters:
                    clist.append(u)
            else:
                for u in transition.clusters:
                    clist.append(u)
                clist.append(cluster)
        
        self.clusters = clist
        
        coord = []
        for u in clist:
            for i in u.cluster:
                coord.append(i)
        coord = np.array(coord)
        
        self.coord = coord
        self.ccx = coord.T[1]
        self.ccy = coord.T[0]

        p0 = []
        p0_x = []
        p0_y = []
        for u in clist:
            p0.append(u.p0)
            p0_x.append(u.p0_x)
            p0_y.append(u.p0_y)
        
        self.p0s = p0
        self.p0_xs = p0_x
        self.p0_ys = p0_y
        
        self.sStart = clist[0].seg_start
        self.sStart_x = clist[0].seg_start_x
        self.sStart_y = clist[0].seg_start_y
        self.sStop = clist[-1].seg_stop
        self.sStop_x = clist[-1].seg_stop_x
        self.sStop_y = clist[-1].seg_stop_y
        
        xSegment = []
        ySegment = []
        
        xSegment.append(clist[0].seg_start_x)
        for u in p0_x:
            xSegment.append(u)
        xSegment.append(clist[-1].seg_stop_x)
        
        ySegment.append(clist[0].seg_start_y)
        for u in p0_y:
            ySegment.append(u)
        ySegment.append(clist[-1].seg_stop_y)
        
        self.xSegment = np.array(xSegment)
        self.ySegment = np.array(ySegment)
        
        self.cStart = clist[0]
        self.cStop = clist[-1]
        
        mean_theta = []
        for u in self.clusters:
            mean_theta.append(u.theta_y)
        self.mean_theta = np.mean(mean_theta)

    def plot(self):
        if self._proSignal == None:
            raise NotImplementedError
        else:
            if self._proSignal._yFlip == False:
                plt.plot(self.xSegment/float(self._proSignal.xNPoints-1)*(self._proSignal.xStop-self._proSignal.xStart)+self._proSignal.xStart, self.ySegment/float(self._proSignal.yNPoints-1)*(self._proSignal.yStop-self._proSignal.yStart)+self._proSignal.yStart, '-')
            else:
                plt.plot(self.xSegment/float(self._proSignal.xNPoints-1)*(self._proSignal.xStop-self._proSignal.xStart)+self._proSignal.xStart, self.ySegment/float(self._proSignal.yNPoints-1)*(self._proSignal.yStart-self._proSignal.yStop)+self._proSignal.yStop, '-')
        
    def copy(self):
        return deepcopy(self)
        
    def linear_fit(self):
        cx, cy = average_y(self.ccx, self.ccy)
        model = lmf.models.LinearModel()
        ret = model.fit(cy, x=cx, slope=1, intercept=0)
        self.pixel_slope, self.pixel_intercept = ret.values.get('slope'), ret.values.get('intercept')
        if self._proSignal._yFlip == True:
            self.slope = self.pixel_slope/(self._proSignal.yNPoints-1)*(self._proSignal.yStart-self._proSignal.yStop)*self._proSignal.xNPoints/(self._proSignal.xStop-self._proSignal.xStart)
            self.intercept = self.pixel_intercept/float(self._proSignal.yNPoints-1)*(self._proSignal.yStart-self._proSignal.yStop)+self._proSignal.yStop-self._proSignal.xStart*self.slope
        else:
            self.slope = self.pixel_slope/self._proSignal.yNPoints*(self._proSignal.yStop-self._proSignal.yStart)*self._proSignal.xNPoints/(self._proSignal.xStop-self._proSignal.xStart)
            self.intercept = self.pixel_intercept/self._proSignal.yNPoints*(self._proSignal.yStop-self._proSignal.yStart)+self._proSignal.yStart+self._proSignal.xStart*self.slope
            


def Initialization(cc, max_gap = 0, _proImage = None):
    """
    
    """
    cc = list(cc)
    leftover = cc.pop(-1)
    clust = []
    for u in cc:
        if _proImage != None:
            temp = Cluster(u, max_gap, _proImage = _proImage)
        else:
            temp = Cluster(u, max_gap)
        ret = _split(temp.corr_x, temp.corr_y, temp.u, temp.v, temp.p0, temp.ratio)
        if ret == None:
            clust.append(temp)
        else:
            cc.append(ret[0])
            cc.append(ret[1])
    clust.append(Cluster(leftover, max_gap, _proImage = _proImage))
    return clust

def DistancePointLine(x, y, a, b):
    """
    Calculates the closest distance between a line and a point
    -------
    (x,y) are the coordinates of the point and 
    (a,b) are the slope and intersect of the line
    """
    return abs(a*x-y+b)/np.sqrt(1+a**2)

def AngleLineXAxis(a):
    """
    Calculates the angle between a line and the x axis.
    -------
    a is the slope of the line
    
    returns the angle in degrees.
    """
    return np.arctan(a)*180./np.pi

def Slope(angle):
    """
    Calculates the slope of a line given the angle between this line and the x axis.
    -------
    the angle must be in degrees.
    """
    return np.tan(angle*np.pi/180.)


def Linkage(image, min_cluster_size=4, gap_size=None):
    """
    Returns a list of clusters each containing a minimum size with all points separated by at most a specified number of pixels
    
    The last cluster is the ensemble of points that belong to no cluster. (index[-1])
    Each cluster is an array of coordinates of the form [y, x], assuming that the input image is an array of arrays with the first index corresponding to the y axis.
    -------
    image           \t must be a binary image (array of arrays).  Points to be clustered must have a value different than 0.
    gap_size        \t is the maximum number of pixels separating points that belong to a cluster.  Must be an integer
    min_cluster_size\t is the minimum ammount of points that a group of points must contain in order to be considered as a cluster.  If the group of points does not meet the requirement, they are placed in the left-over category in any order.
    """
    cc = []
    leftover = []
    img = image.copy()
    index = np.nonzero(img)
    if gap_size == None and index[0].shape < 1000: #The constant still needs to be determined.  Perhaps as density of points?
        gap_size = 1
    else:
        gap_size = 0
    while (index[0].shape[0]!=0):
        group = _group(img, ref=list([index[0][0], index[1][0]]), gap_size=gap_size)
        for u in group:
            img[u[0], u[1]] = 0.
        if group.shape[0] >= min_cluster_size:
            cc.append(group)
        else:
            for u in group:
                leftover.append(u)
        index = np.nonzero(img)
    cc.append(np.array(leftover))
    return np.array(cc)

def _group(img, ref, gap_size=0):
    """
    The _group function takes a binary image and a reference coordinate and generates an array of all the points in the image that are part of the same cluster as the reference point.

    The function returns an array of coordinates that all belong to a single cluster.

    img is a binary image where all non-zero values are pixels that belong to a cluster.
    ref is a coordinate of the reference point for which the _group function has to find all other points belonging to the same cluster.
    gap_size is the maximum size of gaps between pixels (or groups of pixels) for them to be considered as part of a same cluster.
    """
    index = 0
    mylist = []
    mylist.append(list(ref))
    image = img.copy()
    image[ref[0],ref[1]] = 0.
    while (np.array(mylist).shape[0]>index):
        ret = _next(image, mylist[index], gap_size=gap_size)
        if ret!=None:
            mylist.append(ret)
            image[ret[0], ret[1]] = 0.
        else:
            index=index+1
    return np.array(mylist)

def _next(img, ref, gap_size=0):
    """
    The _next function takes a binary image, a reference point and a gap_size as an input.
    
    The coordinate of the first non-zero point found within the gap_size of the reference point is returned.

    If none of the points around the reference are non-zero, the function returns None.
    """
    ret = None
    xmin, xmax, ymin, ymax = 1, 1, 1, 1
    if ref[0] < img.shape[0]:
        xmax = 0
    if ref[0] > 0:
        xmin = 0
    if ref[1] < img.shape[1]:
        ymax = 0
    if ref[1] > 0:
        ymin = 0
    for i in product((np.arange(ref[0]-gap_size-1+xmin, ref[0]+gap_size+2-xmax, 1))%img.shape[0], (np.arange(ref[1]-gap_size-1+ymin, ref[1]+gap_size+2-ymax, 1))%img.shape[1]):
        if img[i[0], i[1]]!=0.:
            ret = list(i)
            break
    return ret

def _segment_start_stop(xx, yy, a, b):
    """
    Returns the start and end points of a given segment parametrized by (a, b) passing through a cluster of points with coordinates (yy, xx)
    
    xx \t array-like.  X coordinates of all the points of the cluster
    yy \t array-like.  Y coordinates of all the points of the cluster
    a, b \t float.  slope and intercept of the segment for which the end and start points need to be identified
    """
    if a > 0.:
        xmax = max(xx)
        xmin = min(xx)
        ymax = max(yy)
        ymin = min(yy)
        xmax_y = a*xmax+b
        xmin_y = a*xmin+b
        if xmax == xmin:
            pass
        else:
            if xmin_y < ymin:
                xmin = (ymin-b)/a
                #ymin est bon et ymin_x est le x associé.
            elif ymin <= xmin_y:
                ymin = xmin_y.copy()
                #xmin_y est bon et xmin est le x associé.
            if xmax_y < ymax:
                ymax = xmax_y.copy()
                #xmax_y est bon est xmax est le x associé
            elif ymax <= xmax_y:
                xmax = (ymax-b)/a
                #ymax est bon et ymax_x est le x associé
    elif a < 0.:
        xmax = max(xx)
        xmin = min(xx)
        ymax = min(yy)
        ymin = max(yy)
        xmax_y = a*xmax+b
        xmin_y = a*xmin+b
        if xmax == xmin:
            pass
        else:
            if xmin_y > ymin:
                xmin = (ymin-b)/a
                #ymin est bon et ymin_x est le x associé.
            elif ymin >= xmin_y:
                ymin = xmin_y.copy()
                #xmin_y est bon et xmin est le x associé.
            if xmax_y > ymax:
                ymax = xmax_y.copy()
                #xmax_y est bon est xmax est le x associé
            elif ymax >= xmax_y:
                xmax = (ymax-b)/a
                #ymax est bon et ymax_x est le x associé
    elif a == 0.:
        xmax = max(xx)
        xmin = min(xx)
        ymax = a*xmax+b
        ymin = ymax.copy()
    return np.array((ymin, xmin)), np.array((ymax, xmax))

def _backrotate(cc, u, v, p0):
    """
    Assuming an initial cluster (c) with center p0, directional vector u and its perpendicular vector v that was translated and rotated so that p0=(0,0) and u_y=0, u_x=1 into coordinates cc (of the shape (y,x))
    _backrotate() will transform the primed referential coordinates, cc, into the initial coordinates, c.
    
    returns the (y,x) coordinates of the initial non-primed referential.
    """
    cpx = np.zeros(cc[0].shape[0], dtype=np.int32)
    cpy = np.zeros(cc[0].shape[0], dtype=np.int32)
    index = np.linspace(0, cc[0].shape[0]-1, cc[0].shape[0], dtype=np.int32)
    for i, j, k in izip(cc[0], cc[1], index):
        cpx[k] = int(round(u[1]*j-u[0]*i+p0[1]))
        cpy[k] = int(round(-v[1]*j+v[0]*i+p0[0]))
    cp = np.array((cpy, cpx))
    return cp.T

def _start_stop(corrx, corry, u, v, p0):
    """
    Returns the start and end points of a segment that passes through a cluster.
    
    The coordinates of both these 2 points are the intersect of the smallest window containing the entire cluster and the line passing through p0 along the unitary vector u.
    
    corrx \t array-like. X coordinates of all the points of the cluster
    corry \t array-like. Y coordinates of all the points of the cluster
    u \t (y, x) coordinates.  Unitary directional vector of the cluster
    v \t (y, x) coordinates.  Perpendicular vector to u, also unitary.
    p0 \t euclidian center of the cluster (p0_x = mean(corrx), p0_y = mean(corry))
    """
    index = np.argsort(corrx)
    x0 = corrx[index[0]]
    x1 = corrx[index[-1]]
    y0 = corry[index[0]]
    y1 = corry[index[-1]]
    xstart = int(round(u[1]*x0-u[0]*y0+p0[1]))
    ystart = int(round(-v[1]*x0+v[0]*y0+p0[0]))
    xstop = int(round(u[1]*x1-u[0]*y1+p0[1]))
    ystop = int(round(-v[1]*x1+v[0]*y1+p0[0]))
    if ystart < ystop:
        start = np.array((ystart, xstart))
        stop = np.array((ystop, xstop))
    else:
        stop = np.array((ystart, xstart))
        start = np.array((ystop, xstop))
    return start, stop

def _rotate(cluster, u, v, p0):
    """
    The _rotate function takes a cluster of points and its u, v and p0 vectors and makes a translation (of -p0) followed by a rotation (of pi/2-arccos(v_x)) of the cluster.

    cluster is an array of the coordinates that need to be translated and rotated.
    u is the vector along which the x axis will be after the rotation.
    v is the vector along which the y axis will be after the rotation.
    -p0 is the vector of translation that is applied prior to the rotation.  p0 is typically the middle of the cluster that needs to be rotated.
    """
    clust = cluster.copy()
    clust = clust.T
    cp = np.zeros(clust.shape)
    cp[0] = v[1]*(clust[1]-p0[1])+v[0]*(clust[0]-p0[0])
    cp[1] = u[1]*(clust[1]-p0[1])+u[0]*(clust[0]-p0[0])
    return cp

def u_v_(cluster):
    """
    Takes a cluster of coordinates and calculates the euclidian average and direction vector
    
    returns p0, u, v
    -------
    p0 is the "middle" of the cluster.
    u is the directional unitary vector of the cluster.
    v is the unitary vector perpendicular to u.
    
    p0, u, v are all of the form (y,x) in order to be consistent with the usual matrix definition where the first indice is the y axis.
    """
    p0x = np.mean(cluster.T[1])
    p0y = np.mean(cluster.T[0])
    p0 = np.array([p0y, p0x])
    eigmax = -1.0e100
    for u in cluster:
        x = u[1]-p0x
        y = u[0]-p0y
        xy = x*y
        mm = np.array([[x*x, xy],[xy, y*y]])
        eigvalue, eigvector = eig(mm)
        for i, k in enumerate(eigvalue):
            if k>eigmax:
#                pt = u.copy()
                eigmax = k
                vecmin = eigvector[i-1]
                vecmax = eigvector[i]
    ### The eig() function of numpy currently has a problem with associating the proper eigenvector with its eigenvalue.
    ### The eigenvectors are also inverted ([y,x] instead of [x,y]), which in our case fits our need, but could cause problems to a user.
    u = vecmin[np.array([0,1])].copy()
    v = vecmax[np.array([0,1])].copy()
    return p0, u, v

def _correction_uv(cluster, u, v, p0):
    """
    The _correction_uv function takes a cluster of points and its u, v and p0 vectors, applies the translation followed by rotation specified by these 3 vectors using the _rotate() function before applying a linear fit to the rotated set of data.
    The slope of the fit is extracted and the angle of rotation (so the u and v vectors) are corrected so that when applied, the slope of the cluster is zero.
    """
    cp = _rotate(cluster, u, v, p0)
    index = np.argsort(cp[1])
    mod = lmf.models.LinearModel()
    out = mod.fit(cp[0, index], x=cp[1, index])
    slope = out.values.get('slope')
    theta_y = _theta_y(v)+AngleLineXAxis(slope)*np.pi/180. # corrected theta_y + corr to theta_y - corr  as of 20170703
    u[0] = np.cos(theta_y)
    u[1] = -np.sin(theta_y)
    v[1] = u[0].copy()
    v[0] = -u[1].copy()
#    cp = _rotate(cluster, u, v, p0)
#    length, ratio = _ratio(cp)
#    mod = lmf.models.LinearModel()
#    mod.set_param_hint('slope', value=0.)
#    out = mod.fit(cp[0], x=cp[1])
#    covar = out.covar
    return p0, u, v#, ratio, length, covar

def _ratio(cp):
    """
    returns the ratio of height/length of a rotated cluster, cp.
    """
    length = max(cp[1])-min(cp[1])
    ratio = (max(cp[0])-min(cp[0])+1)/length
    return length, ratio



def _rho(v, p0):
    """
    returns the shortest distance to the origin of the line perpendicular to the unitary vector v and passing through p0
    """
    return v[0]*p0[0]+v[1]*p0[1]

def _theta_y(v):
    """
    returns the angle between the negative y axis and the directional vector u.

    v is the unitary vector perpendicular to the directional vector.
    """
    return np.arccos(v[1])

def _sigma_mp_square(cluster, p0, u):
    """
    returns the variance of the slope as defined in [1]
    
    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.
    """
    ss = 0.
    for j in cluster:
        ss = ss+((u[1]*(j[1]-p0[1])+u[0]*(j[0]-p0[0]))**2)
    ss = 1./ss
    return ss

def _sigma_bp_square(n):
    """
    returns the variance of the intercept as defined in [1]
    
    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.
    """
    return 1./n

def _matrixM(p0, u, v, sigma_mp_sq, sigma_bp_sq):
    """
    returns the covariance matrix of the slope-intercept as defined in [1]
    
    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.
    """
    aa = -u[1]*p0[1]-u[0]*p0[0]
    bb = u[1]/np.sqrt(1-v[1]*v[1])
    matrix = np.zeros((2,2))
    matrix[0][0] = aa*aa*sigma_mp_sq + sigma_bp_sq
    matrix[0][1] = aa*bb*sigma_mp_sq
    matrix[1][0] = matrix[0,1].copy()
    matrix[1][1] = bb*bb*sigma_mp_sq
    return matrix

def _sigmas(p0, u, v, sigma_mp_sq, sigma_bp_sq):
    """
    returns the covariance matrix terms for rho-theta calculated from the slope-intercept covariance matrix terms.
    
    p0 \t euclidian center of the cluster.  (y,x) coordinates
    u \t unitary directional vector.  (y,x) coordinates    
    v \t unitary perpendicular vector to u.  (y,x) coordinates
    sigma_mp_sq \t variance of the slope in the slope-intercept covariance matrix
    sigma_bp_sq \t variance of the intercept in the slope-intercept covariance matrix
    sigma_mp_bp \t --not included--  the slope-intercept is assumed to be 0, as defined in [1]
    
    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.

    """
    matrix = _matrixM(p0, u, v, sigma_mp_sq, sigma_bp_sq)
    sigma_rho_sq = 2*2*matrix[0,0]
    if matrix[1,1]!=0.:
        sigma_theta_sq = 2*2*matrix[1,1]
    else:
        sigma_theta_sq = 2*2*.1
    sigma_rho_theta = matrix [0,1]
    return sigma_rho_sq, sigma_theta_sq, sigma_rho_theta

def Sigmas(cluster, p0, u, v):
    """
    Computes and returns the covariance matrix terms in the rho-theta basis as defined in [1]
    
    cluster \t array of (y, x) coordinates of all points that constitute the cluster for which the Hough transform must be computed
    p0 \t the euclidian center of the cluster
    u \t the unitary directional vector
    v \t the unitary vector perpendicular to u
    
    returns:
    sigma_rho_sq \t the variance of the rho coordinate
    sigma_theta_sq \t the variance of the theta coordinate
    sigma_rho_theta \t the rho-theta covariance term

    [1] Real-time line detection through an improved Hough transform voting scheme, L A.F. Fernandes, M. M. Oliveira.  Pattern Recognition.

    """
    sigma_mp_sq = _sigma_mp_square(cluster, p0, u)
    sigma_bp_sq = _sigma_bp_square(cluster.shape[0])
    sigma_rho_sq, sigma_theta_sq, sigma_rho_theta = _sigmas(p0, u, v, sigma_mp_sq, sigma_bp_sq)
    return sigma_rho_sq, sigma_theta_sq, sigma_rho_theta

def _split(corrx, corry, u, v, p0, ratio):
    """
    Splits a cluster of points into 2 clusters if at least one of the 2 subdivisions has a smaller height to length ratio than the initial cluster.

    The _split() function takes the start and end points along the x axis of the cluster and rotates the cluster such that start_y=end_y=0.
    Afterwards, it tries to split the cluster where abs(y) is maximum and looks if the ratio of either of the 2 subdivisions is smaller than the initial ratio.
    If so, it returns the 2 clusters.  If both subdivided clusters have a worse ratio than the original's, _split() returns None

    corrx \t array of the x axis coordinates of all the points part of the cluster in the translated/rotated frame such that p0'=(0,0) and u'_x=1 and u'_y=0
    corry \t array of the y axis coordinates of all the points part of the cluster in the translated/rotated frame such that p0'=(0,0) and u'_x=1 and u'_y=0
    u \t\t unitary directional vector of the cluster
    v \t\t unitary perpendicular vector to u
    p0 \t\t euclidian center of the cluster
    ratio \t height over length ratio of the cluster that must be split

    For additional information, see [2]
    
    [2] Three-dimentional object recognition from single two-dimensional images, D. G. Lowe, Artificial Intelligence.
    """
    index = np.argsort(corrx)
    a = (corry[index][-1]-corry[index][0])/(corrx[index][-1]-corrx[index][0])
    b = corry[index][0]-a*corrx[index][0]
    d = []
    for i, j in izip(corrx, corry):
        d.append(DistancePointLine(i, j, a, b))
    d = np.array(d)
    ind = np.argmax(d)
    c0 = np.array((corry[index][:ind], corrx[index][:ind]))
    c1 = np.array((corry[index][ind:], corrx[index][ind:]))
    if c0.shape[1]>4 and c1.shape[1]>4:
        p00, u0, v0 = u_v_(c0.T)
        if v0[0]<0.:
            v0 = -v0
        if u0[1]<0.:
            u0 = -u0
        p00, u0, v0 = _correction_uv(c0.T, u0, v0, p00)
        if v0[0]<0.:
            v0 = -v0
        if u0[1]<0.:
            u0 = -u0
        cp0 = _rotate(c0.T, u0, v0, p00)
        length0, ratio0 = _ratio(cp0)
        if ratio >= ratio0:
            c0 = _backrotate(c0, u, v, p0)
            c1 = _backrotate(c1, u, v, p0)
            ret = c0, c1
        elif ratio < ratio0:
            p01, u1, v1 = u_v_(c1.T)
            if v1[0]<0.:
                v1 = -v1
            if u1[1]<0.:
                u1 = -u1
            p01, u1, v1 = _correction_uv(c1.T, u1, v1, p01)
            if v1[0]<0.:
                v1 = -v1
            if u1[1]<0.:
                u1 = -u1
            cp1 = _rotate(c1.T, u1, v1, p01)
            length1, ratio1 = _ratio(cp1)
            if ratio >= ratio1:
                c0 = _backrotate(c0, u, v, p0)
                c1 = _backrotate(c1, u, v, p0)
                ret = c0, c1
            else:
                ret = None
    else:
        ret = None
    return ret

def _init_guess(cc):
    """
    returns the indice of the cluster which should be the initial cluster for grouping clusters into forming lines.  If no cluster is good enough, returns None

    The "best" cluster is the longest with an incertainty on theta_y smaller than pi/8

    cc is an array of clusters.
    """
    ss = np.array(cc).shape[0]
    ss = np.zeros((3, ss))
    for u, i in enumerate(cc):
        ss[0, u] = u
        ss[1, u] = i.length
        ss[2, u] = i.corr_sigma_theta
    s = []
    for u in ss.T:
        if u[2] < np.pi/8:
            s.append(u)
    if s!=[]:
        s = np.array(s).T
        ind = np.argmax(s[1,:])
        ind = s[0, ind]
        return int(round(ind))
    else:
        return None
  
def _rrel(nSegments, xPixels, yPixels):
    """
    returns the distance beyond which the proximity of endpoints becomes irrelevant.

    rrel stands for relevant distance.
    
    nSegments \t number of segments in the image.
    xPixels \t resolution of the image along the x axis in number of pixels
    yPixels \t resolution of the image along the y axis in number of pixels
    """
    d = float(nSegments)/float(xPixels*yPixels)
    return 1.5*np.sqrt(1./(np.pi*d))
    
def _distance(end1, end2):
    """
    returns the distance between 2 endpoints of a segment.
    
    end1, 2 \t (y, x) coordinates of the endpoints
    """
    a = end1[0]-end2[0]
    b = end1[1]-end2[1]
    return np.sqrt(a*a+b*b)
    
def _delta_theta(t1, t2):
    """
    
    """
    return abs(t1-t2)
    
def _perpendicular_dist(a, b, p0):
    """
    
    """
    d0 = DistancePointLine(p0[1], p0[0], a, b)
    return d0

def _parallelism(theta, s, l1, l2):
    """
    
    """
    return 4*theta*s*l2/(np.pi*l1*l1)


def _collinearity(theta, s, g, l1):
    """
    
    """
    return 4*theta*s*(g+l1)/(np.pi*l1*l1)

def _score(cref, cext, side):
    """
    Gives a score to how likely 2 clusters are apart of the same transition.
    
    The function calculates the collinearity factor as presented in [2] and penalizes if the angle of both clusters are not within the smallest incertainty of the 2
    
    cref is the reference cluster that is already in the transition and from which the algorithm is attempting to extend the transition line
    cext is the potential extension cluster.  
    side must be "left" or "right".  It is the side on which cext is from cref.

    [2] Three-dimentional object recognition from single two-dimensional images, D. G. Lowe, Artificial Intelligence.
    """
    if side == "left":
        s = _distance(cref.start, cext.stop)
    elif side == "right":    
        s = _distance(cref.stop, cext.start)
    g = _perpendicular_dist(cref.slope, cref.intercept, cext.p0)
    l1 = cext.length
    dtheta = abs(cref.theta_y-cext.theta_y)
    sigma_t = min((cref.corr_sigma_theta, cext.corr_sigma_theta))
    score = _collinearity(dtheta, s, g, l1)
    if sigma_t != 0.:
        corr = dtheta/sigma_t
    else:
        corr = 10.
    if corr > 1.:
        score = score*corr
    return score

def Transitions(cc, imgShape, _proImage = None):
    """
    Transitions() regroups clusters that are collinear to eachother into lines.
    
    cc is an array of Clusters
    imgShape is the image dimension in number of pixels ((y, x))

    returns a list of Transitions and a list of left-over Clusters.
    """
    nCluster = sum(1 for i in cc if i.corr_sigma_theta <= np.pi/8.)
    tlist = []
    cc = list(cc)
    leftover = deepcopy(cc[-1])
    cc2 = deepcopy(cc[:-1])
    next_tran = 0
    while next_tran != None:
        next_tran, cc2 = _next_transition(cc2, nCluster, imgShape, _proImage = _proImage)
        if next_tran != None:
            tlist.append(next_tran)
    cc2.append(leftover)
    #éliminer les transitions de merde et conserver les bonnes
    return tlist, cc2

def _next_transition(cc, nCluster, imgShape, _proImage = None):
    """
    returns the next best-found transition or None if no ensemble of clusters formed a good enough transition.
    
    cc is an array of all the remaining Clusters
    nCluster is the number of clusters there were initially
    imgShape is the image dimension in number of pixels ((y, x))
    """
    index = []
    init_guess_index = _init_guess(cc)
    if init_guess_index==None:
        return None, cc
    else:
        index.append(init_guess_index)
        clust = deepcopy(cc[init_guess_index])
        tclust = Transition(clust, _proImage = _proImage)
        ind = init_guess_index
        while ind != None:
            ind = _extend(cc, ind, index, nCluster, imgShape, side="left")
            if ind != None:
                index.append(ind)
                clust = deepcopy(cc[ind])
                tclust = Transition(clust, tclust, _proImage = _proImage)
        clust = deepcopy(cc[init_guess_index])
        ind = init_guess_index
        while ind != None:
            ind = _extend(cc, ind, index, nCluster, imgShape, side="right")
            if ind != None:
                index.append(ind)
                clust = deepcopy(cc[ind])
                tclust = Transition(clust, tclust, _proImage = _proImage)
        ind = -np.array(index)
        ind = np.argsort(ind)
        cc2 = deepcopy(cc)
        for u in ind:
            cc2.pop(index[u])
        return tclust, cc2

def _slope_int(p0, p1):
    """
    returns the slope and intercept of a line passing through both p0 and p1.
    
    p0, p1 \t (y,x) coordinates of 2 points the line passes through
    """
    a = (p1[0]-p0[0])/(p1[1]-p0[1])
    b = p0[0]-a*p0[1]
    return a, b
    
def _extend(cc, init_guess_index, indexx, nCluster, imgShape, side):
    """
    the _extend() function takes an initial cluster and a direction and looks for the best, next collinear cluster in a list of clusters.
    returns the indice of the next cluster that is collinear within a certain quality to the initial cluster
    
    cc is an array of Clusters
    init_guess_index is the indice of the cluster for which we will look for a collinear cluster
    side is the direction in which we will be looking for the next collinear cluster.  must be "left" or "right"
    indexx is the indices of all clusters that are already part of the transition being built
    nCluster is the initial number of clusters identified in the image
    imgShape is the image dimension in number of pixels ((y, x))
    """
    rel_r = _rrel(nCluster, imgShape[0], imgShape[1])
    ind = []
    cc = np.array(cc)
    for u, i in enumerate(cc):
        if side == "left":
            rr = _distance(i.stop, cc[init_guess_index].start)
        if side == "right":
            rr = _distance(i.start, cc[init_guess_index].stop)
        if (u not in indexx) and (rr < rel_r): 
            a, b = _slope_int(i.p0, cc[init_guess_index].p0)
            ty = (90.+AngleLineXAxis(a))*np.pi/180.
            dt = abs(cc[init_guess_index].theta_y-i.theta_y)
            if dt >= np.pi/2:
                if ty < min((cc[init_guess_index].theta_y, i.theta_y))+(12.*np.pi/180) or ty > max((cc[init_guess_index].theta_y, i.theta_y))-(12.*np.pi/180):
                    if side == "left" and (cc[init_guess_index].start_y > cc[u].p0_y):
                        ind.append(u)
                    elif side == "right" and (cc[init_guess_index].stop_y < cc[u].p0_y):
                        ind.append(u)
            else:
                if ty > min((cc[init_guess_index].theta_y, i.theta_y))-(12.*np.pi/180) and ty < max((cc[init_guess_index].theta_y, i.theta_y))+(12.*np.pi/180):
                    if side == "left" and (cc[init_guess_index].start_y > cc[u].p0_y):
                        ind.append(u)
                    elif side == "right" and (cc[init_guess_index].stop_y < cc[u].p0_y):
                        ind.append(u)
    cc = list(cc)
    ind = np.array(ind)
    score = np.zeros(ind.shape[0])
    for u, i in enumerate(ind):
        score[u] = _score(cc[init_guess_index], cc[i], side)
    try:
        if min(score) <= 1.:
            best = np.argmin(score)
            return ind[best]
        else:
            return None
    except (ValueError):
        return None

def average_y(ccx, ccy):
    """
    
    """
    cx, cy = [], []
    for u in ccx:
        if u not in cx:
            index = np.where(u==ccx)    
            cx.append(u)
            cy.append(sum(ccy[index[0]])/float(index[0].size))
    return np.array(cx), np.array(cy)
    








#def _kernel(angle, dist, sigma_rho, sigma_theta, sigma_rho_theta, rho, theta_y):
#    """
#    The Kernel function allows the user to compute the angle-distance kernel of the Hough Transform with the specified angle, distance and variances/covariances
#    """
#    angle = angle*np.pi/180.
#    r = sigma_rho_theta/(sigma_rho*sigma_theta)
#    kernel = np.zeros((dist.shape[0],angle.shape[0]))
#    for u, i in enumerate(angle):
#        for j, k in enumerate(dist):
#            z0 = (k-rho)*(k-rho)/(sigma_rho*sigma_rho)-2*r*(k-rho)*(i-theta_y)/(sigma_rho*sigma_theta)+(i-theta_y)*(i-theta_y)/(sigma_theta*sigma_theta)
##            z1 = (k+rho)*(k+rho)/(sigma_rho*sigma_rho)-2*r*(k+rho)*(i-theta_y-np.pi)/(sigma_rho*sigma_theta)+(i-theta_y-np.pi)*(i-theta_y-np.pi)/(sigma_theta*sigma_theta)
##            z2 = (k+rho)*(k+rho)/(sigma_rho*sigma_rho)-2*r*(k+rho)*(i-theta_y+np.pi)/(sigma_rho*sigma_theta)+(i-theta_y+np.pi)*(i-theta_y+np.pi)/(sigma_theta*sigma_theta)
#            kernel[j, u] = 1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z0/(2*(1-r*r)))#+1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z1/(2*(1-r*r)))+1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z2/(2*(1-r*r)))
#    return kernel