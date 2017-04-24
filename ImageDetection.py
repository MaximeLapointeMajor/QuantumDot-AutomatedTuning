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
import lmfit as lmf

#from scipy.optimize import curve_fit

class segment:
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

class Cluster:
    """
    
    """
    def __init__(self, cluster, max_gap):
        
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
#        if u[1]<0.:
#            u = -u
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
#        self.theta_x = 90.-theta_y
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

class Transition:
    """
    
    """
    
    def __init__(self, cluster, transition=None):
        
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
        
        self.xSegment = xSegment
        self.ySegment = ySegment
        
        self.cStart = clist[0]
        self.cStop = clist[-1]
        
        
        



def Initialization(cc, max_gap=0):
    """
    
    """
    cc = list(cc)
    leftover = cc.pop(-1)
    clust = []
    for u in cc:
        temp = Cluster(u, max_gap)
        ret = _split(temp.corr_x, temp.corr_y, temp.u, temp.v, temp.p0, temp.ratio)
        if ret == None:
            clust.append(temp)
        else:
            cc.append(ret[0])
            cc.append(ret[1])
    clust.append(Cluster(leftover, max_gap))
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


def Linkage(image, gap_size=0, min_cluster_size=4):
    """
    Returns a list of clusters each containing a minimum number of points separated by at most a specified number of pixels
    
    The last cluster is the ensemble of points that belong to no cluster. (index[-1])
    Each cluster is an array of coordinates of the form [x, y], assuming that the input image is an array of arrays with the first index corresponding to the x axis.
    -------
    image           \t must be a binary image.  Points to be clustered must have a value different than 0.
    gap_size        \t is the maximum number of pixels separating points that belong to a cluster.  Must be an integer
    min_cluster_size\t is the minimum ammount of points that a group of points must contain in order to be considered as a cluster.  If the group of points does not meet the requirement, they are placed in the left-over category in any order.
    """
    cc = []
    leftover = []
    img = image.copy()
    index = np.nonzero(img)
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
    for i in product((np.arange(ref[0]-gap_size-1, ref[0]+gap_size+2, 1))%img.shape[0], (np.arange(ref[1]-gap_size-1, ref[1]+gap_size+2, 1))%img.shape[1]):
        if img[i[0], i[1]]!=0.:
            ret = list(i)
            break
    return ret

def _segment_start_stop(x, y, a, b):
    """
    
    """
    if a > 0.:
        xmax = max(x)
        xmin = min(x)
        ymax = max(y)
        ymin = min(y)
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
        xmax = max(x)
        xmin = min(x)
        ymax = min(y)
        ymin = max(y)
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
        xmax = max(x)
        xmin = min(x)
        ymax = a*xmax+b
        ymin = ymax.copy()
    return np.array((ymin, xmin)), np.array((ymax, xmax))

def _backrotate(cc, u, v, p0):
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
    theta_y = _theta_y(v)+AngleLineXAxis(slope)*np.pi/180.
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
    
    """
    length = max(cp[1])-min(cp[1])
    ratio = (max(cp[0])-min(cp[0])+1)/length
    return length, ratio



def _rho(v, p0):
    """
    
    """
    return v[0]*p0[0]+v[1]*p0[1]

def _theta_y(v):
    """
    
    """
    return np.arccos(v[1])

def _sigma_mp_square(cluster, p0, u):
    """
    
    """
    ss = 0.
    for j in cluster:
        ss = ss+((u[1]*(j[1]-p0[1])+u[0]*(j[0]-p0[0]))**2)
    ss = 1./ss
    return ss

def _sigma_bp_square(n):
    """
    
    """
    return 1./n

def _matrixM(p0, u, v, sigma_mp_sq, sigma_bp_sq):
    """
    
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
    
    """
    sigma_mp_sq = _sigma_mp_square(cluster, p0, u)
    sigma_bp_sq = _sigma_bp_square(cluster.shape[0])
    sigma_rho_sq, sigma_theta_sq, sigma_rho_theta = _sigmas(p0, u, v, sigma_mp_sq, sigma_bp_sq)
    return sigma_rho_sq, sigma_theta_sq, sigma_rho_theta

def _split(corrx, corry, u, v, p0, ratio):
    """
    corrx is an array of the x axis coordinates of all the points part of the cluster.
    corry is an array of the y axis coordinates of all the points part of the cluster.
    sPoint is the splitting x-coordinate. If not specified, sPoint=None and will be taken as the maximum distance along the y axis (max(abs(corry)))
    
    
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
    
    """
    d = float(nSegments)/float(xPixels*yPixels)
    return 2*np.sqrt(1./(np.pi*d))
    
def _distance(end1, end2):
    """
    
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

def _score(theta, s, g, l1):
    """
    
    """
    return _collinearity(theta, s, g, l1)

def Transitions(cc, imgShape):
    """
    
    """
    nCluster = np.array(cc).shape[0]
    tlist = []
    cc = list(cc)
    cc2 = cc
    next_tran = 0
    while next_tran != None:
        next_tran, cc2 = _next_transition(cc2, nCluster, imgShape)
        if next_tran != None:
            tlist.append(next_tran)
    #éliminer les transitions de merde et conserver les bonnes
    return tlist, cc2

def _next_transition(cc, nCluster, imgShape):
    """
    
    """
    index = []
    init_guess_index = _init_guess(cc)
    if init_guess_index==None:
        return None, cc
    else:
        index.append(init_guess_index)
        clust = cc[init_guess_index]
        tclust = Transition(clust)
        #ici on pop cc[init_guess_index]
        ind = init_guess_index
        while ind != None:
            ind = _extend(cc, ind, nCluster, imgShape, side="left")
            if ind != None:
                index.append(ind)
                tclust = Transition(cc[ind], tclust)
        clust = cc[init_guess_index]
        ind = init_guess_index
        while ind != None:
            ind = _extend(cc, ind, nCluster, imgShape, side="right")
            if ind != None:
                tclust = Transition(cc[ind], tclust)
                index.append(ind)
        ind = -np.array(index)
        ind = np.argsort(ind)
        cc2 = cc
        for u in ind:
            cc2.pop(index[u])
        return tclust, cc2

    #going left
    
    #going right
    
    
def _extend(cc, init_guess_index, nCluster, imgShape, side):
    """
    
    """
    rel_r = _rrel(nCluster, imgShape[0], imgShape[1])
    ind, r, dtheta = [], [], []
    cc = np.array(cc)
    for u, i in enumerate(cc):
        if side == "left":
            rr = _distance(i.stop, cc[init_guess_index].start)
            s = 1
        if side == "right":
            rr = _distance(i.start, cc[init_guess_index].stop)
            s = -1.
        dt = abs(cc[init_guess_index].theta_y-i.theta_y)
        if (u != init_guess_index) and (i.p0_x*s < cc[init_guess_index].p0_x*s) and (dt < min((cc[init_guess_index].corr_sigma_theta,i.corr_sigma_theta))) and (rr < rel_r): # or (_distance(_distance(i.stop, cc[init_guess_index].start) < rel_r) < rel_r): [[[[[si ça link pas bien cest prob la 2e cond manquante]]]]]
            ind.append(u)
            r.append(rr)
            dtheta.append(dt)
    cc = list(cc)
    ind, r, dtheta = np.array(ind), np.array(r), np.array(dtheta)
    s, collin, score = np.zeros(ind.shape[0]), np.zeros(ind.shape[0]), np.zeros(ind.shape[0])
    for u, i in enumerate(ind):
        s[u] = _perpendicular_dist(cc[init_guess_index].slope, cc[init_guess_index].intercept, cc[i].p0)
        collin[u] = _collinearity(dtheta[u], s[u], r[u], cc[i].length)
        score[u] = _score(dtheta[u], s[u], r[u], cc[i].length)
    try:
        if min(score) < 1.:
            best = np.argmin(score)
            return ind[best]
        else:
            return None
    except (ValueError):
        return None
    
    
    
    
    
    
    
    
    
    
    
    
    





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



