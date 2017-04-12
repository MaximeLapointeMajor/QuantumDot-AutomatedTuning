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
from itertools import product
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
        
        p0, u, v, pt, ratio, length, covar = u_v_(cluster)
        if v[0]<0.:
            v = -v
#        if u[1]<0.:
#            u = -u
        nPoints = cluster.T[0].shape[0]
        
        self.pt = pt
        self.pt_x = pt[1]
        self.pt_y = pt[0]
        self.cluster = cluster
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

        self.correl_coeff = correl_coeff
        self.correction_coeff = correction
        self.corr_sigma_theta = np.sqrt(sigma_theta_sq)*correction
        self.corr_sigma_rho = np.sqrt(sigma_rho_sq)*correction
        self.corr_sigma_rt = sigma_rho_theta*correction*correction
        self.corr_x = cp[1]
        self.corr_y = cp[0]
        
        self.covar = covar




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
    
    """
    ret = None
    for i in product((np.arange(ref[0]-gap_size-1, ref[0]+gap_size+2, 1))%img.shape[0], (np.arange(ref[1]-gap_size-1, ref[1]+gap_size+2, 1))%img.shape[1]):
        if img[i[0], i[1]]!=0.:
            ret = list(i)
            break
    return ret

def _rotate(cluster, u, v, p0):
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
                pt = u.copy()
                eigmax = k
                vecmin = eigvector[i-1]
                vecmax = eigvector[i]
    ### The eig() function of numpy currently has a problem with associating the proper eigenvector with its eigenvalue.
    ### The eigenvectors are also inverted ([y,x] instead of [x,y]), which in our case fits our need, but could cause problems to a user.
    u = vecmin[np.array([0,1])].copy()
    v = vecmax[np.array([0,1])].copy()
    cp = _rotate(cluster, u, v, p0)
    mod = lmf.models.LinearModel()
    out = mod.fit(cp[0], x=cp[1])
    slope = out.values.get('slope')
    theta_y = _theta_y(v)-AngleLineXAxis(slope)*np.pi/180.
    u[0] = np.cos(theta_y)
    u[1] = -np.sin(theta_y)
    v[1] = u[0].copy()
    v[0] = -u[1].copy()
    cp = _rotate(cluster, u, v, p0)
    length = max(cp[1])-min(cp[1])
    ratio = (max(cp[0])-min(cp[0]))/length
    mod = lmf.models.LinearModel()
    mod.set_param_hint('slope', value=0.)
    out = mod.fit(cp[0], x=cp[1])
    covar = out.covar
    return p0, u, v, pt, ratio, length, covar

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



def _kernel(angle, dist, sigma_rho, sigma_theta, sigma_rho_theta, rho, theta_y):
    angle = angle*np.pi/180.
    r = sigma_rho_theta/(sigma_rho*sigma_theta)
    kernel = np.zeros((dist.shape[0],angle.shape[0]))
    for u, i in enumerate(angle):
        for j, k in enumerate(dist):
            z0 = (k-rho)*(k-rho)/(sigma_rho*sigma_rho)-2*r*(k-rho)*(i-theta_y)/(sigma_rho*sigma_theta)+(i-theta_y)*(i-theta_y)/(sigma_theta*sigma_theta)
#            z1 = (k+rho)*(k+rho)/(sigma_rho*sigma_rho)-2*r*(k+rho)*(i-theta_y-2*np.pi)/(sigma_rho*sigma_theta)+(i-theta_y-2*np.pi)*(i-theta_y-2*np.pi)/(sigma_theta*sigma_theta)
#            z2 = (k+rho)*(k+rho)/(sigma_rho*sigma_rho)-2*r*(k+rho)*(i-theta_y+2*np.pi)/(sigma_rho*sigma_theta)+(i-theta_y+2*np.pi)*(i-theta_y+2*np.pi)/(sigma_theta*sigma_theta)
            kernel[j, u] = 1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z0/(2*(1-r*r)))#+1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z1/(2*(1-r*r)))+1/(2*np.pi*sigma_rho*sigma_theta*np.sqrt(1-r*r))*np.exp(-z2/(2*(1-r*r)))
    return kernel








