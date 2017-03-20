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

def Cluster(image, gap_size=0, min_cluster_size=4):
    cc = []
    leftover = []
    img = image.copy()
    index = np.nonzero(img)
    while (index[0].shape[0]!=0):
        group = _linkage(img, gap_size=gap_size, first_call=True)
        for u in group:
            img[u[0], u[1]] = 0.
        if group.shape[0] >= min_cluster_size:
            cc.append(group)
        else:
            for u in group:
                leftover.append(u)
        index = np.nonzero(img)
    cc.append(leftover)
    return np.array(cc)


def _linkage(img, gap_size=0, mylist=None, first_call=False):
    if mylist==None:
        mylist = []
        p0 = np.array(np.where(img!=0.)).T
        mylist.append(list(p0[0]))
    for i in product((np.arange(mylist[-1][0]-gap_size-1, mylist[-1][0]+gap_size+2, 1))%img.shape[0], (np.arange(mylist[-1][1]-gap_size-1, mylist[-1][1]+gap_size+2, 1))%img.shape[1]):
        if img[i[0], i[1]]!=0. and list(i) not in mylist:
            mylist.append(list(i))
            mylist = _linkage(img, gap_size=gap_size, mylist=mylist, first_call=False)
    if first_call==True:
        mylist=np.array(mylist)
    return mylist




#def ExtDroite(t,p):
#    a=float(p[-1,1]-p[-2,1])/float(p[-2,0]-p[-2,0])
#    angle=AngleLineXAxis(a)
#    adown=Slope(angle+10)
#    aup=Slope(angle-10)
#    bdown=p[-1,1]-adown*p[-1,0]
#    bup=p[-1,1]-aup*p[-1,0]
#    index=np.where(t!=0)
#    index=np.array(index).T
#    tt=np.zeros(t.shape)
#    for u in index:
#        d=np.sqrt(float(p[-1,1]-u[0])+(p[-1,0]-u[1]))
#        if (d<20) or (d<150) and (u[0]<aup*u[1]+bup) and (u[0]>adown*u[1]+bdown):
#            tt[u[0],u[1]]=1
#    pp=transform.probabilistic_hough_line(tt,threshold=20,line_length=80,line_gap=20)
#    if pp==[]:
#        pp=transform.probabilistic_hough_line(tt,threshold=15,line_length=80,line_gap=20)
#    if pp==[]:
#        pp=transform.probabilistic_hough_line(tt,threshold=12,line_length=50,line_gap=15)
#    if pp==[]:
#        return p
#    else:
#        pp=np.array(pp)
#        for u, i in enumerate(pp):
#            pass
#
#def Extension(t,p):
#    pp=list(p)
#    index=np.where(t!=0)
#    index=np.array(index).T
#    tt=np.zeros(t.shape)
#    angle=AngleLineXAxis((p[1,1]-p[0,1])/(p[1,0]-p[0,0]))
#    aup=Slope(angle+10)
#    adown=Slope(angle-10)
#    bup=-aup*p[0,0]+p[0,1]
#    bdown=-adown*p[0,0]+p[0,1]
#    pp=list()
#    for u in index:
#        d=np.sqrt((p[0,0]-u[1])**2+(p[0,1]-u[0])*2)
#        if (d<100) and (u[0]<aup*u[1]+bup) and (u[0]>adown*u[1]+bdown):
#            tt[u[1],u[0]]=1
#    tt=transform.probabilistic_hough_line(tt,threshold=15,line_length=40,line_gap=15)
#    if tt!=[]:
#        tt=np.array(tt)
#        t0=tt[0,0]
#        t1=tt[0,1]
#        tmax=max(t0[0],t1[0])
#        if tmax==t0[0]:
#            pp.append(t1)
#            pp.append(t0)
#            pp.append(p)
#        else:
#            pp.append(t0)
#            pp.append(t1)
#            pp.append(p)
##    tt=np.zeros(t.shape)
##    angle=AngleDroiteAxeX((p[-1,1]-p[-2,1])/(p[-1,0]-p[-2,0]))
##    bup=-aup*p[-1,0]+p[-1,1]
##    bdown=-adown*p[-1,0]+p[-1,0]
##    for u in index:
##        d=sqrt((p[-1,0]-u[1])**2+(p[-1,1]-u[0])*2)
##        if (d<100) and (u[0]<aup*u[1]+bup) and (u[0]>adown*u[1]+bdown):
##            tt[u[1],u[0]]=1
##    tt=transform.probabilistic_hough_line(tt,threshold=15,line_length=40,line_gap=15)
##    if yy!=[]:
##        pass
##    else:
##        #ajouter le point à gauche + moyenner le point de droite
#    return pp
#
#def ParamTransition(pp):
#    a=np.zeros(pp.shape[0])
#    b=np.zeros(pp.shape[0])
#    x=[]
#    y=[]
#    p=np.copy(pp)
#    for u, i in enumerate (pp):
#        a[u]=float(i[1,1]-i[0,1])/float(i[1,0]-i[0,0]+1e-25)
#        b[u]=float(-i[1,0]*a[u])+i[1,1]
#        x.append(np.linspace(i[0,0],i[1,0],abs(i[1,0]-i[0,0])+1))
#        y.append(x[u]*a[u]+b[u])
#    y=np.array(y)
#    x=np.array(x)
#    index1=np.where(a<0.)
#    a=np.delete(a,index1)
#    b=np.delete(b,index1)
#    x=np.delete(x,index1)
#    y=np.delete(y,index1)
#    p=np.delete(p,index1,0)
#    index2=np.where(a>5.2)
#    a=np.delete(a,index2)
#    b=np.delete(b,index2)
#    x=np.delete(x,index2)
#    y=np.delete(y,index2)
#    p=np.delete(p,index2,0)
#    return a, b, x, y, p
#
#def Regroupe(pp,a,b):
#    index=[]
#    for u, i in enumerate(pp):
#        temp=[]
#        for j, k in enumerate(a):
#            d1=DistancePointLine(i[0,0],i[0,1],k,b[j])
#            d2=DistancePointLine(i[1,0],i[1,1],k,b[j])
#            if ((d1<5) or (d2<5)):
#                temp.append(j)
#        if temp==[]:
#            index.append(None)
#        else:
#            index.append(np.array(temp))
#    return np.array(index)
#
#def Filtre(t,a,b,p,index):
#    temp=np.zeros(t.shape)
#    index2=np.where(t!=0)
#    index2=np.array(index2).T
#    for i in index:
#        p0=min(p[i,0,0],p[i,1,0])
#        p1=max(p[i,0,0],p[i,1,0])
#        for j in index2:
#            d=DistancePointLine(j[1],j[0],a[i],b[i])
#            if (d<6) and (j[1]<(p1+10)) and (j[1]>(p0-10)):
#                temp[j[0],j[1]]=1
#    return temp
#
#def InlierPoints(t):
#    temp=np.zeros(t.shape)
#    index=np.where(t!=0)
#    index=np.array(index).T
#    model_robust,inliers=measure.ransac(index,measure.LineModel,min_samples=20,residual_threshold=1,max_trials=1000)
#    for u, i in enumerate(inliers):
#        if i==True:
#            temp[index[u,0],index[u,1]]=1
#    return temp
#
#def TrueTransition(t):
#    h=transform.hough_line(t)
#    peak=transform.hough_line_peaks(h[0],h[1],h[2])
#    peak=np.array(peak)
#    a=-np.tan(np.pi/2-peak[1,0])
#    py=np.sin(peak[1,0])*abs(peak[2,0])
#    px=np.cos(peak[1,0])*abs(peak[2,0])
#    if peak[2,0]<0:
#        py=-py
#    b=py-a*px
##    pp=transform.probabilistic_hough_line(t,line_gap=20)
##    pp=np.array(pp)
#    index=np.where(t!=0)
##    xmin=t.shape[1]
##    xmax=0.
##    for u in pp:
##        for i in u:
##            if i[0]<xmin:
##                xmin=i[0]
##            if i[0]>xmax:
##                xmax=i[0]
#    xmin=min(index[1])
#    xmax=max(index[1])
#    y0=a*xmin+b
#    y1=a*xmax+b
#    p=np.array([[xmin,y0],[xmax,y1]])
#    return a, b, p
#
#def RemovePoints(t,a,b,p):
#    temp=np.copy(t)
#    index=np.where(t!=0)
#    index=np.array(index).T
#    x0=min(p[0,0],p[1,0])
#    x1=max(p[0,0],p[1,0])
#    for u, i in enumerate(index):
#        d=DistancePointLine(i[1],i[0],a,b)
#        if (d<5) and (i[1]<(x1+5)) and (i[1]>(x0-5)):
#            temp[i[0],i[1]]=0
#    return temp
#
#def Sequence(t,n,tran):
#    p=transform.probabilistic_hough_line(t)
#    if p==[]:
#        return t, None
#    else:
#        p=np.array(p)
#        a,b,x,y,pp=ParamTransition(p)
#        plt.figure(n)
#        plt.imshow(t,aspect='auto',cmap=cm.bone)
#        for u in tran:
#            plt.plot(u.xx,u.yy,'b')
#        for u in range(pp.shape[0]):
#            plt.plot(x[u],y[u],'r')
#        index=Regroupe(pp,a,b)
#        temp=Filtre(t,a,b,p,index[0])
#        temp=InlierPoints(temp)
#        aa, bb, pp=TrueTransition(temp)
#        xx=np.linspace(pp[0,0],pp[1,0],abs(pp[0,0]-pp[1,0])+1)
#        yy=aa*xx+bb
#        plt.plot(xx,yy,'g')
#        temp=RemovePoints(t,aa,bb,pp)
#        return temp, pp
#
#def Looping(t):
#    tran=np.array([])
#    for u in range(40):
#        t, p=Sequence(t,u,tran)
#        if p==None:
#            break
#        else:
#            pp=Segment(p[0],p[1])
#            tran=list(tran)
#            tran.append(pp)
#            tran=np.array(tran)
#    return tran