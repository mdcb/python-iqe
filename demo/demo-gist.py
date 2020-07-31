#!/usr/bin/env python3

import os
import numpy as np
from astropy.io import fits
from iqe import iqe

import gist
def plot_cross(x,y,sz=1,color='red'):
  gist.pldj([x],[y-sz],[x],[y+sz],color=color)
  gist.pldj([x-sz],[y],[x+sz],[y],color=color)

filename=f'{os.path.abspath(os.path.dirname(__file__))}/ascam1_20080710T230802.fits'
data=fits.open(filename)[0].data


z=data[686-4-1-5-5:686+4-1+5-5,711-4-1-1-5:711+4-1-1+5]

#z=numpy.flipud(z)
ana=iqe(z)
gist.fma()
gist.pli(z)
plot_cross(ana[0]+.5,ana[1]+.5)

# sub-image
Z = z[10:18,6:14].copy()
ana=iqe(Z)
gist.fma()
gist.pli(Z)
plot_cross(ana[0]+.5,ana[1]+.5)

# add hotspot
Z = z[10:18,6:14].copy()
Z[3,2]=10.*Z.max()
anah=iqe(Z)
gist.fma()
gist.pli(Z)
plot_cross(anah[0]+.5,anah[1]+.5)

# mask hotspot
msk=np.ones_like(Z)
msk[3,2]=0.0
#gist.pli(msk*Z)
anam=iqe(Z,msk)
plot_cross(anam[0]+.5,anam[1]+.5,color='magenta')



#ar=686-4-1-5-5
#ac=711-4-1-1-5
#gist.fma()
#gist.pli(data)
#plot_cross(ana[0]+.5+ac,ana[1]+.5+ar,sz=100)


