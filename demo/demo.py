#!/usr/bin/env python3

import os
import numpy as np
from astropy.io import fits
from iqe import iqe

filename=f'{os.path.abspath(os.path.dirname(__file__))}/ascam1_20080710T230802.fits'
data=fits.open(filename)[0].data


z=data[686-4-1-5-5:686+4-1+5-5,711-4-1-1-5:711+4-1-1+5]

#z=numpy.flipud(z)
ana=iqe(z)
print(ana)


# sub-image
Z = z[10:18,6:14].copy()
ana=iqe(Z)
print(ana)

# add hotspot
Z = z[10:18,6:14].copy()
Z[3,2]=10.*Z.max()
ana=iqe(Z)
print(ana)

# mask hotspot
msk=np.ones_like(Z)
msk[3,2]=0.0
ana=iqe(Z,msk)
print(ana)


#ar=686-4-1-5-5
#ac=711-4-1-1-5
#gist.fma()
#gist.pli(data)
#plot_cross(ana[0]+.5+ac,ana[1]+.5+ar,sz=100)


