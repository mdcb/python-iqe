#!/usr/bin/env python3

import os
import numpy as np
from astropy.io import fits
from iqe import iqe
import matplotlib.pyplot as plt

origin = 'upper'

#filename=f'{os.path.abspath(os.path.dirname(__file__))}/ascam1_20080710T230802.fits'

filename = 'ascam1_20080710T230802.fits'
data=fits.open(filename)[0].data

f, axarr = plt.subplots(2,3)

from astropy.visualization import ZScaleInterval, SinhStretch,ImageNormalize

norm = ImageNormalize(data,
      interval=ZScaleInterval(contrast=0.700),
      stretch=SinhStretch(a=0.4333333333333333))

axarr[0,0].imshow(data,norm=norm,origin=origin, cmap='gist_heat')

#------------------------------------------------------------------

z=data[686-4-1-5-5:686+4-1+5-5,711-4-1-1-5:711+4-1-1+5]
ana=iqe(z)
print('sub', ana)
axarr[0,1].plot([ana[0]],[ana[1]],marker='+', color='green')

norm = ImageNormalize(z,
      interval=ZScaleInterval(contrast=0.700),
      stretch=SinhStretch(a=0.4333333333333333))
norm = None

axarr[0,1].imshow(z,norm=norm,origin=origin, cmap='gist_heat')

#------------------------------------------------------------------

# sub-image
Z = z[10:18,6:14].copy()
ana=iqe(Z)
print('sub', ana)

axarr[0,2].plot([ana[0]],[ana[1]],marker='+', color='green')

norm = ImageNormalize(Z,
      interval=ZScaleInterval(contrast=0.700),
      stretch=SinhStretch(a=0.4333333333333333))
norm = None

axarr[0,2].imshow(Z,norm=norm,origin=origin, cmap='gist_heat')

#------------------------------------------------------------------




# add hotspot
Z = z[10:18,6:14].copy()
Z[3,2]=10.*Z.max()
ana=iqe(Z)
print('roi+hot', ana)
axarr[1,0].plot([ana[0]],[ana[1]],marker='+', color='green')

norm = ImageNormalize(Z,
      interval=ZScaleInterval(contrast=0.700),
      stretch=SinhStretch(a=0.4333333333333333))
norm = None

axarr[1,0].imshow(Z,norm=norm,origin=origin, cmap='gist_heat')
#------------------------------------------------------------------

# mask hotspot
msk=np.ones_like(Z)
msk[3,2]=0.0
ana=iqe(Z, msk)
print('roi+hot+msk', ana)
axarr[1,1].plot([ana[0]],[ana[1]],marker='+', color='green')

zz = z[10:18,6:14].copy()

norm = ImageNormalize(zz,
      interval=ZScaleInterval(contrast=0.700),
      stretch=SinhStretch(a=0.4333333333333333))
norm = None

axarr[1,1].imshow(zz,norm=norm,origin=origin, cmap='gist_heat')

#------------------------------------------------------------------

plt.show()


