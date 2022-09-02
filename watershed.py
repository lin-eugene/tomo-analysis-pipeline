import mrcfile
import numpy as np
import sys, copy, os
from numba import jit
from numpy.ma.core import where
from scipy.ndimage.measurements import label, maximum_position
from skimage import feature, io, color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage as ndi

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils

"""
1. Import tomogram data
"""

file = mrcfile.open("./raw/tomo01_b8_bp100-10000.mrc")
#file = mrcfile.open("./raw/tomo20_b8_bp100-10000.mrc")

#print out dimensions of tomogram
print("Tomogram dimensions (Z,Y,X) = ",file.data.shape)

#copy tomogram data
data = copy.deepcopy(file.data) #copy data

""" 
2. Making histograms

TODO to self — try to generalise text position for many different histograms
"""

maxI = np.max(file.data)
minI = np.min(file.data)
hist_data = file.data.flatten() #turning mrcfile into 1d array for histogram plotting

#calculating statistics
mean = np.mean(hist_data)
stdev = np.std(hist_data)

#plotting histogram
hist1 = plt.figure()
plt.hist(hist_data, bins = 100)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of voxel intensities')
plt.text(-4, 0.8*1e6, r'$\mu = %1.5f, \sigma = %1.5f$' %(mean,stdev))
plt.grid(True)
plt.show()

"""

3. 
Normalise intensity values (mean = 0 , stdev = 1)
Generate z-projections

"""
#Normalise data
print("Normalising data...")

norm_data = (data - mean)/stdev
hist_data2 = norm_data.flatten() #turning mrcfile into 1d array for histogram plotting

hist2 = plt.figure()
plt.hist(hist_data2, bins = 100)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.xlim([-5, 5])
plt.title('Histogram of voxel intensities')
plt.grid(True)
plt.show()


#taking slices of tomogram (NB - array is in dimensions z-y-x)
#slice between z1 and z2

print("Generating slices...")
z1 = 40
z2 = 41
slices = norm_data[z1:z2,:,:]

proj = -1*np.sum(slices,axis=0) #generate projection of data across z (by summing) – create y by x array
#proj_z = np.transpose(proj_z) #transpose - create matrix with dimensions x by y

img = plt.figure(1)
img.set_size_inches(10,10)
plt.imshow(proj, cmap='gray', )
plt.title("Z-projection, slices %s-%s" %(z1,z2))
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

"""

4. Thresholding

NOTE — looks like sigma = 3 might be the best choice

"""

print('Thresholding...')

thres1 = (proj > 1)
thres2 = (proj > 2)
thres3 = (proj > 3)
thres4 = (proj > 4)

fig1, ax1 = plt.subplots(nrows=1, ncols=5, figsize = (30,7))

ax1[0].imshow(proj, cmap='gray')
ax1[0].invert_yaxis()
ax1[0].set_title('projection')

ax1[1].imshow(thres1, cmap='gray')
ax1[1].invert_yaxis()
ax1[1].set_title(r'threshold = $\sigma$')

ax1[2].imshow(thres2, cmap='gray')
ax1[2].invert_yaxis()
ax1[2].set_title(r'threshold = 2$\sigma$')

ax1[3].imshow(thres3, cmap='gray')
ax1[3].invert_yaxis()
ax1[3].set_title(r'threshold = 3$\sigma$')

ax1[4].imshow(thres4, cmap='gray')
ax1[4].invert_yaxis()
ax1[4].set_title(r'threshold = 4$\sigma$')

fig1.suptitle('Thresholding results')

fig1.tight_layout()
plt.show()

"""

5. Coarse mapping of area with protein
Threshold + Gaussian + Threshold

"""

thres_mask = thres3
thres_mask = thres_mask.astype(np.float32) #Convert boolean into float (so that Gaussian blur can be applied)
maskblur = ndi.gaussian_filter(thres_mask,10) #Apply blur to threshold
maskblur = maskblur/np.amax(maskblur) #Normalise
maskarea = (maskblur > 0.24)

#normalise image for plotting
norm_proj = cv2.normalize(proj, None, alpha=0, beta=1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#plot
plt.figure()
plt.imshow(color.label2rgb(maskarea, norm_proj, bg_label=0))
plt.title('Mask area')
plt.gca().invert_yaxis()
plt.show()



