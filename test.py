import mrcfile
import numpy as np
import sys, copy, os
#from numba import jit
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

#file = mrcfile.open("./raw/session1/tomo01_b8_bp100-10000.mrc")
#file = mrcfile.open("./raw/20220307_Baker/tomo8_b8_bp10000-500.mrc")
file = mrcfile.open("./raw/20220222_Baker/tomo07_b8_bp10000-500.mrc")
#file = mrcfile.open("./raw/session1/tomo20_b8_bp100-10000.mrc")

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
3. Generate projections over z-axis
"""


#taking slices of tomogram (NB - array is in dimensions z-y-x)
#slice between z1 and z2
z1 = 50
z2 = 53
slices = data[z1:z2,:,:]

proj_z = np.sum(slices,axis=0)*-1 #generate projection of data across z (by summing) – create y by x array
#proj_z = np.transpose(proj_z) #transpose - create matrix with dimensions x by y

img = plt.figure()
img.set_size_inches(10,10)
plt.imshow(proj_z, cmap='gray')
plt.title("Z-projection, slices %s-%s" %(z1,z2))
plt.colorbar()
plt.show()


"""
4. Generate projections over y-axis + FFT
"""

#taking slices of tomogram (NB - array is in dimensions z-y-x)
#slice between y1 and y2
y1 = 150
y2 = 350
slices = data[:,y1:y2,:]

proj_y = -1*np.sum(slices, axis = 1)
FT_proj_y = np.fft.fft2(proj_y)

img2 = plt.figure()
img2.set_size_inches(10,10)
plt.imshow(np.log10(abs(FT_proj_y)),cmap="gray")
plt.title("FT of y-projection")
plt.colorbar()
plt.show()

"""
5. Generate projections over x-axis + FFT
"""

#taking slices of tomogram (NB - array is in dimensions z-y-x)
#slice between x1 and x2
x1 = 150
x2 = 350
slices = data[:,:,x1:x2]

proj_x = -1*np.sum(slices, axis = 2)
FT_proj_x = np.fft.fft2(proj_x)

img3 = plt.figure()
img3.set_size_inches(10,10)
plt.imshow(np.log10(abs(FT_proj_x)),cmap='gray')
plt.title("FT of x-projection")
plt.colorbar()
plt.show()

"""
6. Edge detection — Canny edge detector
"""

print('Edge detection results...')

#generate projection across z-axis


#taking slices of tomogram (NB - array is in dimensions z-y-x)
z1 = 70
z2 = 73
slices = data[z1:z2,:,:] #slice between z1 and z2

proj_z = -np.sum(slices,axis=0) #generate projection of data across z (by summing) – create y by x array
# proj_z = proj_z[:460,:460]

#normalising intensity values - making them all posiitve and between 0-1
### https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255/50966711
norm_image = cv2.normalize(proj_z, None, alpha=0, beta=1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#compute images with canny filter @ different sigmas
edges1 = feature.canny(proj_z)
edges2 = feature.canny(proj_z, sigma=3)
edges3 = feature.canny(proj_z, sigma=5)
edges4 = feature.canny(proj_z, sigma=10)

#display results

## display results using matplotlib
fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (24,9))

ax[0].imshow(norm_image, cmap='gray')
ax[0].set_title('projection')

ax[1].imshow(edges1, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma = 1$')

ax[2].imshow(edges2, cmap='gray')
ax[2].set_title(r'Canny filter, $\sigma = 3$')

ax[3].imshow(edges3, cmap='gray')
ax[3].set_title(r'Canny filter, $\sigma = 5$')

ax[4].imshow(edges4, cmap='gray')
ax[4].set_title(r'Canny filter, $\sigma = 10$')

for a in ax:
    a.axis('off')
fig.tight_layout()
fig.suptitle('Canny edge detector results for different sigmas')
plt.show()

##overlaying Canny filter over original image
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize = (20,10))
ax2[0].imshow(color.label2rgb(edges2, norm_image, bg_label=0))
ax2[0].set_title(r'Canny filter, $\sigma = 3')

ax2[1].imshow(color.label2rgb(edges3, norm_image,bg_label=0))
ax2[1].set_title(r'Canny filter, $\sigma = 5')

fig2.suptitle('Canny edge detector over tomograms')

plt.show()

##joining edges together
fill_edges2 = ndi.binary_fill_holes(edges2)
fill_edges3 = ndi.binary_fill_holes(edges3)

fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize = (20,10))
ax3[0].imshow(color.label2rgb(fill_edges2, norm_image,bg_label=0))
ax3[0].set_title(r'Canny filter, $\sigma = 3')

ax3[1].imshow(color.label2rgb(fill_edges3, norm_image,bg_label=0))
ax3[1].set_title(r'Canny filter, $\sigma = 5')

fig3.suptitle('Filling Canny edges')
plt.show()

"""
7. LoG
"""
print('Laplacian of Gaussian...')

blob1 = -ndi.gaussian_laplace(proj_z, sigma = 2)
blob2 = -ndi.gaussian_laplace(proj_z, sigma = 3)
blob3 = -ndi.gaussian_laplace(proj_z, sigma = 5)
blob4 = -ndi.gaussian_laplace(proj_z, sigma = 7)

fig4, ax4 = plt.subplots(nrows=1, ncols=5, figsize = (24,9))

ax4[0].imshow(norm_image, cmap='gray')
ax4[0].set_title('projection')

ax4[1].imshow(blob1, cmap='gray')
ax4[1].set_title(r'LoG, $\sigma = 2$')

ax4[2].imshow(blob2, cmap='gray')
ax4[2].set_title(r'LoG, $\sigma = 3$')

ax4[3].imshow(blob3, cmap='gray')
ax4[3].set_title(r'LoG, $\sigma = 5$')

ax4[4].imshow(blob4, cmap='gray')
ax4[4].set_title(r'LoG, $\sigma = 10$')

fig4.suptitle('LoG filter results')

plt.show()

"""
8. Thresholding + blurrying threshold mask + canny mask detection
"""
print('Thresholding...')

thres1 = (norm_image > 0.3)
thres2 = (norm_image > 0.52)
thres3 = (norm_image > 0.7)

fig5, ax5 = plt.subplots(nrows=1, ncols=4, figsize = (24,9))

ax5[0].imshow(norm_image, cmap='gray')
ax5[0].set_title('projection')

ax5[1].imshow(thres1, cmap='gray')
ax5[1].set_title(r'threshold = 0.3')

ax5[2].imshow(thres2, cmap='gray')
ax5[2].set_title(r'threshold = 0.52')

ax5[3].imshow(thres3, cmap='gray')
ax5[3].set_title(r'threshold = 0.7')

fig5.suptitle('Thresholding results')
plt.show()

#Blurring out threshold mask with Gaussian
thres_mask = thres2.astype(np.float32) #Convert boolean into float (so that Gaussian blur can be applied)
thres_maskblur = ndi.gaussian_filter(thres_mask,10) #Apply blur to threshold
thres_maskblur = thres_maskblur/np.amax(thres_maskblur) #Normalise
img_masked = norm_image * thres_maskblur #Apply mask onto image
img_masked = img_masked/np.max(img_masked) #normalise masked image between 0 and 1

fig6, ax6 = plt.subplots(nrows=1, ncols=3, figsize = (10,30))
ax6[0].imshow(proj_z,cmap='gray')
ax6[0].set_title('projection')

ax6[1].imshow(thres_maskblur,cmap='gray')
ax6[1].set_title('Blurred threshold mask, threshold = 0.52')

ax6[2].imshow(img_masked,cmap='gray')
ax6[2].set_title('Mask applied to image')

fig6.suptitle('Thresholding results')
plt.show()

#Canny edge detection on masked image
##Showing blurred threshold mask area
maskarea = (thres_maskblur > 0.2)
plt.figure()
plt.imshow(color.label2rgb(maskarea, norm_image,bg_label=0))
plt.title('Mask area')
plt.show()

edges5 = feature.canny(proj_z, sigma=3, mask=maskarea)

fig8, ax8 = plt.subplots(nrows=2, ncols=2, figsize = (16,16))

ax8[0,0].imshow(norm_image, cmap='gray')
ax8[0,0].set_title('Z-projection, slices %s-%s' %(z1,z2))

ax8[0,1].imshow(img_masked, cmap='gray')
ax8[0,1].set_title('masked projection')

ax8[1,0].imshow(edges5, cmap='gray')
ax8[1,0].set_title('Canny, sigma = 3')

ax8[1,1].imshow(color.label2rgb(edges5, norm_image,bg_label=0))
ax8[1,1].set_title('Canny filter over original projection')

fig8.suptitle('Blurred threshold mask + Canny edge detection')
plt.show()

"""
9. Watershed Segmentation

https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
https://stackoverflow.com/questions/21060804/opencv-canny-watershed
https://medium.com/on-coding/euclidean-distance-transform-d37e06958216

"""
#thresholding

# img_256 = 255 * norm_image
# img_256 = img_256.astype(np.uint8)

thres_imgmask = (img_masked > 0.3)
thres_normimg = norm_image > 0.55

thres = thres_normimg * maskarea

plt.figure()
plt.subplot(221)
plt.imshow(thres, cmap='gray')
plt.subplot(222)
plt.imshow(norm_image, cmap='gray')
plt.subplot(223)
plt.imshow(thres_imgmask, cmap='gray')
plt.subplot(224)
plt.imshow(img_masked, cmap='gray')
plt.show()

#generate markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(thres_imgmask) #Euclidean distance transform
coords = peak_local_max(distance,  footprint = np.ones((3,3)), labels=thres)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=thres) #find labels/segments
#labels = labels/np.max(labels)

#low pass filtering segments/labels
#crop labels
labels = labels[:460,:460] #crop to 460x460
FTlabels = np.fft.fft2(labels)
plt.imshow(np.log10(abs(FTlabels)),cmap='gray')

r = 5
ham = np.hamming(460)[:,None]
ham2d= np.sqrt(np.dot(ham, ham.T)) ** r
plt.imshow(ham2d, cmap='gray')

filtered = FTlabels * ham2d
labels_lowpass = np.fft.ifft2(filtered)
labels_lowpass = labels_lowpass*255/np.max(labels_lowpass)
labels_lowpass = labels_lowpass.astype(np.uint8)
plt.imshow(labels_lowpass, cmap='gray')


## plotting

fig, axes = plt.subplots(ncols=5, figsize=(75, 15), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(thres, cmap=plt.cm.gray)
ax[0].set_title('thresholding', fontsize = 30)
ax[1].imshow(norm_image, cmap = plt.cm.gray)
ax[1].set_title('original projection', fontsize = 30)
ax[2].imshow(-distance, cmap=plt.cm.gray)
ax[2].set_title('Distance transform', fontsize = 30)
ax[3].imshow(labels, cmap=plt.cm.gray)
ax[3].set_title('Separated objects', fontsize = 30)
ax[4].imshow(color.label2rgb(labels_lowpass, norm_image,bg_label=0))
ax[4].set_title('Separated objects', fontsize = 30)

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

fig9, ax9 = plt.subplots(ncols=3, figsize=(45,15))

ax9[0].imshow(norm_image, cmap = plt.cm.gray)
ax9[0].set_title('original projection', fontsize = 30)
ax9[1].imshow(labels)
ax9[1].set_title('Separated objects', fontsize = 30)
ax9[2].imshow(color.label2rgb(labels,norm_image,bg_label=0))
ax9[2].set_title('Separated objects', fontsize = 30)

for a in ax9:
    a.set_axis_off()

fig.tight_layout()
plt.show()


#statistics 
#water content
protein = np.count_nonzero(labels_lowpass)
protein_content = protein/(460*460)
water_content = 1-protein_content
print(water_content)

