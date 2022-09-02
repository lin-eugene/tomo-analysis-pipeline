import mrcfile 
from lib.analysis import *
from lib.editmrc import edit_pixelsize
import mrcfile
import numpy as np
import sys, copy, os
from numpy.ma.core import where
from scipy.ndimage.measurements import label, maximum_position
from skimage import feature, io, color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage as ndi

from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils
import pandas as pd 



class analysis_stacks():
    def __init__(self,rawtomoname,f=1500,r=20,plot=False):
        #initialise names
        filename = rawtomoname
        
        #editing pixel sizes from mrc file (bfilter changes binning in z to 1, so need to change it back)
        edit_pixelsize(filename)

        self.rawtomoname = rawtomoname.split('.')[0]
        self.session = self.rawtomoname.split('/')[-2]
        self.tomo = self.rawtomoname.split('/')[-1]

        self.coarse_mapname = f'data/masks/{self.session}/{self.tomo}_coarsemap.npy'
        self.maskname = f'data/masks/{self.session}/{self.tomo}_mask.npy'

        #load files
        with mrcfile.open(f'{self.rawtomoname}.mrc') as mrc:
            self.rawtomo = copy.deepcopy(mrc.data) * -1
            self.voxel_size = mrc.voxel_size

            self.vox_size = self.voxel_size.view((self.voxel_size.dtype[0], len(self.voxel_size.dtype.names)))
            self.vox_size = self.vox_size[0]

        self.normtomo=normalise_data(self.rawtomo)

        self.coarsemap = np.load(self.coarse_mapname)
        self.mask = np.load(self.maskname)
        self.mask = self.mask.astype(bool)

        #segmentation
        self.f=f

        #radav
        self.r = r

        ##running
        self.segmentation_slices()
        self.make_segmentedtomos()
        self.rdf_slices()

    def segmentation_slices(self,plot=False):
        for i, zslice in enumerate(self.mask):
            img_proc = ImageProcessing(zslice,self.rawtomoname,slice=i,f=self.f)
            proj = self.rawtomo[i,:,:]
            labels = img_proc.labels_filt > 0
            labels = labels.astype(float)

            if(plot):
                norm_proj = cv2.normalize(proj, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                plt.figure()
                plt.subplot(121)
                plt.imshow(color.label2rgb(labels, norm_proj,bg_label=0))
                plt.gca().invert_yaxis()
                plt.subplot(122)
                plt.imshow(proj,cmap='gray')
                plt.gca().invert_yaxis()
                plt.show()
    
    def make_segmentedtomos(self,plot=False):
        tomo_segmented = np.zeros(self.mask.shape)
        
        for i, zslice in enumerate(tomo_segmented):
            segmentname = f'data/segmentation/{self.session}/{self.tomo}/{i}_segments.npy'
            labels_filt = np.load(segmentname)
            labels = labels_filt > 0
            labels = labels.astype(float)
            indices = np.where(labels==0)
            newtomo = np.zeros_like(self.rawtomo[i,:,:])

            #mean = np.mean(self.rawtomo)
            std = np.std(self.rawtomo)
            newtomo[indices] = self.rawtomo[i][indices]-2*std

            tomo_segmented[i] = copy.deepcopy(newtomo)
            
            if(plot):
                hist = tomo_segmented[i,:,:].flatten()

                plt.figure(131)
                plt.subplot(131)
                plt.hist(hist)
                plt.subplot(132)
                plt.imshow(labels,cmap='gray')
                plt.gca().invert_yaxis()
                plt.subplot(122)
                plt.imshow(tomo_segmented[i],cmap='gray')
                plt.gca().invert_yaxis()
                plt.show()
            
            if(plot):
                plt.figure()
                plt.imshow(tomo_segmented[1,:,:],cmap='gray')
                plt.show()

        if(os.path.exists(f'./data/segmentedtomos/{self.session}')==False):
            os.makedirs(f'./data/segmentedtomos/{self.session}')

        outfilename = f'data/segmentedtomos/{self.session}/{self.tomo}_segmented.mrc'
        save_density(tomo_segmented, self.voxel_size, outfilename)

    def rdf_slices(self):
        for i, zslice in enumerate(self.normtomo):
            img = zslice
            peaklist = np.load(f'data/peaklist/{self.session}/{self.tomo}/{i}_peaklist.npy')
            gen_dists = GenerateDistributions(img,peaklist,self.r,self.rawtomoname,slice=i)



#####

class plotting():
    def __init__(self,rawtomoname, baseline=False):
         #initialise names
        self.rawtomoname = rawtomoname.split('.')[0]
        self.session = self.rawtomoname.split('/')[-2]
        self.tomo = self.rawtomoname.split('/')[-1]
        
        self.baseline=baseline


        #load files
        with mrcfile.open(f'{self.rawtomoname}.mrc') as mrc:
            self.rawtomo = copy.deepcopy(mrc.data) * -1
            self.voxel_size = mrc.voxel_size

            self.vox_size = self.voxel_size.view((self.voxel_size.dtype[0], len(self.voxel_size.dtype.names)))
            self.vox_size = self.vox_size[0]
        
        self.average_rdf_slices()
    
    def average_rdf_slices(self):
        if(self.baseline==False):
            radavlist = np.load(f'data/radavdata/radav/{self.session}/{self.tomo}/0_radavlist.npy')
            for i in range(len(self.rawtomo)-1):
                radavlist_b = np.load(f'data/radavdata/radav/{self.session}/{self.tomo}/{i+1}_radavlist.npy')
                radavlist = np.vstack((radavlist,radavlist_b))
        
        if(self.baseline==True):
            radavlist = np.load(f'data/radavdata/radav/{self.session}/{self.tomo}/0_blradavlist.npy')
            for i in range(len(self.rawtomo)-1):
                radavlist_b = np.load(f'data/radavdata/radav/{self.session}/{self.tomo}/{i+1}_blradavlist.npy')
                radavlist = np.vstack((radavlist,radavlist_b))


        self.dropletmean = np.mean(radavlist,axis=0)
        self.dropletstd = np.std(radavlist,axis=0)
        
        r = radavlist.shape[1]
        x_radav = np.arange(0,r,1)*self.vox_size

        plt.figure()
        plt.plot(x_radav,self.dropletmean)
        plt.fill_between(x_radav,self.dropletmean - self.dropletstd, self.dropletmean+self.dropletstd, alpha=0.2)
        plt.ylim([np.min(self.dropletmean-self.dropletstd),np.max(self.dropletmean+self.dropletstd)])
        plt.xlim([0,np.max(x_radav)])
        plt.xlabel('Distance from maximum (Angstroms)')
        plt.ylabel('Intensity')
        plt.show()

    