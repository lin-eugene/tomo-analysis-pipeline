import numpy as np
import sys, copy, os
from scipy import signal
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pickle

class GenerateImage():
    def __init__(self, Gen_new_img):
        """
        Class that generates image of Gaussian blobs
        
        Arguments;
        - Gen_new_img -- TRUE/FALSE, asks class to generate new image or not
        """
        #initialising variables for kernel generation
        self.N = 200 #size of Gaussian kernel
        self.std = 20 #STDEV of Gaussian

        #initialising variables for image generation
        self.size = 1000 #image size
        self.peaks = 50 #number of peaks
        self.x = np.random.randint(self.size, size=self.peaks)
        self.y = np.random.randint(self.size, size=self.peaks)

        picklefile = './pickle/imgclass' #file to pickle
        if(os.path.exists('pickle')==False):
            os.system('mkdir pickle')
        if(Gen_new_img): 
            os.system('rm '+picklefile)

        if(os.path.exists(picklefile)):
            print(f"{picklefile} found, opening...")
            with open(picklefile, 'rb') as handle: #' rb' produces readable binary object
                imgclass = pickle.load(handle)
                self.img = imgclass.img
        else:

            self.Generate2DGaussian()
            self.GenerateImg()

            with open(picklefile, 'wb') as outfile: # 'wb' produces writeable binary object
                pickle.dump(imgclass, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                print('Redfield pickle file saved: restart.')
                print('File:',picklefile)
                outfile.close() 

        
    def Generate2DGaussian(self):
        """
        Function generates 2D Gaussian kernel with:
            N = size
            std = standard deviation

        """
        k1d = signal.gaussian(self.N, std=self.std) # returns row vector
        k1d = k1d.reshape(self.N,1) # reshape k1d into column vector
        self.kernel = np.outer(k1d,k1d) #generating Gaussian kernel (using outer product)   
        
        return self.kernel

    def GenerateImg(self):
        """
        Function generates 2D image with:
            Size — size x size
            x — array of x-coordinates
            y - array of y-coordinates
            kernel - kernel (e.g. Gaussian kernelt)
        """

        A = np.zeros((self.size,self.size))
        A[self.x,self.y] = 1 #Generate peak at specified coordinates
        self.img = signal.convolve2d(A,self.kernel, mode='same') #Generate image – convolve delta peak with Gaussian
        plt.imshow(self.img)
        plt.gca().invert_yaxis()
        plt.show()

        return self.img

class GenerateDistributions():
    def __init__(self, img, Gen_new_dist, Plot_Lines):
        """
        Class to plot out spatial distribution information from image
        
        Arguments:
        - img —- image (2D numpy array)
        - Gen_new_dist -- TRUE/FALSE — asks class to generate new distributions from new set of images
        - Plot_Lines -- TRUE/FALSE - asks class to plot out distributions

        """


        self.img = img
        self.size = img.shape

        # define angle and length of line from local maximum
        self.theta = np.arange(0,360,45) #angles sampled (array from 0 to 360, in increments of 45 degrees)
        self.r = 50

        picklefile = './pickle/distributions' #file to pickle

        if(os.path.exists('pickle')==False):
            os.system('mkdir pickle')
        if(Gen_new_dist): 
            os.system('rm '+picklefile)
        if(os.path.exists(picklefile)):
            print(f"{picklefile} found, opening...")
            with open(picklefile, 'rb') as handle: #' rb' produces readable binary object
                dists = pickle.load(handle)
        else:
            
            self.findMaxima()
            self.FilteringPoints()
            self.DrawLines()

            with open(picklefile, 'wb') as outfile: # 'wb' produces writeable binary object
                pickle.dump(dists, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                print('Redfield pickle file saved: restart.')
                print('File:',picklefile)
                outfile.close()

        if(Plot_Lines==True):
            self.PlotLines()

    def findMaxima(self):
        """
        finds maxima of peaks in image
        requires image as input
        returns coordinates of maxima in image
        """
        # find peaks and coordinates
        self.coords = peak_local_max(self.img, min_distance=10) #coords — N rows (N = number of peaks), 2 columns (x- and y-coordinates)
        coords_rows, coords_cols = self.coords.shape #number of rows = number of peaks

        self.y_peaks = self.coords[:,0] #row vector with y coordinates of peaks (N elements — N = number of peaks)
        self.x_peaks = self.coords[:,1] #row vector with x coordinates of peaks (N elements — N = number of peaks)

        N_peaks = len(self.x_peaks) #number of peaks identified

        print(N_peaks, " peaks identified...")

        print('Creating plot...')
        plt.imshow(img)
        plt.autoscale(False)
        plt.plot(self.x_peaks,self.y_peaks, 'r.')
        plt.gca().invert_yaxis()
        plt.show()


    def FilteringPoints(self):
        """
        function draws lines of specified distance from maximum points on image
        inputs:
            img - image
            coords - coordinates of maxima
            r - length of line
        
        """

        # remove points too close to edge (defined using length of line sampled from local max)
        #   generate a 2D distance matrix (same dimensions as image) – values at i,j = distance from closest edge
        array = np.ones(self.size, dtype=int)
        array[:,[0,-1]] = array[[0,-1]] = 0
        dfromedge = ndi.distance_transform_cdt(array, metric='chessboard')
        disttoosmall = dfromedge[self.y_peaks,self.x_peaks] < self.r #generates Boolean array, with N elements (N = number of peaks); True if peak is too close to edge of image

        # delete any peaks identified that were too close to image
        self.x_peaks = np.delete(self.x_peaks, np.argwhere(disttoosmall==True))
        self.y_peaks = np.delete(self.y_peaks, np.argwhere(disttoosmall==True))

        self.N_angles = len(self.theta) #Number of angles sampled
        self.N_peaks = len(self.x_peaks) #Number of peaks remaining
        N_peaks_disttoosmall = sum(disttoosmall) #Number of peaks deleted

        print(N_peaks_disttoosmall, "peak(s) too close to edge")
        print(f'Sampling from {self.N_peaks} peaks')

        
    def DrawLines(self):
        # find end points:
        # initialising matrices for x and y coordinates for end poiNnts
        #   rows(i) — N_rows = number of angles sampled
        #   columns(j) – N_cols = number of peaks
        #   value at i,j = x, y coordinates of particular end point at particular angle from peak
        endy = np.zeros((self.N_angles,self.N_peaks))
        endx = np.zeros((self.N_angles,self.N_peaks))
        
        #   for each angle
        #       for each peak
        #           compute end point of line drawn out from peak
        for i, angle in enumerate(self.theta):
            endy[i,:] = self.y_peaks + self.r*np.sin(np.radians(angle))
            endx[i,:] = self.x_peaks + self.r*np.cos(np.radians(angle))

        endy = endy.astype(int) 
        #row vector with y-coordinates of end points of line drawn from maximum – N_cols = N_peaks; N_rows = N_angles

        endx = endx.astype(int) 
        #row vector with x-coordinates of end points of line drawn from maximum — N_cols = N_peaks; N_rows = N_angles
    

        #for each angle
        #   for each peak
        #       compute array of x and y coordinates, with interval 1 (to extract intensity values)
        # i = len(theta) – number of angles sampled
        # j = coords_rows – number of peaks
        # k = r – number of points between peak and end point
        # value at i,j,k = x,y coordinates of points on line drawn between maximum and end point

        x = np.zeros((self.N_angles,self.N_peaks,self.r+1))
        y = np.zeros((self.N_angles,self.N_peaks,self.r+1))

        for i in range(self.N_angles):
            for j in range(self.N_peaks):
                y[i,j,:] = np.linspace(self.y_peaks[j], endy[i,j], self.r+1)
                x[i,j,:] = np.linspace(self.x_peaks[j], endx[i,j], self.r+1)

        self.y=y.astype(int)
        self.x=x.astype(int)

        # print('x.shape=',x.shape)
        # print('y.shape=',y.shape)


        """
        calculate intensities:
        """
        # i = N_angles – number of angles sampled
        # j = N_peaks – number of peaks
        # k = r – number of points between peak and end point
        # value at i,j,k = intensity values at particular x,y coordinates
        self.I = self.img[self.y,self.x]

    def PlotLines(self):
        """
        plotting
        """
        print('plotting intensities...')

        # scatter plots for intensity distributions
        rows = self.N_peaks
        cols = self.N_angles
        N_points = self.r+1
        fig, ax = plt.subplots(rows,cols,figsize=(3*cols,3*rows+1),sharex=True,sharey=True)

        fig.suptitle('Intensity distribution from maxima')
        plt.setp(ax[-1,:],xlabel = 'Distance from maximum (pixels)')
        plt.setp(ax[:,0],ylabel = 'Intensity')

        for i in range(self.N_peaks): #for each peak
            for j in range(self.N_angles): #for each angle
                x_ax = np.arange(0,N_points,1)
                ax[i,j].scatter(x_ax, self.I[j,i,:])

        #labelling rows and columns of subplots (https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots)
        col_labels = [f'{self.theta[i]}°' for i in range(self.N_angles)]
        row_labels = [f'Peak ({self.x_peaks[i]},{self.y_peaks[i]})' for i in range(self.N_peaks)]

        pad = 5 #in points

        for axes, col in zip(ax[0], col_labels):
            axes.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        for axes, row in zip(ax[:,0], row_labels):
            axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - pad, 0),
                        xycoords=axes.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        fig.tight_layout()
        plt.show()

        print('plotting lines...')
        # plot lines onto image
        xflat = self.x.ravel() #flattening 3d array (x) into 1d array for plotting
        yflat = self.y.ravel() #flattening 3d array (y) into 1d array for plotting

        plt.figure()
        plt.imshow(img)
        plt.autoscale(False)
        plt.plot(xflat,yflat, 'r.')
        plt.gca().invert_yaxis()
        plt.show()


"""
--------
"""
Gen_new_img = False
Gen_new_dist = False
Plot_Lines = True

genimg = GenerateImage(Gen_new_img)
gendist = GenerateDistributions(genimg.img, Gen_new_dist, Plot_Lines)