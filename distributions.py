from matplotlib import markers
import numpy as np
import sys, copy, os
from scipy import signal, stats
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pickle


class GenerateImage():
    def __init__(self, Gen_new_img, imgsize, N_blobs, imgmode, T, addnoise, filename):
        """
        Class that generates image of Gaussian blobs
        
        Arguments;
        - Gen_new_img -- TRUE/FALSE, asks class to generate new image or not
        - imgsize -- size of image
        - imgmode -- 'random' or 'solid'
        - N_blobs -- number of Guassians in image
        - T — temperature metric (standard deviation in placement of Gaussians)
        - addnoise -- boolean
        - filename -- name of file in a string
        """
        #initialising variables for kernel generation
        self.N = 200 #size of Gaussian kernel
        self.std = 20 #STDEV of Gaussian

        #initialising variables for image generation
        self.size = imgsize #image size
        self.peaks = N_blobs #number of peaks

        self.x = np.random.randint(self.size, size=self.peaks)
        self.y = np.random.randint(self.size, size=self.peaks)

        picklefile = f'./pickle/{filename}' #file to pickle
        if(os.path.exists('pickle')==False):
            os.system('mkdir pickle')
        if(Gen_new_img): 
            os.system('rm '+picklefile)

        if(os.path.exists(picklefile)):
            print(f"{picklefile} found, opening...")
            with open(picklefile, 'rb') as handle: #' rb' produces readable binary object
                self.img = pickle.load(handle)

        else:

            self.generate_2Dgaussian()
            
            if(imgmode=='random'):
                self.generate_img()
            if(imgmode=='solid'):
                self.T=T
                self.n=np.sqrt(N_blobs).astype(int)
                self.generate_solid()
            

            with open(picklefile, 'wb') as outfile: # 'wb' produces writeable binary object
                pickle.dump(self.img, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                print('Image pickle file saved: restart.')
                print('File:',picklefile)
                outfile.close() 
        
        if(addnoise):
            self.add_noise()


    
    def generate_2Dgaussian(self):
        """
        Function generates 2D Gaussian kernel with:
            N = size
            std = standard deviation
        """
        k1d = signal.gaussian(self.N, std=self.std) # returns row vector
        k1d = k1d.reshape(self.N,1) # reshape k1d into column vector
        self.kernel = np.outer(k1d,k1d) #generating Gaussian kernel (using outer product)   
        

    def generate_img(self):
        """
        Function generates 2D image with:
            Size — size x size
            x — array of x-coordinates
            y - array of y-coordinates
            kernel - kernel (e.g. Gaussian kernelt)
        """
        a = np.zeros((self.size,self.size))
        a[self.x,self.y] = 1 #Generate peak at specified coordinates
        self.img = signal.fftconvolve(a,self.kernel, mode='same') #Generate image – convolve delta peak with Gaussian
        plt.imshow(self.img)
        plt.gca().invert_yaxis()
        plt.show()
    

    def generate_solid(self):
        x = np.linspace(0, self.size-1, self.n).astype(int)
        y = np.linspace(0, self.size-1, self.n).astype(int)
        xx, yy = np.meshgrid(x, y)
        i, j = xx.shape

        rand_x = np.random.normal(0, scale=self.T, size=(i,j)).astype(int)
        rand_y = np.random.normal(0, scale=self.T, size=(i,j)).astype(int)

        xx_rand = xx + rand_x
        yy_rand = yy + rand_y

        cond = (xx_rand<0) | (xx_rand>self.size-1) | (yy_rand<0) | (yy_rand>self.size-1)

        cond_r = cond.ravel()
        xx_rand_r = xx_rand.ravel()
        yy_rand_r = yy_rand.ravel()

        xx_rand_r = np.delete(xx_rand_r, np.argwhere(cond_r==True))
        yy_rand_r = np.delete(yy_rand_r, np.argwhere(cond_r==True))

        b = np.zeros((self.size,self.size))
        b[xx_rand_r,yy_rand_r] = 1

        self.img = signal.fftconvolve(b, self.kernel, mode='same')

    
    def add_noise(self):
        mean = 0.0
        std = 0.2
        self.noisyimg = self.img + np.random.normal(mean,std,self.img.shape)



class ImageProcessing():
    def __init__(self, img):
        self.img = img
        self.find_peaks()
    
    def find_peaks(self):
        """
        finds maxima of peaks in image
        requires image as input
        inputs:
            - image

        outputs:
            coords - coordinates of maxima in image
        """
        # find peaks and coordinates
        self.coords = peak_local_max(self.img, threshold_abs=0.3)
        #   self.coords — rows= number of peaks), col 0 = y-coords, col 1 = x-coords

        y_peaks = self.coords[:,0] #row vector with y coordinates of peaks (N elements — N = number of peaks)
        x_peaks = self.coords[:,1] #row vector with x coordinates of peaks (N elements — N = number of peaks)

        N_peaks = len(x_peaks) #number of peaks identified

        print(N_peaks, " peaks identified...")

        print('Creating plot...')
        plt.imshow(self.img)
        plt.autoscale(False)
        plt.plot(x_peaks,y_peaks, 'r.')
        plt.gca().invert_yaxis()
        plt.show()



class GenerateDistributions():
    def __init__(self, img, peaklist, r, dtheta, 
        pl_intensities=False, 
        pl_lines=False, 
        pseudoradav=False, 
        pl_pseudoradav=False,
        radav=True,
        pl_radav=False, 
        compare=False, 
        clust_av=False):
        """
        Class to plot out spatial distribution information from image
        
        Arguments:
        - Inputs
            - img —- image (2D numpy array)
            - peaklist -- x,y-coords of peaks

        - Parameters
            - r -- sampling radius, length of line drawn out from centre of peak
            - dtheta -- sampling angle interval
            - pl_intensities -- BOOLEAN - do you want to plot out intensity values?
            - pl_lines -- BOOLEAN - do you want to plot sampling lines on your image?
            - pseudoradav -- BOOLEAN - do you want to average the intensities over all angles?
            - radav -- BOOLEAN - do you want to radially average?
            - compare -- BOOLEAN - do you want to compare outputs from pseudoradav and radav?
        """
        self.img = img
        self.size = img.shape

        # define angle and length of line from local maximum
        self.theta = np.arange(0,360,dtheta) #angles sampled (array from 0 to 360, in increments of 45 degrees)
        self.r = r

        #peak filtering
        self.peaklist = peaklist
        self.x_peaks = self.peaklist[:,1]
        self.y_peaks = self.peaklist[:,0]


        self.filter_peaks()

        #creating distributions - method 1: draw_lines
       
        if(pl_intensities):
            self.draw_lines()
            self.plot_intensities()

            if(pl_lines):
                self.plot_lines()

        if(pseudoradav):
            self.draw_lines()
            if(pl_pseudoradav):
                self.plot_psuedorad_av() 
            
        if(radav):
            self.radial_average()
            self.radial_average_norm()
            if(pl_radav):
                self.plotradav()
        
        if(compare):
            if(pseudoradav==True and radav==True):
                self.compare_radialmeans()
            else:
                print('pseudoradav and radav both need to be true')

        if(clust_av):
            self.cluster_averaging()
            self.cluster_averaging_norm()

    def filter_peaks(self):
        """
        function draws lines of specified distance from maximum points on image
        inputs:
            img - image
            coords - coordinates of maxima
            r - length of line
        
        outputs:
            coordinates of peaks, with peaks within defined distance to edge removed
        
        """

        # remove points too close to edge (defined using length of line sampled from local max)
        #   generate a 2D distance matrix (same dimensions as image) – values at i,j = distance from closest edge
        array = np.ones(self.size, dtype=int)
        array[:,[0,-1]] = array[[0,-1]] = 0
        dfromedge = ndi.distance_transform_cdt(array, metric='chessboard')
        disttoosmall = dfromedge[self.y_peaks,self.x_peaks] < self.r+50 #generates Boolean array, with N elements (N = number of peaks); True if peak is too close to edge of image

        # delete any peaks identified that were too close to image
        self.x_peaks = np.delete(self.x_peaks, np.argwhere(disttoosmall==True))
        self.y_peaks = np.delete(self.y_peaks, np.argwhere(disttoosmall==True))
        self.coords_filter = np.hstack((np.array([self.y_peaks]).T, np.array([self.x_peaks]).T))
        print('coords_filter.shape = ',self.coords_filter.shape)

        self.N_angles = len(self.theta) #Number of angles sampled
        self.N_peaks = len(self.x_peaks) #Number of peaks remaining
        N_peaks_disttoosmall = sum(disttoosmall) #Number of peaks deleted

        print(N_peaks_disttoosmall, "peak(s) too close to edge")
        print(f'Sampling from {self.N_peaks} peaks')

        if(self.N_peaks==0):
            print('No peaks selected, radius too large')
            
        else:
            pass

        print('Creating plot...')
        plt.imshow(self.img)
        plt.autoscale(False)
        plt.plot(self.x_peaks,self.y_peaks, 'r.')
        plt.gca().invert_yaxis()
        plt.show()


    def draw_lines(self):
        print('drawing radial lines...')
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
    

        # calculating x, y coordinates of lines drawn out radially from peak centre
            # for each angle
            #   for each peak
            #       compute array of x and y coordinates, with interval 1 (to extract intensity values)
            # 
            # i = len(theta) – number of angles sampled
            # j = coords_rows – number of peaks
            # k = r – number of points between peak and end point
            # value at i,j,k = x,y coordinates of points on line drawn between maximum and end point
        print('calculating intensities...')
        x = np.zeros((self.N_angles,self.N_peaks,self.r+1))
        y = np.zeros((self.N_angles,self.N_peaks,self.r+1))

        for i in range(self.N_angles):
            for j in range(self.N_peaks):
                y[i,j,:] = np.linspace(self.y_peaks[j], endy[i,j], self.r+1) #np.linspace -- similar to np.arange, but specifies number of intervals instead of interval spacing
                x[i,j,:] = np.linspace(self.x_peaks[j], endx[i,j], self.r+1)

        self.y=y.astype(int)
        self.x=x.astype(int)


        # print('x.shape=',x.shape)
        # print('y.shape=',y.shape)


        # calculate intensities:
            # i = N_angles – number of angles sampled
            # j = N_peaks – number of peaks
            # k = r – number of points between peak and end point
            # value at i,j,k = intensity values at particular x,y coordinates
        self.I = self.img[self.y,self.x]

       
    def plot_intensities(self):
        print('plotting intensities...')

        # scatter plots for intensity distributions
        self.N_peaks = len(self.I[1,:,1])
        self.N_angles = len(self.I[:,1,1])

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

    def plot_lines(self):
        print('plotting lines...')
        # plot lines onto image
        xflat = self.x.ravel() #flattening 3d array (x) into 1d array for plotting
        yflat = self.y.ravel() #flattening 3d array (y) into 1d array for plotting

        plt.figure()
        plt.imshow(self.img)
        plt.autoscale(False)
        plt.plot(xflat,yflat, 'r.', markersize=0.25)
        plt.gca().invert_yaxis()
        plt.show()

    def plot_psuedorad_av(self):
         #calculate pseudoradial averages:
        # calculating mean and std from distance from centre
        print('calculating pseudo-radial mean and std...')
        self.I_mean = np.mean(self.I, axis=0)
        self.I_std = np.std(self.I, axis=0)

        print('plotting pseudo-radial averages...')
        rows = self.N_peaks
        fig, ax = plt.subplots(rows, ncols=1,figsize=(6, 3*rows+1),sharex=True)
        #fig.suptitle('Intensity distribution from maxima')
        plt.setp(ax[-1],xlabel = 'Distance from maximum (pixels)')
        plt.setp(ax[:],ylabel = 'Intensity')

        self.x_psradav = np.arange(0,self.r+1,1)

        for i in range(rows):
            ax[i].plot(self.x_psradav, self.I_mean[i,:])
            ax[i].fill_between(self.x_psradav, self.I_mean[i,:]-self.I_std[i,:], self.I_mean[i,:]+self.I_std[i,:], alpha=0.2)

        fig.tight_layout()
        plt.show()

    def radial_average(self):
        y, x = np.indices((self.img.shape)) #y = list of indices row-wise, x - list of indices column wise 

        self.radialav_list = np.empty([self.N_peaks,self.r])
        self.radialstd_list = np.empty([self.N_peaks,self.r])

        self.x_radav = np.arange(0,self.r,1)

        for i, coord in enumerate(self.coords_filter):
            x_img = coord[1]
            y_img = coord[0]        
            distmap = np.sqrt((x - x_img)**2 + (y - y_img)**2) #based on distance formula — computes Euclidean distance from centre for all points in array, creating distance field
            distmap = distmap.astype(int)

            distmap = distmap.ravel()
            intensities = self.img.ravel()
            
            #cutting off distances larger than defined radius
            mask = (distmap>self.r)
            mask = mask.ravel()

            distmap = np.delete(distmap, mask)
            intensities = np.delete(intensities, mask)

            self.radialav_list[i], mbinedges, mbinnumber = stats.binned_statistic(distmap, intensities,statistic='mean',bins=self.r)
            self.radialstd_list[i], sbinedges, sbinnumber = stats.binned_statistic(distmap, intensities, statistic='std',bins=self.r)
        
    def radial_average_norm(self):
        y, x = np.indices((self.img.shape)) #y = list of indices row-wise, x - list of indices column wise 
        
        self.radialavnorm_list = np.empty([self.N_peaks,self.r])
        self.radialstdnorm_list = np.empty([self.N_peaks,self.r])

        self.x_radav = np.arange(0,self.r,1)

        for i, coord in enumerate(self.coords_filter):
            x_img = coord[1]
            y_img = coord[0]        
            distmap = np.sqrt((x - x_img)**2 + (y - y_img)**2) #based on distance formula — computes Euclidean distance from centre for all points in array, creating distance field
            distmap = distmap.astype(int)

            distmap = distmap.ravel()
            intensities = self.img/self.img[y_img,x_img]
            intensities = intensities.ravel()
            
            #cutting off distances larger than defined radius
            mask = (distmap>self.r)

            distmap = np.delete(distmap, mask)
            intensities = np.delete(intensities, mask)

            self.radialavnorm_list[i], mbinedges, mbinnumber = stats.binned_statistic(distmap, intensities,statistic='mean',bins=self.r)
            self.radialstdnorm_list[i], sbinedges, sbinnumber = stats.binned_statistic(distmap, intensities, statistic='std',bins=self.r)

        
    def plotradav(self):
        print('plotting radial average')
        plt.figure()
        rows = self.N_peaks

        fig, ax = plt.subplots(rows, ncols=1, figsize=(6,2*rows+1),sharex=True)
        plt.setp(ax[-1],xlabel = 'Distance from maximum (pixels)')
        plt.setp(ax[:],ylabel = 'Intensity')

        for i in range(rows):
            ax[i].plot(self.x_radav, self.radialav_list[i,:])
            ax[i].fill_between(self.x_radav, self.radialav_list[i,:]-self.radialstd_list[i,:], self.radialav_list[i,:]+self.radialstd_list[i,:], alpha=0.2)
        
        fig.tight_layout()
        plt.show()
    
    def compare_radialmeans(self):

        rows = self.N_peaks
        cols = 2
        
        print('comparing pseudo-radial averages and radial averages...')
        plt.figure()
        fig, ax = plt.subplots(rows, cols, figsize=(12,3*rows+1),sharex=True)
        plt.setp(ax[-1],xlabel = 'Distance from maximum (pixels)')
        plt.setp(ax[:],ylabel = 'Intensity')

        for i in range(rows):
            # plot psuedo-radavs
            ax[i,0].plot(self.x_psradav, self.I_mean[i,:])
            ax[i,0].fill_between(self.x_psradav, self.I_mean[i,:]-self.I_std[i,:], self.I_mean[i,:]+self.I_std[i,:], alpha=0.2)

            # plot radavs
            ax[i,1].plot(self.x_radav, self.radialav_list[i,:])
            ax[i,1].fill_between(self.x_radav, self.radialav_list[i,:]-self.radialstd_list[i,:], self.radialav_list[i,:]+self.radialstd_list[i,:], alpha=0.2)

        fig.tight_layout()
        plt.show()
    
    
    def cluster_averaging(self):
        """
        averaging distributions over all clusters
        """
        self.dropletmean = np.mean(self.radialav_list, axis=0)
        self.dropletstd = np.std(self.radialav_list, axis=0)

        plt.figure()
        plt.plot(self.dropletmean)
        plt.fill_between(self.x_radav,self.dropletmean - self.dropletstd, self.dropletmean+self.dropletstd, alpha=0.2)
        plt.ylim([np.min(self.dropletmean-self.dropletstd),np.max(self.dropletmean+self.dropletstd)])
        plt.xlim([0,np.max(self.x_radav)])
        plt.xlabel('Distance from maximum (pixels)')
        plt.ylabel('Intensity')
        plt.show()

    def cluster_averaging_norm(self):
        """
        averaging distributions over all clusters
        """
        self.dropletmean = np.mean(self.radialavnorm_list, axis=0)
        self.dropletstd = np.std(self.radialstdnorm_list, axis=0)

        plt.figure()
        plt.plot(self.dropletmean)
        plt.fill_between(self.x_radav,self.dropletmean - self.dropletstd, self.dropletmean+self.dropletstd, alpha=0.2)
        plt.ylim([np.min(self.dropletmean-self.dropletstd),np.max(self.dropletmean+self.dropletstd)])
        plt.xlim([0,np.max(self.x_radav)])
        plt.xlabel('Distance from maximum (pixels)')
        plt.ylabel('Normalised Intensity')
        plt.show()


"""
-------------
"""

# Gen_new_img = False
# imgsize = 1000
# N_blobs = 50
# addnoise = True
# filename = 'solid'

# gen_img = GenerateImage(Gen_new_img,imgsize,N_blobs,addnoise, filename)
# peaks = ImageProcessing(gen_img.img)

# img = gen_img.img
# peaklist = peaks.coords
# r = 150
# dtheta = 1
# pl_intensities=False
# pl_lines=False
# pseudoradav=False
# radav=True
# pl_pseudoradav = False
# pl_radav = False
# compare = False
# clust_av = True


# gen_dists = GenerateDistributions(img,peaklist,r,dtheta,pl_intensities,pl_lines,pseudoradav, pl_pseudoradav, radav, pl_radav, compare, clust_av)