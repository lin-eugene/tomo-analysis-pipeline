from matplotlib import markers
import numpy as np
import sys, copy, os
from scipy import signal, stats
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import morphology
from scipy import ndimage as ndi
import pickle
from lib.editmrc import *
import plotly.graph_objects as go
import plotly.io as pio
from lib.analysis import *
pio.renderers.default = "notebook"

def normalise_data(data, plthist=False):
    print("Normalising data...")
    data1d = data.flatten() #turning mrcfile into 1d array

    #calculating statistics
    mean = np.mean(data1d)
    stdev = np.std(data1d)

    data = (data - mean)/stdev

    print("Normalised.")

    return data

def compute_edf(data):
    z, y, x = np.indices(data.shape)

    #find center of image
    center = np.asarray(data.shape)
    center = center/2 
    center = center.astype(int)

    edf = np.sqrt((x-center[2])**2 + (y-center[1])**2 + (z-center[0])**2) #Euclidean distance field

    return edf, center

def compute_edf2d(data):
    y, x = np.indices(data.shape)

    #find center of image
    center = np.asarray(data.shape)
    center = center/2 
    center = center.astype(int)

    edf = np.sqrt((x-center[1])**2 + (y-center[0])**2) #Euclidean distance field

    return edf, center

def lowpass_butter(data, f, n, pltfts=False, pltmask=False):
    """
    data = data
    f = frequency cut-off
    n = steepness
    """
    data_freq = np.fft.fftn(data) #FT of 3D tomogram
    data_fshift = np.fft.fftshift(data_freq)

    edf, center = compute_edf2d(data_fshift)
    cent_slice = center[0]

    if(pltfts):
        plt.figure()
        plt.subplot(121)
        plt.imshow(20*np.log(np.abs(data_freq[cent_slice,:,:])),cmap='gray')

        plt.subplot(122)
        plt.imshow(20*np.log(np.abs(data_fshift[cent_slice,:,:])),cmap='gray')
        plt.gca().invert_yaxis()
        plt.title(f'FFT, central z-slice = {cent_slice}')
        plt.show()

    butter = (1/((1+(edf/f))**(2*n))) #filter
    
    data_fshift = data_fshift * butter
    
    if(pltmask):
        plt.figure()
        plt.subplot(121)
        plt.imshow(butter[cent_slice,:,:],cmap='gray')

        plt.subplot(122)
        plt.imshow( (20*np.log10(0.01+np.abs(data_fshift[cent_slice,:,:]))),cmap='gray')
        plt.gca().invert_yaxis()
        plt.show()

    data_filt = np.real(np.fft.ifftn(np.fft.ifftshift(data_fshift)))
    
    return data_filt

def remove_small_obj(thres, img, minsize, plot=False):
    thres_img = thres
    cleaned = morphology.remove_small_objects(thres_img, min_size = minsize)

    if(plot):
        fig, ax = plt.subplots(1, 3, figsize=(15,5))

        ax[0].imshow(img, cmap='gray')
        ax[0].invert_yaxis()

        ax[1].imshow(thres_img, cmap='gray')
        ax[1].invert_yaxis()

        ax[2].imshow(cleaned, cmap='gray')
        ax[2].invert_yaxis()
        plt.show()

    return cleaned

def threshold(proj, value=0.2, plot=False):
    thres = (proj > value)
    thres = thres.astype(bool)

    if(plot):
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize = (10,20))

        ax1[0].imshow(proj,cmap='gray')
        ax1[0].invert_yaxis()
        ax1[0].set_title('projection')
        
        ax1[1].imshow(thres, cmap='gray')
        ax1[1].invert_yaxis()
        ax1[1].set_title(f'threshold = {value}')

        fig1.tight_layout()
        plt.show()

    return thres

class ImageProcessing():
    def __init__(self, img, filename, slice=0, mode='segmentation', baseline=False, f=2500, n=1, save=True):
        """
        inputs:
        - img - thresholded image
        - filename
        - slice
        - mode — 'segmentation' or 'findpeaks'
        - baseline — are you baselining/sanity checking?
        - f — pixel frequency cutoff (for lp filtering segments)
        - n — steepness of lpfilter cutoff
        - save
        """
        self.img = img
        self.slice = slice

        self.filename = filename
        self.filename = self.filename.split('.')[0]
        self.session = self.filename.split('/')[-2]
        self.tomo = self.filename.split('/')[-1]

        self.f = f
        self.n = n

        if(baseline==False):
            if(mode=='segmentation'):
                self.segmentation()
                self.lowpass_segments()
                if(save):
                    self.save(mode)
            
            if(mode=='findpeaks'):
                self.find_peaks()
                if(save):
                    self.save(mode)
        
        if(baseline):
            self.segmentation()
            self.lowpass_segments()
            self.save_baseline()
    
    def segmentation(self):
        distance = ndi.distance_transform_edt(self.img) #Euclidean distance transform
        self.coords = peak_local_max(distance, footprint = np.ones((3,3)), labels=self.img)
        mask_a = np.zeros(distance.shape, dtype=bool)
        mask_a[tuple(self.coords.T)] = True
        markers, _ = ndi.label(mask_a)
        self.labels = watershed(-distance, markers, mask=self.img) #find labels/segments
    
    def lowpass_segments(self):
        self.labels = self.labels > 0
        self.labels_filt = lowpass_butter(self.labels,self.f,self.n)
        self.labels_filt = self.labels_filt*255/np.max(self.labels_filt)
        self.labels_filt = self.labels_filt.astype(np.uint8)

    def find_peaks(self):
        """
        finds maxima of peaks in image
        requires image as input
        inputs:
            - image

        outputs:
            coords - coordinates of maxima in image
        """
        # self.img = self.labels_filt>0
        # self.img = self.img.astype(int)
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
    
    
    def save(self,mode):
        if(os.path.exists('./data/peaklist/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/peaklist/'+self.session+'/'+self.tomo)
        if(os.path.exists('./data/segmentation/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/segmentation/'+self.session+'/'+self.tomo)

        np.save('data/peaklist/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_peaklist.npy', self.coords)
        
        if(mode=="segmentation"):
            np.save('data/segmentation/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_segments.npy', self.labels_filt)
    
    def save_baseline(self):
        if(os.path.exists('./data/peaklist/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/peaklist/'+self.session+'/'+self.tomo)
        if(os.path.exists('./data/segmentation/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/segmentation/'+self.session+'/'+self.tomo)

        np.save('data/peaklist/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blinepeaklist.npy', self.coords)

###########
###########


class GenerateDistributions():
    def __init__(self, img, peaklist, r, outfilename,
        pl_radav=False, 
        clust_av=False,
        pl_cluster=False,
        baseline=False,
        save=True,
        slice=0):
        """
        Class to plot out spatial distribution information from image
        
        Arguments:
        - Inputs
            - img —- image (2D numpy array),
            - peaklist -- x,y-coords of peaks

        - Parameters
            - r -- sampling radius, length of line drawn out from centre of peak
            - dtheta -- sampling angle interval
            - pl_radav -- do you want to plot radial average results
            - clust_av -- do you want to average all radial intensity profiles together
            - pl_cluster -- do you want to plot averaged radial intensity profiles?
        """
        self.img = img
        self.size = img.shape


        self.slice=slice

        self.filename = outfilename
        self.filename = self.filename.split('.')[0]
        self.session = self.filename.split('/')[-2]
        self.tomo = self.filename.split('/')[-1]

        # define angle and length of line from local maximum
        self.r = r

        #peak filtering
        self.peaklist = peaklist
        self.x_peaks = self.peaklist[:,1]
        self.y_peaks = self.peaklist[:,0]



        self.filter_peaks()

        #creating distributions - method 1: draw_lines
       
        self.radial_average()
        self.radial_average_norm()

        if(baseline==False):
            if(pl_radav):
                self.plotradav()

            if(clust_av):
                self.cluster_averaging()
                self.cluster_averaging_norm()
                if(pl_cluster):
                    self.plot_cluster()

            if(save):
                self.slice = slice
                self.save_radav()
        
        if(baseline):
            self.slice = slice
            self.save_baseline()

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

        self.N_peaks = len(self.x_peaks) #Number of peaks remaining
        N_peaks_disttoosmall = sum(disttoosmall) #Number of peaks deleted

        print(N_peaks_disttoosmall, "peak(s) too close to edge")
        print(f'Sampling from {self.N_peaks} peaks')

        if(self.N_peaks==0):
            print('No peaks selected, radius too large')
            
        else:
            pass

        print('Creating plot...')
        plt.figure(figsize=(7,7))
        plt.imshow(self.img,cmap='gray')
        plt.autoscale(False)
        plt.plot(self.x_peaks,self.y_peaks, 'r.')
        plt.gca().invert_yaxis()
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
    
    def cluster_averaging(self):
        """
        averaging distributions over all clusters
        """
        self.dropletmean = np.mean(self.radialav_list, axis=0)
        self.dropletstd = np.std(self.radialav_list, axis=0)

    def cluster_averaging_norm(self):
        """
        averaging distributions over all clusters
        """
        self.dropletmean_norm = np.mean(self.radialavnorm_list, axis=0)
        self.dropletstd_norm = np.std(self.radialavnorm_list, axis=0)

    def plot_cluster(self):
        plt.figure()
        plt.plot(self.dropletmean)
        plt.fill_between(self.x_radav,self.dropletmean - self.dropletstd, self.dropletmean+self.dropletstd, alpha=0.2)
        plt.ylim([np.min(self.dropletmean-self.dropletstd),np.max(self.dropletmean+self.dropletstd)])
        plt.xlim([0,np.max(self.x_radav)])
        plt.axhline(y=0,color='k')
        plt.xlabel('Distance from maximum (pixels)')
        plt.ylabel('Intensity')
        plt.show()

        plt.figure()
        plt.plot(self.dropletmean_norm)
        plt.fill_between(self.x_radav,self.dropletmean_norm - self.dropletstd_norm, self.dropletmean_norm+self.dropletstd_norm, alpha=0.2)
        plt.ylim([np.min(self.dropletmean_norm-self.dropletstd_norm),np.max(self.dropletmean_norm+self.dropletstd_norm)])
        plt.xlim([0,np.max(self.x_radav)])
        plt.axhline(y=0,color='k')
        plt.xlabel('Distance from maximum (pixels)')
        plt.ylabel('Normalised Intensity')
        plt.show()
    
    def save_radav(self):
        if(os.path.exists(f'./data/radavdata/radav/{self.session}/{self.tomo}')==False):
            os.makedirs('./data/radavdata/radav/'+self.session+'/'+self.tomo)

        np.save('data/radavdata/radav/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_radavlist.npy', self.radialav_list)
        np.save('data/radavdata/radav/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_radavstd.npy', self.radialstd_list)
    
        if(os.path.exists('./data/radavdata/radavnorm/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/radavdata/radavnorm/'+self.session+'/'+self.tomo)

        np.save('data/radavdata/radavnorm/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_radavnormlist.npy', self.radialavnorm_list)
        np.save('data/radavdata/radavnorm/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_radavnormstd.npy', self.radialstdnorm_list)

        if(os.path.exists('./data/peaklist/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/peaklist/'+self.session+'/'+self.tomo)

        np.save('data/peaklist/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_filteredpeaklist.npy', self.coords_filter)
    
    def save_baseline(self):
        if(os.path.exists(f'./data/radavdata/radav/{self.session}/{self.tomo}')==False):
            os.makedirs('./data/radavdata/radav/'+self.session+'/'+self.tomo)

        np.save('data/radavdata/radav/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blradavlist.npy', self.radialav_list)
        np.save('data/radavdata/radav/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blradavstd.npy', self.radialstd_list)

        if(os.path.exists('./data/radavdata/radavnorm/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/radavdata/radavnorm/'+self.session+'/'+self.tomo)

        np.save('data/radavdata/radavnorm/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blradavnormlist.npy', self.radialavnorm_list)
        np.save('data/radavdata/radavnorm/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blradavnormstd.npy', self.radialstdnorm_list)

        if(os.path.exists('./data/peaklist/'+self.session+'/'+self.tomo)==False):
            os.makedirs('./data/peaklist/'+self.session+'/'+self.tomo)

        np.save('data/peaklist/'+self.session+'/'+self.tomo+'/'+str(self.slice)+'_blfilteredpeaklist.npy', self.coords_filter)

class sampling():
    def __init__(self, filename, slice, n=50, mode='peaks',
        logsigma=5,threslogval=0.005,thresabs=3,minsize=100,coarsemap=False):
        """
        mode — peaks, uniform, random
        """
        #initialising
        self.filename = filename
        self.filename = self.filename.split('.')[0]
        self.session = self.filename.split('/')[-2]
        self.tomo = self.filename.split('/')[-1]

        self.file = mrcfile.open("./raw/"+filename+".mrc")
        print("Tomogram dimensions (Z,Y,X) = ",self.file.data.shape)
        vox_size = self.file.voxel_size.view((self.file.voxel_size.dtype[0], len(self.file.voxel_size.dtype.names)))
        self.vox_size = vox_size[0]
        self.data = copy.deepcopy(self.file.data) * -1 #copy data and invert intensities
        self.file.close()
        self.data = normalise_data(self.data)

        

        self.slice = slice
        self.proj = self.data[self.slice,:,:]
        self.n = n

        self.mode = mode

        if(coarsemap==True):
            self.coarse_mapname = f'data/masks/{self.session}/{self.tomo}_coarsemap.npy'
            self.coarsemap = np.load(self.coarse_mapname)[slice,:,:]

        if(mode=='peaks'):
            self.logsigma = logsigma
            self.threslogval = threslogval
            self.minsize = minsize
            self.threshabs = thresabs
            
            self.blurthresh()
            self.removesmallobj()

            if(coarsemap==True):
                self.excludesolvent()

            self.gendistancefield()
            self.pickpeaks()

        if(mode=='uniform'):
            self.genuniformsample()

            if(coarsemap==True):
                self.removepoints()

        if(mode=='random'):
            self.genrandomsample()

            if(coarsemap==True):
                self.removepoints()
        
        self.computerdf()
        self.computesupercluster()
        self.save()
    
    def blurthresh(self):
        LoG = -ndi.gaussian_laplace(self.proj, sigma=self.logsigma)

        self.thres_LoG = threshold(LoG, value=self.threslogval, plot=False)


        fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (20,100))
        ax[0].imshow(self.proj,cmap='gray')
        ax[0].invert_yaxis()
        ax[0].set_title('projection')

        ax[1].imshow(LoG,cmap='gray')
        ax[1].invert_yaxis()
        ax[1].set_title('LoG')

        ax[2].imshow(self.thres_LoG,cmap='gray')
        ax[2].invert_yaxis()
        ax[2].set_title('thres_LoG')

        plt.show()
    
    def removesmallobj(self):
        minsize = self.minsize
        plot = True
        cleaned = remove_small_obj(self.thres_LoG, self.proj, minsize, plot)
        self.cleaned = ndi.morphology.binary_fill_holes(cleaned)
    
    def excludesolvent(self):
        self.cleaned = self.cleaned*self.coarsemap
        cleanedplot = self.cleaned.astype(int)
        plt.figure()
        plt.imshow(self.cleaned,cmap='gray')
        plt.gca().invert_yaxis()
        plt.show()
    
    def gendistancefield(self):

        self.distance = ndi.distance_transform_edt(self.cleaned) #Euclidean distance transform
        self.distance_plot = copy.deepcopy(self.distance)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        ax[0].imshow(self.proj,cmap='gray')
        ax[0].invert_yaxis()

        ax[1].imshow(self.distance_plot,cmap='gray')
        ax[1].invert_yaxis()
        plt.show()

        xx, yy = np.mgrid[0:self.distance.shape[0],0:self.distance.shape[1]]

        fig = go.Figure(data=[go.Surface(z=self.distance_plot)])
        fig.show()
    
    def pickpeaks(self):
        self.coords = peak_local_max(self.distance, footprint = np.ones((3,3)), labels=self.thres_LoG, threshold_abs=self.threshabs)

    def genuniformsample(self):
        x_imgsize = self.data.shape[2]
        y_imgsize = self.data.shape[1]
        n = self.n
        x = np.linspace(0, x_imgsize-1, n).astype(int)
        y = np.linspace(0, y_imgsize-1, n).astype(int)

        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1,1)
        yy = yy.reshape(-1,1)
        self.coords = np.concatenate((yy,xx),axis=1)

    def genrandomsample(self):
        npoints = self.n**2
        rand_x = np.random.randint(low=0, high=self.data.shape[2], size=npoints).astype(int)
        rand_y = np.random.randint(low=0, high=self.data.shape[1], size=npoints).astype(int)
        rand_x = rand_x.reshape(-1,1)
        rand_y = rand_y.reshape(-1,1)
        self.coords = np.concatenate((rand_y,rand_x),axis=1)

    def removepoints(self):
        coords = self.coords
        coarsemap = self.coarsemap

        x = coords[:,1]
        y = coords[:,0]

        what = coarsemap[y,x]
        x = np.delete(x, np.argwhere(what==0))
        y = np.delete(y, np.argwhere(what==0))

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        self.coords = np.concatenate((y,x),axis=1)

    def computerdf(self):
        self.gendists = GenerateDistributions(self.proj,peaklist=self.coords,r=70,outfilename=self.filename, clust_av=True, pl_cluster=True, save=False)

        plt.figure()
        plt.plot(self.gendists.x_radav*self.vox_size,self.gendists.dropletmean)
        plt.axhline(y=0,color='k')
        plt.xlabel('Distance from maximum (Angstroms)')
        plt.ylabel('Intensity')
        plt.show()
    
    def computesupercluster(self):
        def imgaveraging(peaklist,img,n):
            '''
            peaklist — list of filtered peaks
            img = z-slice
            n = radius/half width of box
            '''
            x_coords = peaklist[:,1]
            y_coords = peaklist[:,0]   
            box = np.zeros((peaklist.shape[0],n*2, n*2))
            print(box.shape)
            for i, coord in enumerate(peaklist):
                x = coord[1]
                y = coord[0]

                box[i,:,:] = img[y-n:y+n,x-n:x+n]

            box_mean = np.mean(box,axis=0)


            plt.figure()
            plt.imshow(img,cmap='gray')
            plt.gca().invert_yaxis()
            plt.plot(x_coords,y_coords,'r.')
            plt.show()

            plt.figure()
            plt.imshow(box_mean,cmap='gray')
            plt.gca().invert_yaxis()
            plt.show()

            return box, box_mean
        
        peaklist = self.gendists.coords_filter
        img = self.proj
        n = 75
        box, box_mean = imgaveraging(peaklist, img, n)
    
    def save(self):
        if(os.path.exists(f'./data/sampling/{self.session}/{self.tomo}')==False):
            os.makedirs(f'./data/sampling/{self.session}/{self.tomo}')
        
        np.save(f'./data/sampling/{self.session}/{self.tomo}/{self.slice}_{self.mode}_radavlist.npy', self.gendists.radialav_list)
        np.save(f'./data/sampling/{self.session}/{self.tomo}/{self.slice}_{self.mode}_radavstd.npy', self.gendists.radialstd_list)
        np.save(f'./data/sampling/{self.session}/{self.tomo}/{self.slice}_{self.mode}_radavnormlist.npy', self.gendists.radialavnorm_list)
        np.save(f'./data/sampling/{self.session}/{self.tomo}/{self.slice}_{self.mode}_radanormvstd.npy', self.gendists.radialstdnorm_list)


class samplingplot():
    def __init__(self,filename,datafilename,bgfilename,dataslice,bgslice):
         #initialise names
        self.filename = filename.split('.')[0]
        self.session = self.filename.split('/')[-2]
        self.tomo = self.filename.split('/')[-1]

        self.dataslice = dataslice
        self.bgslice = bgslice

        #load files
        with mrcfile.open(f'./raw/{self.filename}.mrc') as mrc:
            self.rawtomo = copy.deepcopy(mrc.data) * -1
            self.voxel_size = mrc.voxel_size

            self.vox_size = self.voxel_size.view((self.voxel_size.dtype[0], len(self.voxel_size.dtype.names)))
            self.vox_size = self.vox_size[0]

        #load data rdfs
        self.datafilename = datafilename
        self.datasession = self.datafilename.split('/')[-2]
        self.datatomo = self.datafilename.split('/')[-1]

        self.datapeaks = np.load(f'./data/sampling/{self.session}/{self.datatomo}/{self.dataslice}_peaks_radavlist.npy')
        # self.datauniform = np.load(f'./data/sampling/{self.session}/{self.datatomo}/{self.dataslice}_uniform_radavlist.npy')
        # self.datarandom = np.load(f'./data/sampling/{self.session}/{self.datatomo}/{self.dataslice}_random_radavlist.npy')

        #load bg rdfs
        self.bgfilename = bgfilename
        self.bgsession = self.bgfilename.split('/')[-2]
        self.bgtomo = self.bgfilename.split('/')[-1]

        # self.bgpeaks = np.load(f'./data/sampling/{self.session}/{self.bgtomo}/{self.bgslice}_peaks_radavlist.npy')
        # self.bguniform = np.load(f'./data/sampling/{self.session}/{self.bgtomo}/{self.bgslice}_uniform_radavlist.npy')
        self.bgrandom = np.load(f'./data/sampling/{self.session}/{self.bgtomo}/{self.bgslice}_random_radavlist.npy')
        
        self.plotrdfslices()
        self.plotnormrdfslices()

    
    def plotrdfslices(self):

        r = self.datapeaks.shape[1]
        self.x_radav = np.arange(0,r,1)*self.vox_size

        self.datapeaksmean = np.mean(self.datapeaks,axis=0)
        self.datapeaksstd = np.std(self.datapeaks,axis=0)
        self.bgrandommean = np.mean(self.bgrandom,axis=0)
        self.bgrandomstd = np.std(self.bgrandom,axis=0)

        self.datapeaks_corr = self.datapeaks-self.bgrandommean
        self.datapeaks_corr_mean = np.mean(self.datapeaks_corr,axis=0)
        self.datapeaks_corr_std = np.std(self.datapeaks_corr,axis=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,5))
        ax[0].plot(self.x_radav,self.datapeaksmean)
        ax[0].fill_between(self.x_radav,self.datapeaksmean - self.datapeaksstd, self.datapeaksmean+self.datapeaksstd, alpha=0.2)
        ax[0].set_ylim([np.min(self.datapeaksmean-self.datapeaksstd),np.max(self.datapeaksmean+self.datapeaksstd)])
        ax[0].set_xlim([0,np.max(self.x_radav)])
        ax[0].axhline(y=0,color='k')
        ax[0].set_xlabel('Distance from maximum (Angstroms)')
        ax[0].set_ylabel('Intensity')
        ax[0].set_title('Data')

        ax[1].plot(self.x_radav,self.bgrandommean)
        ax[1].fill_between(self.x_radav,self.bgrandommean - self.bgrandomstd, self.bgrandommean+self.bgrandomstd, alpha=0.2)
        ax[1].set_ylim([np.min(self.bgrandommean-self.bgrandomstd),np.max(self.bgrandommean+self.bgrandomstd)])
        ax[1].set_xlim([0,np.max(self.x_radav)])
        ax[1].axhline(y=0,color='k')
        ax[1].set_xlabel('Distance from maximum (Angstroms)')
        ax[1].set_ylabel('Intensity')
        ax[1].set_title('Background')

        ax[2].plot(self.x_radav,self.datapeaks_corr_mean)
        ax[2].fill_between(self.x_radav,self.datapeaks_corr_mean - self.datapeaks_corr_std, self.datapeaks_corr_mean+self.datapeaks_corr_std, alpha=0.2)
        ax[2].set_ylim([np.min(self.datapeaks_corr_mean-self.datapeaks_corr_std),np.max(self.datapeaks_corr_mean+self.datapeaks_corr_std)])
        ax[2].set_xlim([0,np.max(self.x_radav)])
        ax[2].axhline(y=0,color='k')
        ax[2].set_xlabel('Distance from maximum (Angstroms)')
        ax[2].set_ylabel('Intensity difference')
        ax[2].set_title('Data (peaks) - background (random)')

        plt.show()

    def plotnormrdfslices(self):

        r = self.datapeaks.shape[1]
        self.x_radav = np.arange(0,r,1)*self.vox_size

        self.datapeaksmean = np.mean(self.datapeaks,axis=0)
        self.datapeaksstd = np.std(self.datapeaks,axis=0)
        self.bgrandommean = np.mean(self.bgrandom,axis=0)
        self.bgrandomstd = np.std(self.bgrandom,axis=0)

        self.datapeaks_corr = self.datapeaks-self.bgrandommean
        self.datapeaks_corr_mean = np.mean(self.datapeaks_corr,axis=0)
        self.datapeaks_corr_mean_norm = self.datapeaks_corr_mean/np.max(self.datapeaks_corr_mean)
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,5))
        ax[0].plot(self.x_radav,self.datapeaksmean)
        ax[0].fill_between(self.x_radav,self.datapeaksmean - self.datapeaksstd, self.datapeaksmean+self.datapeaksstd, alpha=0.2)
        ax[0].set_ylim([np.min(self.datapeaksmean-self.datapeaksstd),np.max(self.datapeaksmean+self.datapeaksstd)])
        ax[0].set_xlim([0,np.max(self.x_radav)])
        ax[0].axhline(y=0,color='k')
        ax[0].set_xlabel('Distance from maximum (Angstroms)')
        ax[0].set_ylabel('Intensity')
        ax[0].set_title('Data')

        ax[1].plot(self.x_radav,self.bgrandommean)
        ax[1].fill_between(self.x_radav,self.bgrandommean - self.bgrandomstd, self.bgrandommean+self.bgrandomstd, alpha=0.2)
        ax[1].set_ylim([np.min(self.bgrandommean-self.bgrandomstd),np.max(self.bgrandommean+self.bgrandomstd)])
        ax[1].set_xlim([0,np.max(self.x_radav)])
        ax[1].axhline(y=0,color='k')
        ax[1].set_xlabel('Distance from maximum (Angstroms)')
        ax[1].set_ylabel('Intensity')
        ax[1].set_title('Background')

        ax[2].plot(self.x_radav,self.datapeaks_corr_mean_norm)
        ax[2].set_ylim([np.min(self.datapeaks_corr_mean_norm)-0.1,np.max(self.datapeaks_corr_mean_norm)])
        ax[2].set_xlim([0,np.max(self.x_radav)])
        ax[2].axhline(y=0,color='k')
        ax[2].set_xlabel('Distance from maximum (Angstroms)')
        ax[2].set_ylabel('Normalised intensity')
        ax[2].set_title('Data (peaks) - background (random)')

        plt.show()

    
    def plotrdfslices_deprecated(self):
        """
        deprecated — decide to use data peaks - bg random
        """
        self.datapeaksmean = np.mean(self.datapeaks,axis=0)
        self.datauniformmean = np.mean(self.datauniform,axis=0)
        self.datarandommean = np.mean(self.datarandom,axis=0)

        self.datapeaksstd = np.std(self.datapeaks,axis=0)
        self.datauniformstd = np.std(self.datauniform,axis=0)
        self.datarandomstd = np.std(self.datarandom,axis=0)

        self.bgpeaksmean = np.mean(self.bgpeaks,axis=0)
        self.bguniformmean = np.mean(self.bguniform,axis=0)
        self.bgrandommean = np.mean(self.bgrandom,axis=0)

        self.bgpeaksstd = np.std(self.bgpeaks,axis=0)
        self.bguniformstd = np.std(self.bguniform,axis=0)
        self.bgrandomstd = np.std(self.bgrandom,axis=0)

        self.peaksdiff = self.datapeaksmean - self.bgpeaksmean
        self.uniformdiff = self.datauniformmean - self.bguniformmean
        self.randomdiff = self.datarandommean - self.bgrandommean

        r = self.datapeaks.shape[1]
        x_radav = np.arange(0,r,1)*self.vox_size

        fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
        fig1.suptitle('Peak picking')
        ax1[0].plot(x_radav,self.datapeaksmean)
        ax1[0].fill_between(x_radav,self.datapeaksmean - self.datapeaksstd, self.datapeaksmean+self.datapeaksstd, alpha=0.2)
        ax1[0].set_ylim([np.min(self.datapeaksmean-self.datapeaksstd),np.max(self.datapeaksmean+self.datapeaksstd)])
        ax1[0].set_xlim([0,np.max(x_radav)])
        ax1[0].set_xlabel('Distance from maximum (Angstroms)')
        ax1[0].set_ylabel('Intensity')
        ax1[0].set_title('Data')

        ax1[1].plot(x_radav,self.bgpeaksmean)
        ax1[1].fill_between(x_radav,self.bgpeaksmean - self.bgpeaksstd, self.bgpeaksmean+self.bgpeaksstd, alpha=0.2)
        ax1[1].set_ylim([np.min(self.bgpeaksmean-self.bgpeaksstd),np.max(self.bgpeaksmean+self.bgpeaksstd)])
        ax1[1].set_xlim([0,np.max(x_radav)])
        ax1[1].set_xlabel('Distance from maximum (Angstroms)')
        ax1[1].set_ylabel('Intensity')
        ax1[1].set_title('Background')

        ax1[2].plot(x_radav,self.peaksdiff)
        ax1[2].set_ylim([np.min(self.peaksdiff),np.max(self.peaksdiff)])
        ax1[2].set_xlim([0,np.max(x_radav)])
        ax1[2].set_xlabel('Distance from maximum (Angstroms)')
        ax1[2].set_ylabel('Intensity difference')
        ax1[2].set_title('Data-Background')

        fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
        fig2.suptitle('Uniform sampling')
        ax2[0].plot(x_radav,self.datauniformmean)
        ax2[0].fill_between(x_radav,self.datauniformmean - self.datauniformstd, self.datauniformmean+self.datauniformstd, alpha=0.2)
        ax2[0].set_ylim([np.min(self.datauniformmean-self.datauniformstd),np.max(self.datauniformmean+self.datauniformstd)])
        ax2[0].set_xlim([0,np.max(x_radav)])
        ax2[0].set_xlabel('Distance from maximum (Angstroms)')
        ax2[0].set_ylabel('Intensity')
        ax2[0].set_title('Data')

        ax2[1].plot(x_radav,self.bguniformmean)
        ax2[1].fill_between(x_radav,self.bguniformmean - self.bguniformstd, self.bguniformmean+self.bguniformstd, alpha=0.2)
        ax2[1].set_ylim([np.min(self.bguniformmean-self.bguniformstd),np.max(self.bguniformmean+self.bguniformstd)])
        ax2[1].set_xlim([0,np.max(x_radav)])
        ax2[1].set_xlabel('Distance from maximum (Angstroms)')
        ax2[1].set_ylabel('Intensity')
        ax2[1].set_title('Background')

        ax2[2].plot(x_radav,self.uniformdiff)
        ax2[2].set_ylim([np.min(self.uniformdiff),np.max(self.uniformdiff)])
        ax2[2].set_xlim([0,np.max(x_radav)])
        ax2[2].set_xlabel('Distance from maximum (Angstroms)')
        ax2[2].set_ylabel('Intensity difference')
        ax2[2].set_title('Data-Background')

        fig3, ax3 = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
        fig3.suptitle('Random sampling')
        ax3[0].plot(x_radav,self.datarandommean)
        ax3[0].fill_between(x_radav,self.datarandommean - self.datarandomstd, self.datarandommean+self.datarandomstd, alpha=0.2)
        ax3[0].set_ylim([np.min(self.datarandommean-self.datarandomstd),np.max(self.datarandommean+self.datarandomstd)])
        ax3[0].set_xlim([0,np.max(x_radav)])
        ax3[0].set_xlabel('Distance from maximum (Angstroms)')
        ax3[0].set_ylabel('Intensity')
        ax3[0].set_title('Data')

        ax3[1].plot(x_radav,self.bgrandommean)
        ax3[1].fill_between(x_radav,self.bgrandommean - self.bgrandomstd, self.bgrandommean+self.bgrandomstd, alpha=0.2)
        ax3[1].set_ylim([np.min(self.bgrandommean-self.bgrandomstd),np.max(self.bgrandommean+self.bgrandomstd)])
        ax3[1].set_xlim([0,np.max(x_radav)])
        ax3[1].set_xlabel('Distance from maximum (Angstroms)')
        ax3[1].set_ylabel('Intensity')
        ax3[1].set_title('Background')

        ax3[2].plot(x_radav,self.randomdiff)
        ax3[2].set_ylim([np.min(self.randomdiff),np.max(self.randomdiff)])
        ax3[2].set_xlim([0,np.max(x_radav)])
        ax3[2].set_xlabel('Distance from maximum (Angstroms)')
        ax3[2].set_ylabel('Intensity difference')
        ax3[2].set_title('Data-Background')

        fig4, ax4 = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
        fig4.suptitle('Difference plots')
        ax4[0].plot(x_radav,self.datapeaksmean-self.bgpeaksmean)
        ax4[0].set_xlim([0,np.max(x_radav)])
        ax4[0].set_xlabel('Distance from maximum (Angstroms)')
        ax4[0].set_ylabel('Intensity difference')
        ax4[0].set_title('Data (peaks) - background (peaks)')

        ax4[1].plot(x_radav,self.datapeaksmean-self.bguniformmean)
        ax4[1].set_xlim([0,np.max(x_radav)])
        ax4[1].set_xlabel('Distance from maximum (Angstroms)')
        ax4[1].set_ylabel('Intensity difference')
        ax4[1].set_title('Data (peaks) - background (uniform)')

        ax4[2].plot(x_radav,self.datapeaksmean-self.bgrandommean)
        ax4[2].set_xlim([0,np.max(x_radav)])
        ax4[2].set_xlabel('Distance from maximum (Angstroms)')
        ax4[2].set_ylabel('Intensity difference')
        ax4[2].set_title('Data (peaks) - background (random)')

        plt.show()

