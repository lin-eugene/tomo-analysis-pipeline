#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3

import wx,string,copy,math,numpy,os
import matplotlib            #import matplotlib
matplotlib.use('WXAgg')      #switch on the wxPython mode
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.cm as cm
import matplotlib.colors as colors
#import nmrglue as ng
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Ellipse
import re
from matplotlib.gridspec import GridSpec

import scipy.optimize as opt
import threading
import mrcfile
from skimage import morphology, color
from scipy import ndimage as ndi
import pickle
import cv2
import sys, os.path


##################################################################################################################

matplotlib.rcParams['xtick.labelsize']=8
matplotlib.rcParams['ytick.labelsize']=8




class deconFrame(wx.Panel):
    """ The main frame of the application
    """
    title = 'Demo: wxPython with matplotlib'

    def __init__(self,parent,tabOne):

        wx.Panel.__init__(self,parent=parent)

        # self.thresh=tabOne.dmax*float(tabOne.threshBox.GetValue()) #get threshold from main tab
        # self.tabOne=tabOne    #get tabone panel from NMR tab
        self.parent=parent    #get decon_tab main parent from notebook

        self.sum='yz','CH'
        #self.pdbfile=tabOne.textbox.GetValue() #get the pdbfile from tabOne
        #self.methlist=tabOne.methList          #get the methyl list from tabOne
        self.shiftXrun=0                      #initialise the shiftX run counter
        self.selected = ''
        self.peaks_drawn=False
        self.resized = False # the dirty flag

        self.filename = sys.argv[1] #specify filename (w/o .mrc)
        self.filename = self.filename.split('.')[0]
        self.session = self.filename.split('/')[-2]
        self.tomo = self.filename.split('/')[-1]

        print(self.filename)
        print(type(self.filename))

        if not os.path.isfile(self.filename+'.mrc'):
            print("file not found")
            sys.exit()


        mrc = mrcfile.open(self.filename+'.mrc')
        # mrc.data = mrc.data
        self.data = mrc.data*(-1.)
        self.data = self.normalise_data(self.data)

        self.thresh = numpy.ones(self.data.shape[0])
        self.object_size = numpy.ones(self.data.shape[0])*10
        self.blur_size = numpy.ones(self.data.shape[0])*20
        self.blurthresh_size = numpy.ones(self.data.shape[0])*0.2
        self.mask = numpy.zeros_like(self.data)
        self.objects = numpy.zeros_like(self.data)
        self.threshedraw = numpy.zeros_like(self.data)

        self.load_params()
        self.create_main_panel()
        self.load_decon_data()
        self.draw_figure()
        self.ftol = 1e-10
        self.Bind(wx.EVT_SIZE,self.OnSize)
        self.Bind(wx.EVT_IDLE,self.OnIdle)
        self.dragging_extra_plot = False
        self.drag_extra_plot = False

    
        # self.canvas.draw()
        self.fig.subplots_adjust(left=0.043, right=0.981, top=0.975, bottom=0.098, wspace=0.186, hspace=0.2)
        self.canvas.draw()

        self.Show(True)
        self.Fit()
        # self.background_save(None)



    def OnSize(self,event):
        self.resized = True # set dirty
        if self.GetAutoLayout():
               self.Layout()

    def OnIdle(self,event):
        # print('Resized:', self.resized)
        if self.resized: 
            # take action if the dirty flag is set
            self.background_save(event)
            self.resized = False # reset the flag

    def readfile(self,infile):
        peak=[]
        peakfile=open(infile,'r')
        for line in peakfile.readlines():
            linetosave=line.split()
            peak.append(linetosave)
        peakfile.close()
        return peak

    def drawing_box(self):
        slice_total = len(self.data[:,0,0])
        self.slice = 0
        self.sliceLbl = wx.StaticBox(self,-1,f'Slices: {slice_total}')
        self.sliceSizer=wx.StaticBoxSizer(self.sliceLbl,wx.HORIZONTAL)

        self.slice_text=wx.StaticText(self, -1, 'Slice:')
        self.up_slice = wx.Button(self, -1, "+", size=(30,22))
        self.down_slice = wx.Button(self, -1, "-", size=(30,22))
        self.slice_box = wx.TextCtrl(self, size=(50,22), style=wx.TE_PROCESS_ENTER)

        self.slice_box.SetValue(str(1))
        
        border = 10
        self.sliceSizer.Add(self.slice_text, 0, border=border, flag=self.flags)
        self.sliceSizer.Add(self.slice_box, 0, border=border, flag=self.flags)
        self.sliceSizer.Add(self.up_slice, 0, border=border, flag=self.flags)
        self.sliceSizer.Add(self.down_slice, 0, border=border, flag=self.flags)

        self.Bind(wx.EVT_BUTTON, self.on_up_slice, self.up_slice)
        self.Bind(wx.EVT_BUTTON, self.on_down_slice, self.down_slice)
        self.sliceSizer.AddSpacer(10)


    def contour_box(self):
        self.cntrLbl = wx.StaticBox(self,-1,'Contours:')
        self.cntrSizer=wx.StaticBoxSizer(self.cntrLbl,wx.HORIZONTAL)

        self.text1=wx.StaticText(self, -1, 'Min:')
        self.up_thresh = wx.Button(self, -1, "+", size=(30,22))
        self.down_thresh = wx.Button(self, -1, "-", size=(30,22))
        self.textbox0 = wx.TextCtrl(self, size=(50,22), style=wx.TE_PROCESS_ENTER)

        self.textbox0.SetValue(str(self.thresh[self.slice]))
        
        border = 10
        self.cntrSizer.Add(self.text1, 0, border=border, flag=self.flags)
        self.cntrSizer.Add(self.textbox0, 0, border=border, flag=self.flags)
        self.cntrSizer.Add(self.up_thresh, 0, border=border, flag=self.flags)
        self.cntrSizer.Add(self.down_thresh, 0, border=border, flag=self.flags)

        self.Bind(wx.EVT_BUTTON, self.on_up_thresh, self.up_thresh)
        self.Bind(wx.EVT_BUTTON, self.on_down_thresh, self.down_thresh)
        self.cntrSizer.AddSpacer(10)

    def object_filter_box(self):
        self.object_box = wx.StaticBox(self,-1,'Object Size:')
        self.object_sizer=wx.StaticBoxSizer(self.object_box,wx.HORIZONTAL)

        self.object_text1=wx.StaticText(self, -1, 'Min:')
        self.object_up_size = wx.Button(self, -1, "+", size=(30,22))
        self.object_down_size = wx.Button(self, -1, "-", size=(30,22))
        self.object_textbox = wx.TextCtrl(self, size=(50,22), style=wx.TE_PROCESS_ENTER)

        self.object_textbox.SetValue(str(self.object_size[self.slice]))
       
        border = 10
        self.object_sizer.Add(self.object_text1, 0, border=border, flag=self.flags)
        self.object_sizer.Add(self.object_textbox, 0, border=border, flag=self.flags)
        self.object_sizer.Add(self.object_up_size, 0, border=border, flag=self.flags)
        self.object_sizer.Add(self.object_down_size, 0, border=border, flag=self.flags)



        self.Bind(wx.EVT_BUTTON, self.on_up_size, self.object_up_size)
        self.Bind(wx.EVT_BUTTON, self.on_down_size, self.object_down_size)
        self.object_sizer.AddSpacer(10)

    def blur_filter_box(self):
        self.blur_box = wx.StaticBox(self,-1,'Course Map:')
        
        self.blur_sizer=wx.StaticBoxSizer(self.blur_box,wx.VERTICAL)

        self.blur_sizer1 =wx.BoxSizer(wx.HORIZONTAL)
        self.blur_sizer2 =wx.BoxSizer(wx.HORIZONTAL)
 

        self.blur_text1=wx.StaticText(self, -1, 'STD:')
        self.blur_up_size = wx.Button(self, -1, "+", size=(30,22))
        self.blur_down_size = wx.Button(self, -1, "-", size=(30,22))
        self.blur_textbox = wx.TextCtrl(self, size=(50,22), style=wx.TE_PROCESS_ENTER)

        self.blur_textbox.SetValue(str(self.blur_size[self.slice]))
        
        border = 10
        self.blur_sizer1.Add(self.blur_text1, 0, border=border, flag=self.flags)
        self.blur_sizer1.Add(self.blur_textbox, 0, border=border, flag=self.flags)
        self.blur_sizer1.Add(self.blur_up_size, 0, border=border, flag=self.flags)
        self.blur_sizer1.Add(self.blur_down_size, 0, border=border, flag=self.flags)

        # self.Bind(wx.EVT_BUTTON, self.on_up_size, self.blur_up_size)
        # self.Bind(wx.EVT_BUTTON, self.on_down_size, self.blur_down_size)


        self.blurthresh_text1=wx.StaticText(self, -1, 'Thresh:')
        self.blurthresh_up_size = wx.Button(self, -1, "+", size=(30,22))
        self.blurthresh_down_size = wx.Button(self, -1, "-", size=(30,22))
        self.blurthresh_textbox = wx.TextCtrl(self, size=(50,22), style=wx.TE_PROCESS_ENTER)

        self.blurthresh_textbox.SetValue(str(self.blurthresh_size[self.slice]))
        
        border = 10
        self.blur_sizer2.Add(self.blurthresh_text1, 0, border=border, flag=self.flags)
        self.blur_sizer2.Add(self.blurthresh_textbox, 0, border=border, flag=self.flags)
        self.blur_sizer2.Add(self.blurthresh_up_size, 0, border=border, flag=self.flags)
        self.blur_sizer2.Add(self.blurthresh_down_size, 0, border=border, flag=self.flags)

        
        self.blur_sizer.Add(self.blur_sizer1, 0, border=border)
        self.blur_sizer.Add(self.blur_sizer2, 0, border=border)

        self.course_map_button = wx.Button(self, -1, "Go!", size=(35,22))

        self.blur_sizer.Add(self.course_map_button, 0, border=border)
        self.Bind(wx.EVT_BUTTON, self.on_course_map_go, self.course_map_button)
        # self.Bind(wx.EVT_BUTTON, self.on_down_size, self.blur_down_size)

        
        self.blur_sizer.AddSpacer(10)
    
    def on_course_map_go(self, event):
        self.blurthresh_size[self.slice] = float(self.blurthresh_textbox.GetValue())
        self.blur_size[self.slice] = float(self.blur_textbox.GetValue())
        self.coarse_map(self.threshed, self.blur_size[self.slice], self.blurthresh_size[self.slice])

        # self.
        self.draw_2d()


    def coarse_map(self, thres, std, threshold, plot=False):

        thres_float = thres.astype(numpy.float32)
        blur = ndi.gaussian_filter(thres_float,std)
        blur = blur/numpy.amax(blur)
        self.mask[self.slice,:,:] = (blur > threshold)

        
        
        
 
    def on_up_size(self, event):
        self.object_size[self.slice] += 1
        self.object_textbox.SetValue(str(self.object_size[self.slice]))
        self.draw_2d()

    def on_down_size(self, event):
        if self.object_size[self.slice] > 0.:
            self.object_size[self.slice] -= 1
            self.object_textbox.SetValue(str(self.object_size[self.slice]))
            self.draw_2d()


    def size_filter(self, image):
        return morphology.remove_small_objects(image, min_size = self.object_size[self.slice])

    def on_up_thresh(self, event):
        
        self.thresh[self.slice] = self.thresh[self.slice] + 0.1
        self.textbox0.SetValue(str(self.thresh[self.slice]))
        self.draw_2d()

    def on_down_thresh(self, event):
        # if self.thresh > 0.1
        self.thresh[self.slice] = self.thresh[self.slice] - 0.1
        print(self.thresh[self.slice])
        self.textbox0.SetValue(str(self.thresh[self.slice]))
        self.draw_2d()


    def on_up_slice(self, event):
        if self.slice < self.data.shape[0]-1:
            self.slice += 1
            self.slice_box.SetValue(str(self.slice))
            self.textbox0.SetValue(str(self.thresh[self.slice]))
            self.object_textbox.SetValue(str(self.object_size[self.slice]))
            self.blur_textbox.SetValue(str(self.blur_size[self.slice]))
            self.blurthresh_textbox.SetValue(str(self.blurthresh_size[self.slice]))

            self.draw_2d()

    def on_down_slice(self, event):
        if self.slice > 0.:
            self.slice -= 1
            self.slice_box.SetValue(str(self.slice))
            self.textbox0.SetValue(str(self.thresh[self.slice]))
            self.object_textbox.SetValue(str(self.object_size[self.slice]))
            self.blur_textbox.SetValue(str(self.blur_size[self.slice]))
            self.blurthresh_textbox.SetValue(str(self.blurthresh_size[self.slice]))

            self.draw_2d()

    

    

    def normalise_data(self, data):
        print("Normalising data...")
        data1d = data.flatten() #turning mrcfile into 1d array
        #calculating statistics
        mean = numpy.mean(data1d)
        stdev = numpy.std(data1d)
        data = (data - mean)/stdev
        
        return data

    
    def draw_2d(self):
        # levels=self.GetLevels()

        self.first_open = True

        

        # self.axes.clear()
        # self.axes1D.clear()

        # colormap=cm.Blues
        # colormap2=cm.Reds
        # # colormap=cm.seismic

        # # x = self.twod_data.shape[1] - 10

        x = range(self.data.shape[2])
        y = range(self.data.shape[1])
        
        
        
        z = self.data[self.slice,:,:]

        self.threshed = z>self.thresh[self.slice]
        if self.object_size[self.slice] > 0:
            self.threshed = self.size_filter(self.threshed)

        # if self.course_mapped == False:

        self.z_im = self.data[self.slice,:,:]
        self.objects[self.slice]=self.threshed*self.mask[self.slice,:,:]
        self.threshedraw[self.slice]=self.threshed
        #     self.main_image = self.axes_im.imshow(self.z_im, cmap='gray') #
        # else:
        #     # self.axes_im.imshow(self.z_im, cmap='gray') #
        #     self.main_image = self.axes_im.imshow(self.course_mapped_z_im, cmap='gray', alpha=0.5)

        # self.course_mapped = True

        self.z_im = cv2.normalize(self.z_im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.course_mapped_z_im = color.label2rgb(self.mask[self.slice,:,:], self.z_im, bg_label=0)

        self.zslice = self.axes_zslice.imshow(z, cmap='gray')
        self.main_image = self.axes_im.imshow(self.course_mapped_z_im, cmap='gray')
        
        self.axes_original_threshed.imshow(self.threshed, cmap='gray') #
        self.threshed_image = self.axes.imshow(self.threshed*self.mask[self.slice,:,:], cmap='gray') #
        
        
        self.canvas.draw()
        self.axes_zslice.invert_yaxis()
        self.axes.invert_yaxis()
        self.axes_im.invert_yaxis()
        self.axes_original_threshed.invert_yaxis()
        self.save_mask()
        self.save_params()
        self.background_save(None)


        # self.background = self.canvas.copy_from_bbox(self.axes.bbox)

    def save_mask(self):
        if(os.path.exists('./data/masks/'+self.session)==False):
            os.makedirs('./data/masks/'+self.session)
        
        
        numpy.save(f'data/masks/{self.session}/{self.tomo}_mask.npy', self.objects)
        numpy.save(f'data/masks/{self.session}/{self.tomo}_coarsemap.npy',self.mask)
        numpy.save(f'data/masks/{self.session}/{self.tomo}_threshraw.npy',self.threshedraw)

    def save_params(self):
        parameters = (self.mask, self.object_size, self.blur_size, self.blurthresh_size, self.thresh)

        if(os.path.exists('./save/'+self.session)==False):
            os.makedirs('./save/'+self.session)

        pickle.dump(parameters, open(f'save/{self.session}/{self.tomo}.params', 'wb'))

    def load_params(self):
        if os.path.exists(f'save/{self.session}/{self.tomo}.params'):
            self.mask, self.object_size, self.blur_size, self.blurthresh_size, self.thresh = pickle.load(open(f'save/{self.session}/{self.tomo}.params', 'rb'))

    def create_main_panel(self):
        """ Creates the main panel with all the controls on it:
             * mpl canvas
             * mpl navigation toolbar
             * Control panel for interaction
        """
        self.fig = Figure(constrained_layout=False)
        # self.fig.clear()
        # if(self.tabOne.dim==2):
        self.axes_zslice = self.fig.add_subplot(221)
        self.axes_im = self.fig.add_subplot(222)
        self.axes_original_threshed = self.fig.add_subplot(223)
        self.axes = self.fig.add_subplot(224)
        

        
        
        self.scatter_paint_data = numpy.array(([],[]))
        self.course_mapped = False
        # self.scatter_paint_data_y = []

        ## Initialise main matplotlib canvas
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.SetMinSize(wx.Size(1,1))
        self.painting = False
        # self.brush_size = 

        self.paint_point = self.axes.scatter([],[], marker = 's',color='r', s = 30.)
        
        self.scatter_paint = self.axes.scatter(self.scatter_paint_data[0,:],self.scatter_paint_data[1,:], marker = 's', s = 30., color='k')

        self.scatter_paint_im = self.axes_im.scatter(self.scatter_paint_data[0,:],self.scatter_paint_data[1,:], marker = 's', s = 30., color='r', alpha=0.2)
        

        self.canvas.mpl_connect('button_press_event', self.onPress)
        self.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.canvas.mpl_connect('button_release_event', self.onRelease)

        ## Initialise navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        ## Adding our control boxes
        self.flags = wx.ALIGN_LEFT | wx.BOTTOM | wx.TOP | wx.LEFT | wx.ALIGN_CENTER_VERTICAL
        self.drawing_box()
        self.contour_box()
        self.object_filter_box()
        self.blur_filter_box()
        # self.shiftX_box()

        ## Piece together the control boxes
        self.hbox = wx.BoxSizer(wx.VERTICAL)
        self.hbox.AddSpacer(5)
        self.hbox.Add(self.sliceSizer)
        self.hbox.AddSpacer(5)
        self.hbox.Add(self.cntrSizer)
        self.hbox.AddSpacer(5)
        self.hbox.Add(self.object_sizer)
        self.hbox.AddSpacer(5)
        self.hbox.Add(self.blur_sizer)
        # self.hbox.Add(self.shiftxSizer)

        ## Main vertical sizer
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.EXPAND) # Main matplotlib canvas
        self.vbox.Add(self.toolbar, 0, wx.EXPAND) # Add navigation toolbar
        self.vbox.AddSpacer(10)

        self.vbox_big = wx.BoxSizer(wx.HORIZONTAL)
        self.vbox_big.Add(self.vbox, 1, wx.EXPAND) # Add control boxes
        self.vbox_big.Add(self.hbox, 0, wx.EXPAND) # Add control boxes
        self.vbox_big.AddSpacer(10)



        ## Make vbox the main sizer
        self.SetSizer(self.vbox_big)
        self.vbox_big.Fit(self)
        print('done create plane')
        


    def onPress(self, event):
        if event.inaxes == self.axes_im:
            #print(event.xdata, event.ydata)
            self.painting = True
            self.save_params()


    def onMove(self, event):
        if self.painting == True and event.inaxes == self.axes_im:
            #print("painting", event.xdata, event.ydata)
            xdat = int(numpy.round(event.xdata))
            ydat = int(numpy.round(event.ydata))
            
            # self.scatter_paint_data_x.append(xdat)
            # self.scatter_paint_data_y.append(ydat)
            # print(self.scatter_paint_data.shape)
            # if (self.scatter_paint_data.shape[1] == 0):
            #     self.scatter_paint_data = numpy.expand_dims(numpy.array((xdat, ydat)), 0)
            # else:
            #     #  if (xdat, ydat) not in self.scatter_paint_data:
            #         # print(self.scatter_paint_data)
            #         self.scatter_paint_data = numpy.unique(numpy.vstack((self.scatter_paint_data, (xdat, ydat))), axis = 1)
            # test = numpy.array((self.scatter_paint_data_x, self.scatter_paint_data_y))
            # self.scatter_paint.set_offsets(self.scatter_paint_data)
            # self.scatter_paint_im.set_offsets(self.scatter_paint_data)
            # self.paint_point.set_offsets((event.xdata, event.ydata))
            # self.axes.draw_artist(self.scatter_paint)
            # self.axes_im.draw_artist(self.scatter_paint_im)
            # self.axes.draw_artist(self.paint_point)
            self.painting_radius = 20
            y_min = max(0,ydat-20)
            y_max = min(self.mask.shape[1],ydat+20)
            x_min = max(0,xdat-20)
            x_max = min(self.mask.shape[2],xdat+20)
            self.mask[self.slice,y_min:y_max,x_min:x_max] = 0
            self.course_mapped_z_im = color.label2rgb(self.mask[self.slice,:,:], self.z_im, bg_label=0)
            self.main_image.set_data(self.course_mapped_z_im)
            self.axes_im.draw_artist(self.main_image)
            self.threshed_image.set_data(self.threshed*self.mask[self.slice,:,:])
            
            self.axes_im.draw_artist(self.main_image)
            self.axes.draw_artist(self.threshed_image)
            self.canvas.blit(self.fig.bbox)
            self.canvas.flush_events()
            


            # self.axes.scatter(event.xdata, event.ydata, color='r', alpha=0.2)
            # self.canvas.draw()
            # self.draw_2d()

    def onRelease(self, event):
        self.painting = False
        # self.paint_point.set_offsets(([], []))
        # self.axes.draw_artist(self.paint_point)
        # self.canvas.blit(self.fig.bbox)
        # self.canvas.flush_events()


    def create_status_bar(self):
        self.statusbar = self.CreateStatusBar()

    def GetData(self,infile):
        # print infile
        input=self.readfile(infile)
        # print input
        xs=[]
        ys=[]
        zs=[]
        Xs=[]
        Ys=[]
        Zs=[]
        for i in range(len(input)):
            if(len(input[i])!=0):
                xs.append(float(input[i][0]))
                ys.append(float(input[i][1]))
                zs.append(float(input[i][2]))
            else:
                Xs.append(xs)
                Ys.append(ys)
                Zs.append(zs)
                zs=[]
                ys=[]
                xs=[]
        if(len(xs)!=0):
            Xs.append(xs)
            Ys.append(ys)
            Zs.append(zs)
        return numpy.array(Xs),numpy.array(Ys),numpy.array(Zs)

    def load_decon_data(self):
        if(os.path.exists('out/yz.decon')):
            self.yz_decon_Xs,self.yz_decon_Ys,self.yz_decon_Zs=self.GetData('out/yz.decon')
            # self.yz_decon_Zs*=numpy.max(self.ZZ1)/numpy.max(self.yz_decon_Zs)
        if(os.path.exists('out/xz.decon')):
            self.xz_decon_Xs,self.xz_decon_Ys,self.xz_decon_Zs=self.GetData('out/xz.decon')
            # self.xz_decon_Zs*=numpy.max(self.ZZ1)/numpy.max(self.xz_decon_Zs)
        if(os.path.exists('out/xy.decon')):
            self.xy_decon_Xs,self.xy_decon_Ys,self.xy_decon_Zs=self.GetData('out/xy.decon')
            # self.xy_decon_Zs*=numpy.max(self.ZZ1)/numpy.max(self.xy_decon_Zs)

    def background_save(self, event):
        # if not self.first_open:
        #     try:
        #         self.line1.remove()
        #     except:
        #         print('cant remove!')
        #     try:
        #         self.line_decon.remove()
        #     except:
        #         print('cant remove decon!')


        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        # # print('back update', self.background)
        # self.line1, = self.axes1D.plot(self.uc0.ppms_scale, self.twod_data[:,0], color='k')
        # if(self.cb_calc.IsChecked() ):
        #     self.line_decon, = self.axes1D.plot(self.uc0.ppms_scale, self.twod_data[:,0], color = 'C0')
        # self.axes1D.set_ylim(numpy.min(self.twod_data),numpy.max(self.twod_data))
        # self.axes1D.set_zorder(-100)
        # self.axes.patch.set_visible(False)
        self.first_open = False


    def on_xlims_change(self, event_ax):
        if self.first_open == False:

            self.background_save(event_ax)

    def on_ylims_change(self, event_ax):
        if self.first_open == False:
            self.background_save(event_ax)


    def draw_figure(self,scale='y'):
        """ Redraws the figure
        """
        # levels=self.GetLevels()

       

        self.draw_2d()
            # self.peak_fitted_input = False


        

        self.canvas.draw()

    





    

    def on_cb_grid_auto(self, event):
        self.draw_figure()

    def on_slider_width(self, event):
        self.draw_figure()

    def on_draw_button(self, event):
        self.draw_figure()


    def on_shiftx_button(self, event):
        self.shiftXrun=1

        chain=self.textbox_chain.GetValue()
        shiftx2.runShiftx2(self.pdbfile,self.methlist,chain)
        self.draw_figure()


    def on_AutoFit_button(self, event):
        self.thresh=float(self.textbox0.GetValue())
        self.plotty=analslices1d(self.tabOne.peak,self.thresh)
        self.cb_grid_auto.SetValue(1)
        self.draw_figure()


    def on_N_button(self, event):
        self.ComboBox1.SetSelection(self.ComboBox1.GetSelection()+1)
        self.draw_figure()

    def on_P_button(self, event):
        self.ComboBox1.SetSelection(self.ComboBox1.GetSelection()-1)
        self.draw_figure()



    def on_pick(self, event):
        if self.drag_extra_plot == True:
            self.dragging_extra_plot = True
            self.drag_extra_plot = False
            self.origin = (event.xdata, event.ydata)


    def on_release(self, event):
        if self.dragging_extra_plot == True:
            self.dragging_extra_plot = False
            self.drag_extra_plot = False
            # self.background_save(None)



            
    def on_move(self, event):
        if self.dragging_extra_plot == True:
            # self.dragging_extra_plot = True
            for line in self.dt.extra_plots[-1].collections:
                line.remove()
            del(self.dt.extra_plots[-1])
            dx = self.origin[0]-event.xdata
            dy = self.origin[1]-event.ydata
            self.dt.extra_plots.append(self.axes.contour(self.dt.ucs[-1][0].ppm_scale()-dx,self.dt.ucs[-1][1].ppm_scale()-dy, self.dt.data[-1].T, cmap=self.dt.color_list[len(self.dt.extra_plots)], levels=self.dt.levels, linewidths=0.5))
            self.canvas.draw()
            return


            
            
        self.pressed = True
        self.origin = event.xdata
        # print(self.tabOne.index0)
        if self.first_open == True:
            self.background_save(event)
        if event.xdata != None and event.ydata != None and event.inaxes == self.axes:
            x = numpy.argwhere((self.uc1.ppms_scale - event.ydata) < (self.uc1.ppms_scale[1]-self.uc1.ppms_scale[0])/10.0)[0]
            #print event.ydata, self.uc1.ppms_scale]
            self.canvas.restore_region(self.background)
            self.line1.set_ydata(self.twod_data[:,int(x)])
            self.axes.draw_artist(self.line1)
            if(self.cb_calc.IsChecked()):
                self.line_decon.set_ydata(self.twod_data_decon[:,int(x)])
                self.axes.draw_artist(self.line_decon)
            self.axes1D.set_ylim(numpy.min(self.twod_data),numpy.max(self.twod_data))

            self.canvas.blit(self.axes.bbox)                                
            # self.fig.canvas.draw()
        #  def on_pick(self,event):
        # print(event.mouseevent.inaxes)
        
    def on_pick_peak(self, event):
        if event.mouseevent.inaxes==self.axes:
            ind = event.ind
            print('picked:', self.peaks_text[ind[0]].get_text())
            self.ftol = 1e-10
            self.selected = self.peaks_text[ind[0]].get_text()
            self.peaks_text[ind[0]].set_color('r')
            update_colors = []
            for x in range(len(self.peaks_text)):
                if x == ind[0]:
                    self.peaks_text[x].set_color('r')
                    update_colors.append('r')

                else:
                    self.peaks_text[x].set_color('k')
                    update_colors.append('k')

            event.artist.set_color(update_colors)
            self.fuda_number = 0

            for key in self.line_fitting.plotting_resim_data.keys():
                if self.selected == str(key):
                    self.line_fitting.plot_fuda_fit(str(self.selected), 0, self.axes_proj, self.canvas)
            self.axes.draw_artist(event.artist)
            self.axes.draw_artist(self.peaks_text[ind[0]])
            self.draw_bar()
            self.canvas.blit(self.axes.bbox)
            self.background_save(None)




    def on_text_enter(self, event):
        self.draw_figure()

    def on_save_plot(self, event):
        file_choices = "PNG (*.png)|*.png"
        dlg = wx.FileDialog(
            self,
            message="Save plot as...",
            defaultDir=os.getcwd(),
            defaultFile="plot.png",
            wildcard=file_choices,
            style=wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Saved to %s" % path)

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """ A demo using wxPython with matplotlib:

         * Use the matplotlib navigation bar
         * Add values to the text box and press Enter (or click "Draw!")
         * Show or hide the grid
         * Drag the slider to modify the width of the bars
         * Save the plot to a file using the File menu
         * Click on a bar to receive an informative message
        """
        dlg = wx.MessageDialog(self, msg, "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def flash_status_message(self, msg, flash_len_ms=1500):
        self.statusbar.SetStatusText(msg)
        self.timeroff = wx.Timer(self)
        self.Bind(
            wx.EVT_TIMER,
            self.on_flash_status_off,
            self.timeroff)
        self.timeroff.Start(flash_len_ms, oneShot=True)

    def on_flash_status_off(self, event):
        self.statusbar.SetStatusText('')

    def onFocus(self, event):
        print("Projection has focus!")

    #FGA added
    def onGetFile(self, e, textBox):
        #get dialog box here
        cwd = os.getcwd()
        dlg = wx.FileDialog(self, message="Choose a file", defaultDir=os.getcwd(), defaultFile="",
            wildcard="PDB file (*.pdb)|*.pdb|" , style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            #print path
            #fu=self.dirBox.GetValue()
            #print fu
            #print path.split(fu)
            #splitPath = path.split(cwd)
            #textBox.SetValue('.' + splitPath[1])
            print("You chose the following file(s):")
            print(path)
            textBox.SetValue(path)
        dlg.Destroy()
