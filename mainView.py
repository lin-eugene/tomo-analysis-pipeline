#!/usr/bin/python
####################################################################
# Front end for deconNMR spectral visualisation and fitting software
#
# A.Baldwin 10th Dec 2010
# A.Baldwin 12th June 2015  #sorted out for data
# A.Baldwin 20th Feb 2019   #got a proper distribution ready






import os,sys, platform
pathname, scriptname = os.path.split(sys.argv[0])   #get location where this script was executed
if(os.path.exists(os.pathsep+os.path.join(os.getcwd(),'bin') )): #does this path exist?
    os.environ["PATH"]+=os.pathsep+os.path.join(os.getcwd(),'bin')  #if running from an app, this will add bins to the system path
if(len(os.path.dirname(sys.executable).split('deconRun.app'))>1):
    from os.path import expanduser  #if running using the app, change working folder to user directory
    os.chdir(expanduser("~"))

#adding temp location.
#binaries will be copied here, so needs to be in system's path
#only for pyinstaller linux app.
if(platform.uname()[0]=='Linux'):
    try:
        print('MEIPASS:',sys._MEIPASS)
        os.environ["PATH"]+=os.pathsep+sys._MEIPASS
    except:
        pass
    try: #cleanup files in tmp
        files=os.listdir('/tmp')
        for file in files:
            if(len(file.split('MEI'))>1):
                test=os.path.join('/tmp',file)
                if test!=sys._MEIPASS:
                    print('Removing temp file:',test)
                    os.system('rm -rf '+test)
                else:
                    print('this is our guy',test)
    except:
        pass



#cleanup MEIPASS
#removing old temp directories
#import subprocess
#subprocess.call(['ls','-l'])

# Begin importing
import wx
import texttable

from deconFrame import deconFrame



########################################################################
class NotebookDemo(wx.Notebook):
    """
    Notebook class

    """
    def __init__(self, parent,panel,deconParFile):
        wx.Notebook.__init__(self, panel, id=wx.ID_ANY, style=
                             wx.BK_DEFAULT
                             #wx.BK_TOP
                             #wx.BK_BOTTOM
                             #wx.BK_LEFT
                             #wx.BK_RIGHT
                             )

        self.parent=parent
        self.deconParFile=deconParFile

        self.MAGMAONLY='n'

        # if(self.MAGMAONLY=='n'):
        self.tabOne = deconFrame(self,self.deconParFile)
        self.AddPage(self.tabOne, "Segmentation")

    

    



########################################################################
class MyApp(wx.Frame):
    """
    Frame that holds all other widgets
    """

    #----------------------------------------------------------------------
    def __init__(self,deconParFile):
        """Constructor"""
        self.monitorWidth, self.monitorHeight = wx.GetDisplaySize()
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "deconRun - "+os.getcwd().split('/')[-1], wx.DefaultPosition,
                          #size=(self.monitorWidth*0.95, self.monitorHeight*0.85),
                          size=(1600,950)
                          )
        panel = wx.Panel(self)

        self.create_menu()
        self.create_status_bar()

        self.SetBackgroundColour('WHITE')

        self.notebook = NotebookDemo(self,panel,deconParFile)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)

        panel.SetSizerAndFit(sizer)
        # self.Bind(wx.EVT_SIZE, self.OnSize)

        self.Layout()

        self.Show()

        # if(os.path.exists(deconParFile)==1):
        #     self.TestPath(deconParFile)
        #     self.DoLoad(deconParFile)
        #     #self.DoLoad(os.path.join(os.getcwd(), deconParFile))

        #self.Maximize(True)

    def OnSize(self, event):
        print('resized!')
    def create_status_bar(self):
        self.statusbar = self.CreateStatusBar()

    def create_menu(self):
        self.menubar = wx.MenuBar()
        menu_file = wx.Menu()

        m_new = menu_file.Append(-1, "&New\tCtrl-N", "New session")
        self.Bind(wx.EVT_MENU, self.OnNew, m_new)
        menu_file.AppendSeparator()

        m_load = menu_file.Append(-1, "&Open\tCtrl-L", "Open session file")
        self.Bind(wx.EVT_MENU, self.OnLoadResults, m_load)
        menu_file.AppendSeparator()

        m_save = menu_file.Append(-1, "&Save\tCtrl-S", "Save status")
        self.Bind(wx.EVT_MENU, self.OnSaveResults, m_save)
        menu_file.AppendSeparator()


        m_exit = menu_file.Append(-1, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.OnQuit, m_exit)
        menu_help = wx.Menu()
        m_about = menu_help.Append(-1, "&About\tF1", "About the demo")
        self.Bind(wx.EVT_MENU, self.on_about, m_about)
        self.menubar.Append(menu_file, "&File")
        self.menubar.Append(menu_help, "&Help")
        self.SetMenuBar(self.menubar)

    def OnQuit(self, e):
        self.Destroy()

    def OnNew(self,event):
        file_choices='*'
        dlg = wx.FileDialog(
            self,
            message="Save session...",
            defaultDir=os.getcwd(),
            #defaultFile=os.path.split(self.deconParFile)[1],
            defaultFile='deconParFile',
            wildcard=file_choices,
            style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.deconParFile=path

            outy=open(self.deconParFile,'w');outy.close()

            #self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Loaded %s" % path)
            os.chdir(os.path.dirname(path))
            print("CWD: ",os.getcwd())

            self.DoLoad(path)

    def OnSaveResults(self, event):
        file_choices='*'
        dlg = wx.FileDialog(
            self,
            message="Save session...",
            defaultDir=os.getcwd(),
            #defaultFile=os.path.split(self.deconParFile)[1],
            defaultFile='deconParFile',
            wildcard=file_choices,
            style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()

            self.deconParFile=path
            if(os.path.exists(self.deconParFile)==0):
                outy=open(self.deconParFile,'w');outy.close()
            # self.notebook.tabMagma.deconParFile=path
            # self.notebook.tabMagma.OnButtonSave(True)

            self.notebook.tabOne.deconParFile=path
            self.notebook.tabOne.OnButtonSave(True)

            #self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Saved %s" % path)


    #FGA added
    def OnLoadResults(self, event):
        file_choices='*'
        dlg = wx.FileDialog(
            self,
            message="Load session...",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=file_choices,
            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:


            path = dlg.GetPath()
            self.deconParFile=path
            #self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Loaded %s" % path)
            os.chdir(os.path.dirname(path))
            print("CWD: ",os.getcwd())

            self.DoLoad(path)
    def TestPath(self,deconParFile):
        Parse=self.notebook.tabOne.Parse #messy...

        self.deconParFile=deconParFile
        indir=Parse(self.deconParFile,'indir')
        fiddir=Parse(self.deconParFile,'fiddir')
        print('start:',indir,fiddir)
        if(indir!=0):
            if(os.path.exists(str(indir))==0):
                indir=self.CheckPath(str(indir))
            if fiddir!=0:
                if(os.path.exists(str(fiddir))==0):
                    fiddir=self.CheckFidPath(str(indir),str(fiddir))

        #self.WriteFID(indir,fiddir)
        print('finish:',indir,fiddir)

    def CheckPath(self,indir):
        print('cannot find ',indir,'. Trying to update:')
        tast=indir.split("/")
        loop=len(tast)-1
        ref=self.deconParFile

        for i in range(len(tast)): #looping backwards along the files
            ii=loop-i
            test=os.path.join(os.getcwd(),indir.split("/")[ii],ref)
            #print 'testing:',test
            if(os.path.exists(test)==1):
                print('Found new indir:',os.path.join(os.getcwd(),indir.split("/")[ii]))
                #sys.exit(100)
                #os.setcwd(indir)
                print(os.getcwd())
                return os.path.join(os.getcwd(),indir.split("/")[ii])

        print('Cannot find directory',indir)
        sys.exit(100)

    def CheckFidPath(self,indir,fiddir):
        print('cannot find fiddir: ',fiddir,'. Trying to update:')
        #test=fiddir.split(indir)
        tast=fiddir.split("/")
        print(indir)
        for i in range(len(tast)): #looping backwards along the files
            ii=len(tast)-1-i-1

            splitty=tast[ii] #point to split
            print(splitty)
            click=os.path.join(indir,splitty)
            #try:
            #    click=os.path.join(splitty,fiddir.split(splitty)[-1])
            #except:
            #    click=os.path.join(splitty,fiddir.split(splitty)[-2])
            print(click)
            test=os.path.join(indir,click)
            print('testing:',test)
            if(os.path.exists(test)==1):
                print('Found new fiddir:',test)
                #sys.exit(100)
                return test
        print('Cannot find directory',indir)
        sys.exit(100)


        print(test)
        print(indir,fiddir)
        fidnew=os.path.join(indir,test[-1])
        if(os.path.exists(fidnew)):
            print('Newfid found:',fidnew)
            return fidnew
        print('Cannot update fidfile.')
        return str(0)


    def WriteFID(self,indir,fiddir):
        try:
            decfile=os.path.join(indir,self.deconParFile)
        except:
            return
        if(os.path.exists(decfile)==0):
            return
        dec=[]
        inny=open(decfile)
        for line in inny.readlines():
            test=line.split()
            tick=0
            if(len(test)>0):
                if(test[0]=='fiddir'):
                    dec.append('fiddir = '+fiddir+'\n')
                    tick=1
                if(test[0]=='indir'):
                    dec.append('indir = '+fiddir+'\n')
                    tick=1
            if(tick==0):
                dec.append(line)
        inny.close()
        outy=open(decfile,'w')
        for de in dec:
            outy.write(de)
        outy.close()



   
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
        msg="UniDecNMR"
        dlg = wx.MessageDialog(self, msg, "UniDecNMR", wx.OK)
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


    def OnClose(self, event):
        self.Close(True)
        sys.exit(100)





#----------------------------------------------------------------------
if __name__ == "__main__":
    #print sys.argv
    if(len(sys.argv)==2):
        deconParFile=sys.argv[1]
    else:
        deconParFile='deconParFile'

    if(os.path.exists(deconParFile)==0):
        outy=open('deconParFile','w');outy.close()

    app = wx.App()
    frame = MyApp(deconParFile)
    app.MainLoop()
