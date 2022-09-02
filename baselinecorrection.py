import mrcfile
import copy
from lib.analysis import *
import numpy as np
from skimage import color
import cv2

def make_square(rawtomo):
    mid_zslice = int(rawtomo.shape[0]/2)

    print('input dimensions for square mask (x1,x2,y1,y2)')
    dim = input()
    dim = dim.split(',')
    dim = list(map(int,dim))

    x1 = dim[0]
    x2 = dim[1]
    y1 = dim[2]
    y2 = dim[3]

    square = np.zeros_like(rawtomo)
    square[:,y1:y2,x1:x2] = 1

    norm_proj = cv2.normalize(rawtomo[mid_zslice,:,:], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    plt.figure()
    plt.imshow(color.label2rgb(square[mid_zslice,:,:], norm_proj,bg_label=0))
    plt.gca().invert_yaxis()
    plt.show()

    return dim

def baseline_correction(normtomo):
    mid_zslice = int(normtomo.shape[0]/2)

    plt.figure()
    plt.imshow(normtomo[mid_zslice,:,:],cmap='gray')
    plt.gca().invert_yaxis()
    plt.show()

    dim = make_square(normtomo)

    print('accept? (Y/N)')
    accept = input()
    if(accept=='N'):
        dim = make_square(normtomo)

    x1 = dim[0]
    x2 = dim[1]
    y1 = dim[2]
    y2 = dim[3]

    baseline = np.mean(normtomo[:,y1:y2,x1:x2])

    return baseline, dim

#Baseline correction
rawtomoname = 'raw/20220307_Baker/tomo8_b8_bp10000-500_s81-101.mrc' #can become sys.arg argument

rawtomoname = rawtomoname.split('.')[0]
session = rawtomoname.split('/')[-2]
tomo = rawtomoname.split('/')[-1]

coarse_mapname = f'data/masks/{session}/{tomo}_coarsemap.npy'
maskname = f'data/masks/{session}/{tomo}_mask.npy'
threshedname = f'data/masks/{session}/{tomo}_threshraw.npy'

#load files
with mrcfile.open(f'{rawtomoname}.mrc') as mrc:
    rawtomo = copy.deepcopy(mrc.data) * -1
    voxel_size = mrc.voxel_size
    print(voxel_size)
print(rawtomo.shape)
normtomo=normalise_data(rawtomo)

coarsemap = np.load(coarse_mapname)
mask = np.load(maskname).astype(bool)
threshed = np.load(threshedname).astype(bool)

baseline, _ = baseline_correction(normtomo)
print(baseline)