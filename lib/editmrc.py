import mrcfile
import sys, os
import numpy as np

"""
crops tomograms into a defined slice range
How to use:
- cd to directory
- python3 crop.py <filename> <slice range>
    filename — <session>/<tomo>

NB — slice range goes ± slice rang
    e.g. if slice range = 10, crop.py crops ±10 slices from central slice
"""

def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = grid_spacing
        if origin is not None:
            mrc.header['origin']['x'] = origin[2]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[0]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("done")


def edit_pixelsize(filename):
    filename = filename.split('.')[0]
    session = filename.split('/')[-2]
    tomo = filename.split('/')[-1]

    if not os.path.isfile(f'raw/{session}/{tomo}.mrc'):
                print("file not found")
                sys.exit()

    with mrcfile.open(f'./raw/{session}/{tomo}.mrc') as mrc:
        data = mrc.data
        grid_spacing = mrc.voxel_size
        vox_size = grid_spacing.view((grid_spacing.dtype[0], len(grid_spacing.dtype.names)))
        vox_size = vox_size[0]
        grid_spacing = vox_size
    
    outfilename = f'./raw/{session}/{tomo}.mrc'
    save_density(data, grid_spacing=grid_spacing, outfilename=outfilename)

def crop(filename, slice_range):
    
    filename = filename.split('.')[0]
    session = filename.split('/')[-2]
    tomo = filename.split('/')[-1]


    if not os.path.isfile(f'raw/{session}/{tomo}.mrc'):
                print("file not found")
                sys.exit()

    with mrcfile.open(f'./raw/{session}/{tomo}.mrc') as mrc:
        data = mrc.data
        grid_spacing = mrc.voxel_size
        vox_size = grid_spacing.view((grid_spacing.dtype[0], len(grid_spacing.dtype.names)))
        vox_size = vox_size[0]
        grid_spacing = vox_size

    z_size = len(data[:,0,0])
    midslice = z_size/2
    midslice = int(midslice)

    newdata = data[range(midslice-slice_range,midslice+slice_range,1),:,:]

    print('newdata.shape=',newdata.shape)

    outfilename = f'./raw/{session}/{tomo}_s{midslice-slice_range}-{midslice+slice_range}.mrc'

    save_density(newdata, grid_spacing=grid_spacing, outfilename=outfilename)

#####

