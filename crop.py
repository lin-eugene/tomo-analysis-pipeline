from lib.editmrc import *
filename = sys.argv[1]
slice_range = int(sys.argv[2])

crop(filename,slice_range)