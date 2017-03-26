###### PARAMETERS ######
COLORSPACE = 'HLS'
ORIENT = 12
PIX_PER_CELL = 8 # number of pixels to calculate the gradient
CELL_PER_BLOCK = 2 # the local area over which the histogram counts in a given cell will be normalized
# Color Bin
COLOR_BIN_SHAPE = (16, 16)
# Color Hist
NUM_HIST_BINS = 32

BUFFER_LEN = 10
MIN_HEAT_THRES = 14

########################

UNBIAS_DATA = True
RETRAIN = False
LOAD_DYELAX = False
SAVE_LOAD_APPENDIX = "_unbiased"

BATCH_SIZE = 20
START_FRAME = 0
FRAMES_TO_PROCESS = None #100 # None for the full video