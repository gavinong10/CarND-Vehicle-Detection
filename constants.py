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
MIN_HEAT_THRES = 14 #17 #14

########################

UNBIAS_DATA = False
RETRAIN = True
SAVE_LOAD_APPENDIX = "_biased"
OUTPUT_NAME = "biased_output.mp4"

BATCH_SIZE = 20
START_FRAME = 300
FRAMES_TO_PROCESS = 60 #None #60 #None #100 # None for the full video