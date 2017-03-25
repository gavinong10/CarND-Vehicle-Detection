from sklearn.model_selection import ShuffleSplit
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import cv2
import itertools
import pandas as pd
import numpy as np
import copy
import glob
import pickle
import skvideo.io

import slide
import postprocess
import data
import model
import preprocess

def prepare_df():
    # Read in all the car and non-car classes
    vehicle_data = data.retrieve_data("vehicles")
    non_vehicle_data = data.retrieve_data("non-vehicles")

    # TODO: Ensure there is no class imbalance; fix as necessary
    df = pd.concat([vehicle_data, non_vehicle_data])
    
    # Since n_splits is 1, generator should only produce one object
    train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df))
    df['dataset'] = 'test'
    df['dataset'].iloc[train_idx.tolist()] = 'train'
    
    return df
    
def get_generator_for_frames(filepath='project_video.mp4', batch_size=100):
    cap = cv2.VideoCapture(filepath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return extract_frames_from_video(cap, batch_size), n_frames
    
def extract_frames_from_video(cap, batch_size):
    # Load in images from video
    frames = []
    count = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frames.append(frame)
        
        count += 1
        if count == batch_size:
            yield np.array(frames)
            count = 0
            frames = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    if len(frames) > 0:
        yield np.array(frames)
        
def produce_heatmaps(frames, svc_model, X_scaler, le, orient, pix_per_cell, cell_per_block, colorspace, spatial_size, hist_bins):
    heatmaps = model.get_sliding_window_preds(frames, svc_model, X_scaler, le, orient, pix_per_cell, cell_per_block, 
                               y_start=400, y_stop=656, cell_stride=2,
                               scale=1, colorspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)

    heatmaps += model.get_sliding_window_preds(frames, svc_model, X_scaler, le, orient, pix_per_cell, cell_per_block, 
                                   y_start=400, y_stop=656, cell_stride=2,
                                   scale=1.5, colorspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)

    heatmaps += model.get_sliding_window_preds(frames, svc_model, X_scaler, le, orient, pix_per_cell, cell_per_block, 
                                   y_start=400, y_stop=500, cell_stride=2,
                                   scale=0.75, colorspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)
    
    return heatmaps
    
def main():
    ###### PARAMETERS ######
    COLORSPACE = 'HLS'
    ORIENT = 12
    PIX_PER_CELL = 8 # number of pixels to calculate the gradient
    CELL_PER_BLOCK = 2 # the local area over which the histogram counts in a given cell will be normalized
    # Color Bin
    COLOR_BIN_SHAPE = (16, 16)
    # Color Hist
    NUM_HIST_BINS = 32
    ########################
    
    # df = prepare_df()
    # svc_model, le, X_scaler = model.train_model(df, COLORSPACE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, COLOR_BIN_SHAPE, NUM_HIST_BINS)
    # pickle.dump(svc_model, open('svc_model.p', 'wb'))
    # pickle.dump(le, open('le.p', 'wb'))
    # pickle.dump(X_scaler, open('X_scaler.p', 'wb'))
    
    svc_model = pickle.load(open('svc_model.p', 'rb'))
    le = pickle.load(open('le.p', 'rb'))
    X_scaler = pickle.load(open('X_scaler.p', 'rb'))
    
    # Load in images from video
    batch_size=50
    frames_to_process = 50
    
    frames_generator, n_frames = get_generator_for_frames(batch_size=batch_size)
    n_frames = min(frames_to_process - frames_to_process % batch_size, n_frames)
    
    output_name = "output.mp4"
    smoother = None
    
    writer = None
    
    frames_processed = 0
    for frames in frames_generator:
        heatmaps = produce_heatmaps(frames, svc_model, X_scaler, le, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, 
                               COLORSPACE, COLOR_BIN_SHAPE, NUM_HIST_BINS)
        
        if smoother is None:
            smoother = model.RingBufSmoother(heatmaps[0].shape, threshold=7)

        # Extend and apply rolling threshold through the heatmaps
        thresholded_heatmaps = []
        for heatmap in heatmaps:
            smoother.extend(heatmap)
            thresholded_heatmaps.append(smoother.rolling_threshold())
            
        car_segmentation, num_cars = postprocess.segment_cars(thresholded_heatmaps)
        imgs_superimposed = postprocess.draw_boxes(frames, car_segmentation, num_cars)
        
        # Write images to video
        
        if writer is None:
            #print(tuple([n_frames] + list(imgs_superimposed.shape[1:])))
            writer = skvideo.io.FFmpegWriter(output_name)#, tuple([n_frames] + list(imgs_superimposed.shape[1:])) )
            
        vid_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_superimposed]
        skvideo.io.vwrite(output_name, vid_frames)
        
        for i in range(len(vid_frames)):
            writer.writeFrame(vid_frames[i])
        writer.close()
        frames_processed += len(frames)
        if frames_processed >= frames_to_process:
            break

    # Release everything if job is finished
    # cap.release()
    writer.close()

if __name__ == "__main__":
    main()