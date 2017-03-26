from sklearn.model_selection import ShuffleSplit
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.externals import joblib
import cv2
import itertools
import pandas as pd
import numpy as np
import copy
import glob
import pickle
import skvideo.io

import postprocess
import data
import model
import constants as c

def prepare_df():
    # Read in all the car and non-car classes
    vehicle_data = data.retrieve_data("vehicles")
    non_vehicle_data = data.retrieve_data("non-vehicles")

    df = pd.concat([vehicle_data, non_vehicle_data])
    
    # Since n_splits is 1, generator should only produce one object
    train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df))
    df['dataset'] = 'test'
    df['dataset'].iloc[train_idx.tolist()] = 'train'

    # Shuffle the dataframe - in case there are iteration based operations done on it
    df = df.sample(frac=1)
    
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

def produce_heatmaps(frames, svc_model, X_scaler, le):
    heatmaps = model.get_sliding_window_preds(frames, svc_model, X_scaler, le, c.ORIENT, c.PIX_PER_CELL, c.CELL_PER_BLOCK, 
                               y_start=400, y_stop=656, cell_stride=2,
                               scale=1, colorspace=c.COLORSPACE, spatial_size=c.COLOR_BIN_SHAPE, hist_bins=c.NUM_HIST_BINS)

    heatmaps += model.get_sliding_window_preds(frames, svc_model, X_scaler, le, c.ORIENT, c.PIX_PER_CELL, c.CELL_PER_BLOCK, 
                                   y_start=400, y_stop=656, cell_stride=2,
                                   scale=1.5, colorspace=c.COLORSPACE, spatial_size=c.COLOR_BIN_SHAPE, hist_bins=c.NUM_HIST_BINS)

    heatmaps += model.get_sliding_window_preds(frames, svc_model, X_scaler, le, c.ORIENT, c.PIX_PER_CELL, c.CELL_PER_BLOCK, 
                                   y_start=400, y_stop=500, cell_stride=2,
                                   scale=0.75, colorspace=c.COLORSPACE, spatial_size=c.COLOR_BIN_SHAPE, hist_bins=c.NUM_HIST_BINS)
    
    return heatmaps

def gen_images_from_video():
    """
    A function to generate heatmap images from the video stream
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if c.RETRAIN:
        df = prepare_df()
        svc_model, le, X_scaler = model.train_model(df)
        pickle.dump(svc_model, open('models/svc_model%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
        pickle.dump(le, open('models/le%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
        pickle.dump(X_scaler, open('models/X_scaler%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
    
    svc_model = pickle.load(open('models/svc_model%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    le = pickle.load(open('models/le%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    X_scaler = pickle.load(open('models/X_scaler%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    
    # Load in images from video
    frames_generator, n_frames = get_generator_for_frames(batch_size=c.BATCH_SIZE)
    
    frames_to_process = c.FRAMES_TO_PROCESS or n_frames

    smoother = None    
    frames_processed = 0
    start_offset = 0

    plt.figure(1)
    fig = plt.figure(figsize=(20, 5 * c.FRAMES_TO_PROCESS))
    plt.figure(2)
    fig = plt.figure(figsize=(20, 5 * c.FRAMES_TO_PROCESS))
    gs = gridspec.GridSpec(c.FRAMES_TO_PROCESS, 2)

    for frames in frames_generator:
        if start_offset < c.START_FRAME:
            start_offset += len(frames)
            continue
            
        heatmaps = produce_heatmaps(frames, svc_model, X_scaler, le)
        
        if smoother is None:
            smoother = postprocess.RingBufSmoother(heatmaps[0].shape, length=c.BUFFER_LEN, threshold=c.MIN_HEAT_THRES)

        # Extend and apply rolling threshold through the heatmaps
        thresholded_heatmaps = []
        for heatmap in heatmaps:
            smoother.extend(heatmap)
            thresholded_heatmaps.append(smoother.rolling_threshold())
            
        car_segmentation, num_cars = postprocess.segment_cars(thresholded_heatmaps)
        imgs_superimposed = postprocess.draw_boxes(frames, car_segmentation, num_cars)

        raw_bboxes = postprocess.draw_boxes(np.zeros_like(frames), car_segmentation, num_cars)

        for i in range(0, c.BATCH_SIZE):
            plt.figure(1)
            orig_frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            plt.subplot(gs[frames_processed + i, 0]).imshow(orig_frame_rgb)
            plt.subplot(gs[frames_processed + i, 1]).imshow(heatmaps[i], cmap='hot')

            plt.figure(2)
            plt.subplot(gs[frames_processed + i, 0]).imshow(orig_frame_rgb)
            plt.subplot(gs[frames_processed + i, 1]).imshow(raw_bboxes[i])

        
        frames_processed += len(frames)
        if frames_processed >= frames_to_process:
            break
    
    plt.savefig('output_images/heatmaps.png')
    plt.savefig('output_images/labelled_bboxes.png')
    # Release everything if job is finished
    # cap.release()
    
def main(): 
    if c.RETRAIN:
        df = prepare_df()
        svc_model, le, X_scaler = model.train_model(df)
        pickle.dump(svc_model, open('models/svc_model%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
        pickle.dump(le, open('models/le%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
        pickle.dump(X_scaler, open('models/X_scaler%s.p' % (c.SAVE_LOAD_APPENDIX), 'wb'))
    
    svc_model = pickle.load(open('models/svc_model%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    le = pickle.load(open('models/le%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    X_scaler = pickle.load(open('models/X_scaler%s.p' % (c.SAVE_LOAD_APPENDIX), 'rb'))
    
    # Load in images from video
    frames_generator, n_frames = get_generator_for_frames(batch_size=c.BATCH_SIZE)
    
    frames_to_process = c.FRAMES_TO_PROCESS or n_frames

    smoother = None    
    writer = None
    frames_processed = 0
    start_offset = 0
    for frames in frames_generator:
        if start_offset < c.START_FRAME:
            start_offset += len(frames)
            continue
            
        heatmaps = produce_heatmaps(frames, svc_model, X_scaler, le)
        
        if smoother is None:
            smoother = postprocess.RingBufSmoother(heatmaps[0].shape, length=c.BUFFER_LEN, threshold=c.MIN_HEAT_THRES)

        # Extend and apply rolling threshold through the heatmaps
        thresholded_heatmaps = []
        for heatmap in heatmaps:
            smoother.extend(heatmap)
            thresholded_heatmaps.append(smoother.rolling_threshold())
            
        car_segmentation, num_cars = postprocess.segment_cars(thresholded_heatmaps)
        imgs_superimposed = postprocess.draw_boxes(frames, car_segmentation, num_cars)
        
        # Write images to video
        
        if writer is None:
            writer = skvideo.io.FFmpegWriter(c.OUTPUT_NAME)
            
        vid_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_superimposed]
        
        for i in range(len(vid_frames)):
            writer.writeFrame(vid_frames[i])
        frames_processed += len(frames)
        if frames_processed >= frames_to_process:
            break

    # Release everything if job is finished
    # cap.release()
    writer.close()

if __name__ == "__main__":
    main()
    #gen_images_from_video()