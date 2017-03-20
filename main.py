from sklearn.model_selection import ShuffleSplit
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import cv2
import itertools
import pandas as pd
import numpy as np
import copy
import glob

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

def main():
    df = prepare_df()
    
    colorspace = 'YUV'
    orient = 10
    pix_per_cell = 8 # number of pixels to calculate the gradient
    cell_per_block = 2 # the local area over which the histogram counts in a given cell will be normalized

    svc_model, le, X_scaler = model.train_model(df, colorspace, orient, pix_per_cell, cell_per_block)
    
    


if __name__ == "__main__":
    main()