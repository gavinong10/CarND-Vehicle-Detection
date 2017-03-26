import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from skimage.feature import hog
import constants as c

def return_model():
    svc = LinearSVC()
    
    return svc

def train_model(df, validate=False):
    X_scaler = StandardScaler()
    le = LabelEncoder()

    if validate:
        # TODO: split into train/validation sets, predict on validation set and 
        # report the accuracy
        pass
    
    # Create a model using all the data (Balance the vehicles and non-vehicles classes through sampling for removing bias)
    if c.UNBIAS_DATA:
        len_imbalance = sum(df["category"] == "vehicles") - sum(df["category"] == "non-vehicles")
        train_df = pd.concat([df[df["category"] == "non-vehicles"], \
                                df[df["category"] == "non-vehicles"].sample(len_imbalance, replace=True), \
                                df[df["category"] == "vehicles"]])

    else:
        train_df = df.copy()

    # Read images as a batch for training on hog features, color bins and
    # histogram features

    X_train_batches = []
    for start in range(0, len(train_df), c.BATCH_SIZE):
        # Read images
        imgs = train_df['image'].iloc[start:start + c.BATCH_SIZE].apply(cv2.imread)

        # Convert color
        imgs = np.array([cv2.cvtColor(img, eval("cv2.COLOR_BGR2" + c.COLORSPACE)) for img in imgs])

        hog_features = get_hog_features(imgs, c.ORIENT, c.PIX_PER_CELL, c.CELL_PER_BLOCK, vis=False, feature_vec=True)
        hog_features = hog_features.reshape(hog_features.shape[0], -1)
        spatial_features = get_color_bin_features(imgs, c.COLOR_BIN_SHAPE)
        hist_features = get_color_hist_features(imgs, c.NUM_HIST_BINS)
        
        X_train_batches.append(np.concatenate((hog_features, spatial_features, hist_features), axis=1))

    X_train = np.concatenate(X_train_batches)
    X_train = X_scaler.fit_transform(X_train)
    y_train = le.fit_transform(train_df['category'])
    svc_model = return_model()
    svc_model.fit(X_train, y_train)
    
    return svc_model, le, X_scaler

# Define a function to compute binned color features  
def get_color_bin_features(imgs, size=(32, 32)):
    features = []
    for img in imgs:
        features.append(cv2.resize(img, size).ravel())
    return np.stack(features)

# Define a function to compute color histogram features  
def get_color_hist_features(imgs, nbins=32, bins_range=(0, 256)):
    hist_features = []
    for img in imgs:
        channels_hist = [np.histogram(img[:,:,i], bins=nbins, range=bins_range)[0] for i in range(img.shape[2])]
        # Concatenate the histograms into a single feature vector
        hist_features.append(np.hstack(channels_hist))
    # Return the individual histograms, bin_centers and feature vector
    return np.stack(hist_features)

# Define a function to return HOG features and visualization
def get_hog_features(imgs, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        pass # TODO
    else: 
        features = []
        for img in imgs:
            img_features = np.stack([
                hog(img[:,:,ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                        visualise=vis, feature_vector=feature_vec) \
                        for ch in range(img.shape[2])])
            features.append(img_features)
        features = np.stack(features)
        return features

def get_sliding_window_preds(imgs, model, scaler, le, orient, pix_per_cell, cell_per_block, y_start=400, y_stop=656, cell_stride=2, scale=1, colorspace="HLS", spatial_size=(16, 16), hist_bins=32):
    """
    Gets detection predictions from a trained model on a sliding window over the given images.
    :param imgs: The images for which to get predictions.
    :param model: The model to make predictions.
    :param scaler: The scaler used to normalize the data when training.
    :param y_start: The pixel value on the y axis at which to start searching for cars (Top of
                    search window).
    :param y_stop: The pixel value on the y axis at which to stop searching for cars (Bottom of
                   search window).
    :param cell_stride: The stride of the sliding window, in HOG cells.
    :param scale: The scale of the sliding window relative to the training window size
                         (64x64).
    :param color_space: The color space to which to convert the images.
    :return: A heatmap of the predictions at each sliding window location for all images in imgs.
    """
    heatmaps = np.zeros(imgs.shape[:3])

    imgs_cvt = np.array([cv2.cvtColor(img, eval("cv2.COLOR_BGR2" + colorspace)) for img in imgs])
    imgs_cropped = imgs_cvt[:, y_start:y_stop, :, :]

    height, width = imgs_cropped.shape[1:3]

    # Scale the images based on the window scale. Because the model was trained on 64x64 patches
    # of HOG features, we still need that many features, so if we want a smaller window, we need
    # To size up the image so 64x64 is relatively smaller.
    if scale != 1:
        imgs_cropped = np.array([cv2.resize(img_cropped, (int(width / scale), int(height / scale))) for img_cropped in imgs_cropped])
        height, width = imgs_cropped.shape[1:3]

    num_blocks_x = (width // pix_per_cell) - 1
    num_blocks_y = (height // pix_per_cell) - 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 

    num_steps_x = (num_blocks_x - nblocks_per_window) // cell_stride
    num_steps_y = (num_blocks_y - nblocks_per_window) // cell_stride

    # Compute hog features over whole image for efficiency.
    # hog_features = get_HOG_features(imgs_cropped, feature_vec=False)
        
    hog_features = get_hog_features(imgs_cropped, c.ORIENT, c.PIX_PER_CELL, c.CELL_PER_BLOCK, vis=False, feature_vec=False)

    for x_step in range(num_steps_x):
        for y_step in range(num_steps_y):
            y_pos = y_step * cell_stride
            x_pos = x_step * cell_stride

            
            # Extract HOG for this patch
            patch_HOG_channels = [np.reshape(hog_features[:,
                                  ch,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window], (len(imgs), -1)) for ch in range(imgs_cropped.shape[3])]

            patch_HOG = np.concatenate(patch_HOG_channels, axis=1)

            xleft = x_pos * pix_per_cell
            ytop = y_pos * pix_per_cell

            # Extract the image patch
            patches = imgs_cropped[:,
                                 ytop:ytop + window,
                                 xleft:xleft + window]

            # Get color features
            patch_color_bins = get_color_bin_features(patches, c.COLOR_BIN_SHAPE)
            patch_color_hists = get_color_hist_features(patches, c.NUM_HIST_BINS)


            # Combine and normalize features
            patch_features = np.concatenate([patch_HOG, patch_color_bins, patch_color_hists],
                                            axis=1)
            patch_features_norm = scaler.transform(patch_features)

            # Make prediction
            patch_preds = model.predict(patch_features_norm)
            # Reshape so it can be broadcast with the 3D heatmaps array.
            patch_preds = np.reshape(patch_preds, [len(patch_preds), 1, 1])

            # Get the patch coordinates relative to the original image scale
            xleft_abs = np.int(xleft * scale)
            ytop_abs = np.int(ytop * scale) + y_start
            window_width_abs = np.int(window * scale)

            # Add prediction to the heatmap
            heatmaps[:, ytop_abs :ytop_abs + window_width_abs,
                        xleft_abs:xleft_abs + window_width_abs] += le.inverse_transform(patch_preds) == "vehicles"

    return heatmaps

