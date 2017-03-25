import numpy as np
import cv2
from preprocess import bin_spatial, color_hist, get_hog_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import preprocess

def return_model():
    svc = LinearSVC()
    
    return svc

def train_model(df, colorspace, orient, pix_per_cell, cell_per_block, spatial_size=(16, 16), hist_bins=32):
    X_scaler = StandardScaler()
    le = LabelEncoder()
    
    # Create a model using all the data (Balance the vehicles and non-vehicles classes through sampling for removing bias)
    len_imbalance = sum(df["category"] == "vehicles") - sum(df["category"] == "non-vehicles")
    balanced_df = pd.concat([df[df["category"] == "non-vehicles"], \
                             df[df["category"] == "non-vehicles"].sample(len_imbalance, replace=True), \
                             df[df["category"] == "vehicles"]])

    hog_features = get_all_hog_features(balanced_df["image"], colorspace, orient, pix_per_cell, cell_per_block)
    spatial_features = get_all_color_bin_features(balanced_df["image"], spatial_size)
    hist_features = get_all_color_hist_features(balanced_df["image"], hist_bins)
        
    X_train = np.hstack((hog_features, spatial_features, hist_features))
    X_train = X_scaler.fit_transform(X_train)
    y_train = le.fit_transform(balanced_df['category'])
    svc_model = return_model()
    svc_model.fit(X_train, y_train)
    
    return svc_model, le, X_scaler
        
def get_all_color_bin_features(img_files, spatial_size):
    """
    Calculates color bin features for the given images by downsizing and taking each pixel as
    representative of the colors of the surrounding pixels in the full-size image.
    :param imgs: The images for which to calculate color bin features.
    :param shape: A tuple, (height, width) - the shape to which imgs should be downsized.
    :return: The color bin features for imgs.
    """
    # Sized to hold the ravelled pixels of each downsized image.
    features = [] #np.empty([imgs.shape[0], shape[0] * shape[1] * imgs.shape[3]])

    # Resize and ravel every image to get color bin features.
    for i, img_file in enumerate(img_files):
        img = cv2.imread(img_file)
        features.append(cv2.resize(img, spatial_size).ravel())

    return np.vstack(features)

def get_all_color_hist_features(img_files, nbins, bins_range=(0, 256)):
    """
    Calculates color histogram features for each channel of the given images.
    :param imgs: The images for which to calculate a color histogram.
    :param nbins: The number of histogram bins to sort the color values into.
    :param bins_range: The range of values over all bins.
    :return: The color histogram features of each channel for every image in imgs.
    """
    sample_img = cv2.imread(img_files.iloc[0])
    num_features = sample_img.shape[-1] * nbins
    hist_features = np.empty([len(img_files), num_features])

    for i, img_file in enumerate(img_files):
        img = cv2.imread(img_file)
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features[i] = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

def get_all_hog_features(img_files, colorspace, orient, pix_per_cell, cell_per_block):
    total_hog_features = []
    for idx, img_file in enumerate(img_files):
        # Read in image
        img = cv2.imread(img_file)
        #img = mpimg.imread(item['image'])
        if colorspace != 'RGB':
            conv_img = cv2.cvtColor(img, eval("cv2.COLOR_BGR2" + colorspace))
        else:
            conv_img = np.copy(img)
            
        hog_features = []
        for chan in range(conv_img.shape[2]):
            hog_features = np.hstack([hog_features, preprocess.get_hog_features(conv_img[:,:,chan], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)])
            
        total_hog_features.append(hog_features)
        
    return np.array(total_hog_features)

# def get_sliding_window_preds(imgs, model, scaler, le, orient, pix_per_cell, cell_per_block, y_start=400, y_stop=656, cell_stride=2,
#                              scale=1, colorspace="HLS", spatial_size=(16, 16), hist_bins=32):
#     """
#     Gets detection predictions from a trained model on a sliding window over the given images.
#     :param imgs: The images for which to get predictions.
#     :param model: The model to make predictions.
#     :param scaler: The scaler used to normalize the data when training.
#     :param y_start: The pixel value on the y axis at which to start searching for cars (Top of
#                     search window).
#     :param y_stop: The pixel value on the y axis at which to stop searching for cars (Bottom of
#                    search window).
#     :param cell_stride: The stride of the sliding window, in HOG cells.
#     :param scale: The scale of the sliding window relative to the training window size
#                          (64x64).
#     :param color_space: The color space to which to convert the images.
#     :return: A heatmap of the predictions at each sliding window location for all images in imgs.
#     """
#     heatmaps = np.zeros([len(imgs)] + list(imgs[0].shape[:2]))
    
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell)-1 
#     # Get the patch coordinates relative to the original image scale
#     window_width_abs = np.int(window * scale)

#     for idx, img in enumerate(imgs):
#         img_tosearch = img[y_start:y_stop,:,:]
#         ctrans_tosearch = cv2.cvtColor(img_tosearch, eval("cv2.COLOR_BGR2" + colorspace))
#         if scale != 1:
#             imshape = ctrans_tosearch.shape
#             ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

#         # ch1 = ctrans_tosearch[:,:,0]
#         # ch2 = ctrans_tosearch[:,:,1]
#         # ch3 = ctrans_tosearch[:,:,2]

#         # Define blocks and steps as above
#         nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell)-1
#         nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell)-1 
#         nfeat_per_block = orient*cell_per_block**2

        
#         cells_per_step = 2  # Instead of overlap, define how many cells to step
#         nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#         nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
#         # Compute individual channel HOG features for the entire image
#         hog = np.array([[preprocess.get_hog_features(ctrans_tosearch[:,:,ch], orient, pix_per_cell, cell_per_block, feature_vec=True)] for ch in range(3)])
        
#         total_features = []
#         xlefts = []
#         ytops = []
#         for x_step in range(nxsteps):
#             for y_step in range(nysteps):
#                 y_pos = y_step * cell_stride
#                 x_pos = x_step * cell_stride
                
#                 print(hog.shape)

#                 # Extract HOG for this patch
#                 patch_HOG = hog[:, y_pos:y_pos + nblocks_per_window,
#                                       x_pos:x_pos + nblocks_per_window].ravel()
                
#                 xleft = x_pos * pix_per_cell
#                 xlefts.append(xleft)
                
#                 ytop = y_pos * pix_per_cell
#                 ytops.append(ytop)

#                 # Extract the image patch
#                 subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
#                 # Get color features
#                 spatial_features = preprocess.bin_spatial(subimg, size=spatial_size)
#                 hist_features = preprocess.color_hist(subimg, nbins=hist_bins)

#                 # Combine and normalize features
#                 patch_features = np.hstack([patch_HOG, spatial_features, hist_features])
#                 total_features.append(patch_features)
                
#         total_features = np.vstack(total_features)

#         xlefts = np.vstack(xlefts)
#         ytops = np.vstack(ytops)
        
#         # Total Features Shape (912, 6744)
#         total_features_norm = scaler.transform(total_features)

#         # Make prediction
#         patch_preds = model.predict(total_features_norm)
#         # Reshape so it can be broadcast with the 3D heatmaps array.
#         patch_preds = np.reshape(patch_preds, [len(patch_preds), 1, 1])

#         # Add prediction to the heatmap
#         for patch_idx, (xleft, ytop) in enumerate(zip(xlefts, ytops)):
#             xleft_abs = int(xleft * scale) - 1
#             ytop_abs = int((ytop * scale) + y_start) - 1
            
#             #heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            
#             heatmaps[idx, ytop_abs:ytop_abs + window_width_abs,
#                         xleft_abs:xleft_abs + window_width_abs] += (le.inverse_transform([patch_preds[patch_idx]])[0] == "vehicles")

#     return heatmaps


# def get_sliding_window_preds(imgs, model, scaler, y_start=400, y_stop=656, cell_stride=2,
#                              scale=1, color_space="HLS"):

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
    hog_features = []
    for img_cropped in imgs_cropped:
        hog_features_image = []
        for ch in range(3):
            hog_features_image.append(preprocess.get_hog_features(img_cropped[:,:,ch], orient, pix_per_cell, cell_per_block, feature_vec=False))
        hog_features.append(hog_features_image)
        
    hog_features = np.array(hog_features)

    for x_step in range(num_steps_x):
        for y_step in range(num_steps_y):
            y_pos = y_step * cell_stride
            x_pos = x_step * cell_stride

            
            # Extract HOG for this patch
            c1_HOG = hog_features[:,
                                  0,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c2_HOG = hog_features[:,
                                  1,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c3_HOG = hog_features[:,
                                  2,
                                  y_pos:y_pos + nblocks_per_window,
                                  x_pos:x_pos + nblocks_per_window]
            c1_HOG_ravelled = np.reshape(c1_HOG, [len(imgs), -1])
            c2_HOG_ravelled = np.reshape(c2_HOG, [len(imgs), -1])
            c3_HOG_ravelled = np.reshape(c3_HOG, [len(imgs), -1])

            patch_HOG = np.concatenate((c1_HOG_ravelled, c2_HOG_ravelled, c3_HOG_ravelled), axis=1)

            xleft = x_pos * pix_per_cell
            ytop = y_pos * pix_per_cell

            # Extract the image patch
            patches = imgs_cropped[:,
                                 ytop:ytop + window,
                                 xleft:xleft + window]

            # Get color features
            patch_color_bins = np.array([preprocess.bin_spatial(patch, size=spatial_size) for patch in patches])
            patch_color_hists = np.array([preprocess.color_hist(patch, nbins=hist_bins) for patch in patches])


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

class RingBufSmoother(object):
        """
        Smoothes heatmaps across several iterations and applies thresholds
        """

        def __init__(self, shape, length=10, threshold=4):
            self.length = length
            self.data = np.zeros([length] + list(shape), dtype=np.float32)
            self.threshold = threshold
            self.index = 0
            self.count = 0

        def extend(self, x):
            """
            Adds array x to ring buffer.
            :param x: The element to add to the RingBuffer
            """
            self.data[self.index] = x
            self.index = (self.index + 1) % self.length
            
            self.count += 1
            if self.count > len(self.data):
                self.count = len(self.data)

        def mean(self):
            return np.mean(self.data[:self.count], axis = 0)
        
        def rolling_threshold(self):
            heatmap = self.mean() 
            heatmap[heatmap < self.threshold] = 0
            return heatmap
