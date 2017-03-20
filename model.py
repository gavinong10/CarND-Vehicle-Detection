import numpy as np
import cv2
from preprocess import bin_spatial, color_hist, get_hog_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import preprocess

def return_model():
    svc = LinearSVC()
    
    return svc

def train_model(df, colorspace, orient, pix_per_cell, cell_per_block, spatial_size=(16, 16), hist_bins=32):
    X_scaler = MinMaxScaler()
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
    sample_img = cv2.imread(img_files[0])
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

def get_sliding_window_preds(img_files, model, scaler, le, orient, pix_per_cell, cell_per_block, y_start=400, y_stop=656, cell_stride=2,
                             scale=1, colorspace="HLS", spatial_size=(16, 16), hist_bins=32):
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
    :param window_scale: The scale of the sliding window relative to the training window size
                         (64x64).
    :param color_space: The color space to which to convert the images.
    :return: A heatmap of the predictions at each sliding window location for all images in imgs.
    """
    heatmaps = np.zeros([len(img_files)] + list(cv2.imread(img_files[0]).shape[:2]))
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    # Get the patch coordinates relative to the original image scale
    window_width_abs = np.int(window * scale)

    for idx, img_file in enumerate(img_files):
        img = cv2.imread(img_file)
        img_tosearch = img[y_start:y_stop,:,:]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, eval("cv2.COLOR_BGR2" + colorspace))
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2

        
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = preprocess.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = preprocess.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = preprocess.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        hog_features = np.vstack([hog1,hog2,hog3])

        total_features = []
        xlefts = []
        ytops = []
        for x_step in range(nxsteps):
            for y_step in range(nysteps):
                y_pos = y_step * cell_stride
                x_pos = x_step * cell_stride

                # Extract HOG for this patch
                c1_HOG = hog1[y_pos:y_pos + nblocks_per_window,
                                      x_pos:x_pos + nblocks_per_window]
                c2_HOG = hog2[y_pos:y_pos + nblocks_per_window,
                                      x_pos:x_pos + nblocks_per_window]
                c3_HOG = hog3[y_pos:y_pos + nblocks_per_window,
                                      x_pos:x_pos + nblocks_per_window]
                c1_HOG_ravelled = c1_HOG.ravel()
                c2_HOG_ravelled = c2_HOG.ravel()
                c3_HOG_ravelled = c3_HOG.ravel()

                patch_HOG = np.hstack((c1_HOG_ravelled, c2_HOG_ravelled, c3_HOG_ravelled))
                
                xleft = x_pos * pix_per_cell
                xlefts.append(xleft)
                
                ytop = y_pos * pix_per_cell
                ytops.append(ytop)

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
                # Get color features
                spatial_features = preprocess.bin_spatial(subimg, size=spatial_size)
                hist_features = preprocess.color_hist(subimg, nbins=hist_bins)

                # Combine and normalize features
                patch_features = np.hstack([patch_HOG, spatial_features, hist_features])
                total_features.append(patch_features)
                
        total_features = np.vstack(total_features)

        xlefts = np.vstack(xlefts)
        ytops = np.vstack(ytops)
        
        total_features_norm = scaler.transform(total_features)

        # Make prediction
        patch_preds = model.predict(total_features_norm)
        # Reshape so it can be broadcast with the 3D heatmaps array.
        patch_preds = np.reshape(patch_preds, [len(patch_preds), 1, 1])

        # Add prediction to the heatmap
        for patch_idx, (xleft, ytop) in enumerate(zip(xlefts, ytops)):
            xleft_abs = int(xleft * scale) - 1
            ytop_abs = int((ytop * scale) + y_start) - 1
            
            #heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            
            heatmaps[idx, ytop_abs:ytop_abs + window_width_abs,
                        xleft_abs:xleft_abs + window_width_abs] += (le.inverse_transform([patch_preds[patch_idx]])[0] == "vehicles")

    return heatmaps
# # Define a single function that can extract features using hog sub-sampling and make predictions
# def find_cars(imgs, ystart, ystop, scale, svc, le, X_scaler, orient, pix_per_cell, cell_per_block, colorspace='YUV', spatial_size=(16, 16), hist_bins=32):
    
#     draw_img = np.copy(img)
#     img = img.astype(np.float32)/255
    
#     img_tosearch = img[ystart:ystop,:,:]
#     ctrans_tosearch = cv2.cvtColor(img_tosearch, eval("cv2.COLOR_BGR2" + colorspace))
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
#     ch1 = ctrans_tosearch[:,:,0]
#     ch2 = ctrans_tosearch[:,:,1]
#     ch3 = ctrans_tosearch[:,:,2]

#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell)-1
#     nyblocks = (ch1.shape[0] // pix_per_cell)-1 
#     nfeat_per_block = orient*cell_per_block**2
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell)-1 
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
#     # Compute individual channel HOG features for the entire image
#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb*cells_per_step
#             xpos = xb*cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

#             xleft = xpos*pix_per_cell
#             ytop = ypos*pix_per_cell

#             # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
#             # Get color features
#             spatial_features = bin_spatial(subimg, size=spatial_size)
#             hist_features = color_hist(subimg, nbins=hist_bins)

#             # Scale features and make a prediction
#             test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
#             #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
#             test_prediction = svc.predict(test_features)
                
#             if le.inverse_transform([test_prediction])[0] == "vehicles":
#                 xbox_left = np.int(xleft*scale)
#                 ytop_draw = np.int(ytop*scale)
#                 win_draw = np.int(window*scale)
#                 bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
#     return bboxes, draw_img

# # (More efficient window search) A single function that can extract features using hog sub-sampling and make predictions
# def find_cars_hog(img, ystart, ystop, scale, svc, le, X_scaler, orient, pix_per_cell, cell_per_block, colorspace='YUV'):
#     draw_img = np.copy(img)
#     img = img.astype(np.float32)/255
    
#     # crop the image
#     img_tosearch = img[ystart:ystop,:,:]
#     # Do a color transform
#     ctrans_tosearch = cv2.cvtColor(img_tosearch, eval("cv2.COLOR_BGR2" + colorspace))
#     # Resize the image as a fraction
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
#     #Extract channels
#     ch1 = ctrans_tosearch[:,:,0]
#     ch2 = ctrans_tosearch[:,:,1]
#     ch3 = ctrans_tosearch[:,:,2]

#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell)-1
#     nyblocks = (ch1.shape[0] // pix_per_cell)-1 
#     nfeat_per_block = orient*cell_per_block**2
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell)-1 
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
#     # Compute individual channel HOG features for the entire image
#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
#     bboxes = []
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb*cells_per_step
#             xpos = xb*cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

#             xleft = xpos*pix_per_cell
#             ytop = ypos*pix_per_cell

#             # Scale features and make a prediction
#             test_features = X_scaler.transform(np.hstack((hog_features,)).reshape(1, -1))    
#             #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
#             test_prediction = svc.predict(test_features)
            
#             xbox_left = np.int(xleft*scale)
#             ytop_draw = np.int(ytop*scale)
#             win_draw = np.int(window*scale)
            
#             if le.inverse_transform([test_prediction])[0] == "vehicles":
#                 bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
#     return bboxes, draw_img

                
# # Define a function to compute binned color features  
# def bin_spatial(img, size=(32, 32)):
#     # Use cv2.resize().ravel() to create the feature vector
#     features = cv2.resize(img, size).ravel() 
#     # Return the feature vector
#     return features

# # Define a function to compute color histogram features  
# def color_hist(img, nbins=32, bins_range=(0, 256)):
#     # Compute the histogram of the color channels separately
#     channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
#     channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
#     channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
#     # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#     # Return the individual histograms, bin_centers and feature vector
#     return hist_features
