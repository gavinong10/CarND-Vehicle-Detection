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

def train_model(df, colorspace, orient, pix_per_cell, cell_per_block):
    X_scaler = MinMaxScaler()
    le = LabelEncoder()
    
    # Create a model using all the data (Balance the vehicles and non-vehicles classes through sampling for removing bias)
    len_imbalance = sum(df["category"] == "vehicles") - sum(df["category"] == "non-vehicles")
    balanced_df = pd.concat([df[df["category"] == "non-vehicles"], \
                             df[df["category"] == "non-vehicles"].sample(len_imbalance, replace=True), \
                             df[df["category"] == "vehicles"]])

    X_train = get_all_hog_features(balanced_df, colorspace, orient, pix_per_cell, cell_per_block)
    X_train = X_scaler.fit_transform(np.array(X_train))
    y_train = le.fit_transform(balanced_df['category'])
    svc_model = return_model()
    svc_model.fit(X_train, y_train)
    
    return svc_model, le, X_scaler

def get_all_hog_features(df, colorspace, orient, pix_per_cell, cell_per_block):
    total_hog_features = []
    for (idx, item) in df.iterrows():
        # Read in image
        img = cv2.imread(item['image'])
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

# (More efficient window search) A single function that can extract features using hog sub-sampling and make predictions
def find_cars_hog(img, ystart, ystop, scale, svc, le, X_scaler, orient, pix_per_cell, cell_per_block, colorspace='YUV'):
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    # crop the image
    img_tosearch = img[ystart:ystop,:,:]
    # Do a color transform
    ctrans_tosearch = cv2.cvtColor(img_tosearch, eval("cv2.COLOR_BGR2" + colorspace))
    # Resize the image as a fraction
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    #Extract channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features,)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            
            if le.inverse_transform([test_prediction])[0] == "vehicles":
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return bboxes, draw_img
