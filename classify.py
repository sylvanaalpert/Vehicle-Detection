
import matplotlib.image as mpimg
import os
from fnmatch import fnmatch
import numpy as np
import cv2
import utils
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_image_filenames(datadir): 
    pattern = "*.png"
    img_fns = []
    
    for path, subdirs, files in os.walk(datadir):
        for name in files:
            if fnmatch(name, pattern):
                img_fns.append(os.path.join(path, name))
    return img_fns

def train_classifier(datadir, outputfile):
    
    # Read data and calculate features
    vehicles = get_image_filenames(os.path.join(datadir, 'vehicles'))
    non_vehicles = get_image_filenames(os.path.join(datadir, 'non-vehicles'))
    
    color_space = 'YCrCb'
    n_orient = 9 
    pix_per_cell = 16 
    cell_per_block = 2 
    spatial_size = (32, 32) 
    hist_bins = 32
    hog_channel = 'ALL'
    
    vehicle_feat = extract_features(vehicles, color_space, spatial_size, hist_bins, n_orient, pix_per_cell, cell_per_block, hog_channel)
    non_vehicle_feat = extract_features(non_vehicles, color_space, spatial_size, hist_bins, n_orient, pix_per_cell, cell_per_block, hog_channel)
        
    # Train classifier 
    X = np.vstack((vehicle_feat, non_vehicle_feat)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(vehicle_feat)), np.zeros(len(non_vehicle_feat))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:', n_orient,'orientations', pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    results = (svc,
        X_scaler,
        color_space, 
        n_orient,
        pix_per_cell,
        cell_per_block, 
        spatial_size, 
        hist_bins)
    
    save_results(outputfile, results)
        
    return results

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        
        # Read in each one by one
        image = mpimg.imread(file) * 255.0
        image = image.astype(np.uint8)
        
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            conv = eval('cv2.COLOR_RGB2' + color_space)
            feature_image = cv2.cvtColor(image, conv)
        else: 
            feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = utils.bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = utils.color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(utils.get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = utils.get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    
    return features

def save_results(output_file, results):
    print('Saving data to pickle file...')
    try:
        with open(output_file, 'wb') as pfile:
            pickle.dump(
                {
                    'svc': results[0],
                    'scaler': results[1],
                    'color_space': results[2],
                    'n_orient' : results[3],
                    'pix_per_cell' : results[4],
                    'cell_per_block' : results[5],
                    'spatial_size' : results[6],
                    'hist_bins' : results[7]
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', output_file, ':', e)
        raise
    print('Data saved in pickle file.')