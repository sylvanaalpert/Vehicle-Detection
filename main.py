import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from detection import VehicleDetector
import os
import cv2
import pickle
import classify
import numpy as np


if __name__ == "__main__":
    
    print_img = True
    
    classifier_file = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection/clf.p'
    
    if os.path.isfile(classifier_file):
        clf_data = pickle.load( open(classifier_file, "rb" ) )
        svc = clf_data["svc"]
        X_scaler = clf_data["scaler"]
        color_space = clf_data["color_space"]
        n_orient = clf_data["n_orient"]
        pix_per_cell = clf_data["pix_per_cell"]
        cell_per_block = clf_data["cell_per_block"]
        spatial_size = clf_data["spatial_size"]
        hist_bins = clf_data["hist_bins"]
    else:
        data_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection/data'
        (svc,
        X_scaler,
        color_space,
        n_orient,
        pix_per_cell,
        cell_per_block, 
        spatial_size, 
        hist_bins)  = classify.train_classifier(data_dir, classifier_file)

    if not print_img:
        
        d = VehicleDetector(svc, X_scaler, color_space, n_orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        video_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection'
        input_name = 'project_video.mp4'
        output_name = 'processed.mp4'
    
        in_file = os.path.join(video_dir, input_name)
        out_file = os.path.join(video_dir, output_name)
        d.process(in_file, out_file)
        
    else:
        data_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection/data'
        output_img_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection/output_images'
        
        car_img = os.path.join(data_dir, 'vehicles/GTI_MiddleClose/image0007.png')
        non_car_img = os.path.join(data_dir, 'non-vehicles/Extras/extra26.png')
        car_img = mpimg.imread(car_img) * 255.0
        car_img = car_img.astype(np.uint8)
        non_car_img = mpimg.imread(non_car_img) * 255.0
        non_car_img = non_car_img.astype(np.uint8)
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(car_img)
        plt.title('Vehicle')
        plt.subplot(1, 2, 2)
        plt.imshow(non_car_img)
        plt.title('Non-Vehicle')
        out_fn = os.path.join(output_img_dir, 'car-no-car.jpg')
        plt.savefig(out_fn, bbox_inches = 'tight', pad_inches = 0)
        
        img_dir = '/Users/sylvanaalpert/CarND/Starter_Kits/CarND-Vehicle-Detection/test_images'
        fn = os.path.join(img_dir, 'test6.jpg')
        test_img = mpimg.imread(fn)
        
        d = VehicleDetector(svc, X_scaler, color_space, n_orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        output = d.pipeline_for_frame(test_img)
        
        plt.figure(3)
        plt.imshow(output)
        plt.title('Test image')
        out_fn = os.path.join(output_img_dir, 'pipeline.jpg')
        plt.savefig(out_fn, bbox_inches = 'tight', pad_inches = 0)
    