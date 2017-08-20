
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import utils
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt



'''
A class that performs vehicle detection over a video
'''
class VehicleDetector(object):

    def __init__(self, svc, X_scaler, color_space, n_orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        self.svc = svc
        self.X_scaler = X_scaler
        self.color_space = color_space
        self.n_orientations = n_orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.prev_heatmap = None


    def process(self, input_video, output_video):
        original = VideoFileClip(input_video)
        #original = original.set_start(4.00, False)
        #original = original.set_end(10.00)
        marked = original.fl_image(self.pipeline_for_frame)
        marked.write_videofile(output_video, audio=False)

    def find_cars(self, img, ystart, ystop, scale):
        bb_list = []

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = utils.convert_color(img_tosearch, conv=self.color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.n_orientations*self.cell_per_block**2

        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = utils.get_hog_features(ch1, self.n_orientations, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = utils.get_hog_features(ch2, self.n_orientations, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = utils.get_hog_features(ch3, self.n_orientations, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop : ytop + window, xleft : xleft + window], (64,64))

                # Get color features
                spatial_features = utils.bin_spatial(subimg, size=self.spatial_size)
                hist_features = utils.color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bb = (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                    bb_list.append(bb)

        return bb_list

    def add_heat(self, heatmap, bbox_list):

        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        heatmap = np.clip(heatmap, 0, 255)
        return heatmap

    def apply_threshold(self, heatmap, high_threshold, low_threshold):

        new_heatmap = np.zeros_like(heatmap)

        if (self.prev_heatmap is not None):
            heatmap = heatmap + self.prev_heatmap

        labels = label(heatmap)
        for car_number in range(1, labels[1]+1):

            hot_area = (labels[0] == car_number)
            values = heatmap[hot_area]

            if any(values > high_threshold):
                # this is a valid window
                car_area = hot_area & (heatmap > low_threshold)
                new_heatmap[car_area] = 1


        self.prev_heatmap = new_heatmap


        return new_heatmap


    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def remove_false_positives(self, image, bb_list):
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = self.add_heat(heat, bb_list)
        heatmap = self.apply_threshold(heatmap, 2, 1)

        return label(heatmap)


    def pipeline_for_frame(self, image):
        bbxs = []

        scales = (0.70, 0.9, 1, 1.5, 2, 2.5)
        for scale in scales:
            bb = self.find_cars(image, 400, image.shape[0], scale)
            bbxs.extend(bb)

        labels = self.remove_false_positives(image, bbxs)
        output = self.draw_labeled_bboxes(image, labels)

        return output
