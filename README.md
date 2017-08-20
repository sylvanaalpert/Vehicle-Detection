## **Vehicle Detection Project**
### By Sylvana Alpert

The goals of this project were:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a linear SVM classifier
* Explore color spaces and extract binned color features to be used for classification.  
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-no-car.jpg
[image4]: ./output_images/positive_detections.jpg
[image5]: ./output_images/heatmap.jpg
[image6]: ./output_images/thresholded.jpg
[image7]: ./output_images/pipeline.jpg


---

### Histogram of Oriented Gradients (HOG) and Classification

The code in file `classify.py` contains all the steps necessary for training and testing the SVM classifier. The pipeline for training is contained in function `train_classifier()`. This function gets all vehicle and non-vehicle images and extract both color and HOG features from them. Furthermore, the data is normalized to have zero mean and unit variance and randomly split into training and testing sets.     

HOG were extracted using `skimage.feature.hog()` (code can be found in `utils.py`) using parameters `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` and the `YCrCb` color space. These parameters were selected due to the classifier's high accuracy score on the testing data (98%). Different combinations of parameters were evaluated (mostly varying the color space, the number of orientations and the pixels per cell) and the chosen parameters yielded the best possible results for the given data.

Here are examples of vehicle and non-vehicle images and their corresponding HOG features:

![alt text][image1]

I trained a linear SVM using `sklearn.svc.LinearSVC()`. The code can be found in function `train_classifier()`. The classifier was saved to a pickle file to be loaded later in subsequent runs of the video frame data.

### Sliding Window Search

The video was processed using the code in `detection.py`. This file contains the `VehicleDetector` class, with all relevant functions for sliding window search per frame, false positive removal and frame by frame tracking of vehicles.

The sliding window search is implemented in function `find_cars()`, which resizes the input image according to a chosen scale and extracts subimages of fixed size from the frame. Alternatively, windows of different sizes could have been used without the need to rescale the input frame.

The size of the window used was 64x64 and the input frame image was rescaled using the following scale factors:
[0.70, 0.9, 1, 1.5, 2, 2.5]. These scales yield effective window size that vary from 44x44 to 160x160 pixels. These values seemed reasonable for the size of the cars that appear in the video. The overlap between windows was automatically calculated, depending on another parameter (cell_per_step, which was set to 2). This is equivalent to having a step of 32 pixels (or 50% overlap).

Here are some example images after going through the pipeline for vehicle detection:

![alt text][image4]                   
---

### Video Implementation

Here's a [link to my video result](https://youtu.be/7fIIkch06Pc).

After the windows with vehicles were identified for each frame, duplicates and false positives had to be cleaned up. To do so, a heatmap of positive detections was created and thresholded using a double threshold approach: if at least one pixel in the region crossed the high threshold, pixels from that region that crossed the lower threshold were marked as a vehicle. The thresholded heatmap was the current frame heatmap summed with the identified regions for cars from the previous frame. Bounding boxes were constructed to encompass the detected areas and plotted over the frame. The code used for cleaning up duplicates and false positives can be found in `detection.py`, mostly in function `remove_false_positives()`, which creates the heatmap and applies the threshold.

Here's an example result showing the heatmap from a test image going through the pipeline, the result of the threshold applied to remove false positives (before applying `scipy.ndimage.measurements.label()`) and the bounding boxes then overlaid on the last frame of video:

Here is the heatmap for the test image:

![alt text][image5]

Here is the thresholded heatmap:

![alt text][image6]

Here the resulting bounding boxes are drawn onto the input test image:

![alt text][image7]

---

### Discussion

One clear problem I faced was that some false positive detections exhibited the values in the same range as regions with vehicles, and could not be differentiated from them. Presumably, using a larger set of scales to scan the image for vehicles and a larger overlap between them, might help accentuate the differences between a true vehicle and one region falsely identified as one. The reason this was not implemented here was for performance and computation time, which was already quite long for the processing of the project video (about 2secs per frame).
Even though the detection of cars works quite reliably, this approach does not render itself as a solution for online tracking, as the computational times per frame prevent real-time processing. A different approach such as object localization and tracking using CNNs could work better.
