##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicle_non]: ./output_images/vehicle_non.png
[vehicle_non_hog]: ./output_images/vehicle_non_hog.png
[test_results]: ./output_images/test_results.png
[heatmap]: ./output_images/heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

[HOG()](https://github.com/subhash/CarND-Vehicle-Detection/blob/master/vehicle-tracking.py#L77) 

I observed instances of the image classes - `vehicle` and `non-vehicle`:

![alt text][vehicle_non]

I parameterized the `skimage.hog()` function (`orientations`, `pixels_per_cell`, and `cells_per_block`) so that the contrast between the two classes will stand out:

![alt text][vehicle_non_hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

[Classifier.train()](https://github.com/subhash/CarND-Vehicle-Detection/blob/master/vehicle-tracking.py#L199)

For each image, I extract the HOG features and augment it with spatially binned (`32x32`) features and color histogram features (`bins=9`). This set of features is unravelled and trained by a SVM classifier with the appropriate labels. Care is taken to shuffle the samples first so that adjacency does not cause a bias. A 20% cross-validation test is extracted in order to test the accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

[Classifier.find_predictions()](https://github.com/subhash/CarND-Vehicle-Detection/blob/master/vehicle-tracking.py#L220)

We first scale the image as specified and then step through the image in `x` and `y`. The step size determines the overlap and I decided to go with `2 * cell_size` through experiments. The scales are provided by upstream code and depends on what kind of search is happening. For eg. for a full-fledged search I use `0.5, 0.66, 1.0` and for tracking vehicles, `0.5, 0.75`. The scales are determined how far the object being searched is from the camera's POV.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is the pipeline running on the test images. The cars in the images are being identified and bounded correctly. 

![alt text][test_results]


The classifier first starts with a broad-based search for vehicles across the image at various scales. Once vehicles are detected, we switch to optimal search - which includes searching around existing vehicles, and searching in the origin spaces (horizon, left and right corners) only.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=PRSXPaGTiGM)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The classifier generates bounding boxes (at various scales) where predictions are favourable. We use this to generate a heatmap and threshold values to retain maximally possible pixels. Using `scipy.ndimage.measurements.label()`, we separate these areas into different possible vehicle estimations and bound each estimation with a final box that represents a vehicle's position. Here are some examples of prediction boxes, heatmaps and final bounding boxes:

![alt text][heatmap]

The other way I prevent false positives is to recognize that an object cannot suddenly appear in a pipeline. It has to originate from either of the corners or the horizon in previous frames. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The biggest issue is deciding on the tradeoff between searching better and searching faster. More scales and more windows give better estimates but slows down the process
* The pipeline logic needs its own insights. My logic fails around sharp curves because the search area is affected
* The pipeline also gets confused with overlapping cars. This could be improved by each vehicles picking it's own updates from the list of prediction boxes
