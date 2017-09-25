# **Traffic Sign Recognition**

## Writeup Template

You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[children_crossing]: ./images/children_crossing.png " "
[general_caution]: ./images/general_caution.png " "
[keep_right]: ./images/keep_right.png " "
[no_entry]: ./images/no_entry.png " "
[stop]: ./images/stop.png " "
[speed_limit_70]: ./images/speed_limit_70.png " "
[speed_limit_30]: ./images/speed_limit_30.png " "
[yield]: ./images/yield.png " "
[roundabout]: ./images/roundabout.png " "
[right_of_way]: ./images/right_of_way.png " "

[test_images]: ./images/test_images.png " "

[virgin_X_train_hist]: ./images/virgin_x_train_hist.png " "
[X_train_hist]: ./images/x_train_hist.png " "

[virgin_X_train_100]: ./images/virgin_x_train_100.png " "
[X_train_100]: ./images/x_train_100.png " "

[ex_virgin]: ./images/ex_virgin.png " "
[ex_pre-processed]: ./images/ex_pre-processed.png " "

[ex_random]: ./images/ex_random.png " "

[ex_data_aug_src]: ./images/ex_data_aug_src.png " "
[ex_data_aug_out]: ./images/ex_data_aug_out.png " "

## Rubric Points
Here I will consider the
[rubric points](https://review.udacity.com/#!/rubrics/481/view) individually
and describe how I addressed each point in my implementation.  

---
### Writeup / README

Provide a Writeup / README that includes all the rubric points and how you
addressed each one. You can submit your writeup as markdown or pdf.
You can use this template as a guide for writing the report.
The submission includes the project code.

You're reading it! and here is a link to my
[project code](./Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

**1. Provide a basic summary of the data set. In the code, the analysis should
be done using python, numpy and/or pandas methods rather than hardcoding
results manually.**

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

**2. Include an exploratory visualization of the dataset.**

To picture what the traffic signs look like, I printed a random image from the
training set and the output is as expected. The class id and the traffic sign
are matching.

![alt text][ex_random]
label: 2 - Speed limit (50km/h)

I printed the first 100 images in the training set. All the images appear to
be from the same class. So, data needs to be shuffled.

![alt text][virgin_X_train_100]

Checked how the images are distributed across all 43 classes. Based on the
histogram in magenta below, It is clear that the distribution is non-uniform.
Max number of images in any class is 2010. Added more images to each class
using data augmentation until all class are at 2010. The histogram in blue to
the right shows the distribution after data augmentation.

![alt text][virgin_X_train_hist] >> ![alt text][X_train_hist]

### Design and Test a Model Architecture

**1. Describe how you preprocessed the image data.**
What techniques were chosen and why did you choose these techniques?
Consider including images showing the output of each preprocessing technique.
Pre-processing refers to techniques such as converting to grayscale,
normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions"
part of the rubric, if you generated additional data for training, describe
why you decided to generate additional data, how you generated the data,
and provide example images of the additional data. Then describe the
characteristics of the augmented training set like number of images in the set,
number of images for each class, etc.)

- First step in my process is data augmentation. I ran the model without data
augmentation and got 95.5% accuracy on the `test_set`. So, decided to create
fake data to boost the accuracy. To create fake data, I used the method
suggested by Vivek Yadav in [this blog post](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3). This method involves
shifting, rotating, skewing (`Affine Transform`), and adjusting the brightness
of the image. I used this [tutorial from OpenCV](http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html) to perform all
these tasks. I had to do a lot of trial and error to figure out how much to
distort the original picture to produce more copies. If the distortion is too
much, valuable information is lost and it makes it harder for the model to
learn. Here is an example of an original image and an augmented image:

![alt text][ex_data_aug_src] >> ![alt text][ex_data_aug_out]

- Next step in pre-processing is to convert the image to grayscale and
normalize. It is not absolute necessary to convert the images to grayscale.
However, several papers suggested that getting rid of the colors will help the
model learn better. I am not 100% sure about this in every image processing
situation but in case of traffic signs, it is applicable. Because,
classification of traffic signs doesn't depend on the color. I haven't
personally experimented with color images. Normalizing the images will center
the data and specifies a common range for all inputs. Normalization also
ensures that wights and biases are updated evenly and gradients are passed
across the network efficiently. Because RELU filters everything below 0,
I normalized the data to be between 0 and 1 instead of -1 and 1.

    Here is an example of a traffic sign image before and after grayscaling and
normalization.

![alt text][ex_virgin] >> ![alt text][ex_pre-processed]

Shown below is first 100 images of the training set after all the preprocessing.

![alt text][X_train_100]


#### 2. Describe your final model architecture
Describe what your final model architecture looks like including model type,
layers, layer sizes, connectivity, etc.) Consider including a diagram and/or
table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image						|
| Convolution 4x4     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 4x4     	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling 2X2	   	| 2x2 stride,  outputs 16x16x128 				|
| Dropout               | keep_prob = 0.75                              |
| Convolution 4x4     	| 1x1 stride, same padding, outputs 16x16x256 	|
| RELU					|												|
| Max pooling 2x2	  	| 2x2 stride,  outputs 8x8x256  				|
| Dropout               | keep_prob = 0.75                              |
| Flatten               | outputs 16384                                 |
| Fully connected		| 1024 hidden nodes								|
| RELU                  |                                               |
| Dropout               | keep_prob = 0.75                              |
| Fully connected		| 512 hidden nodes								|
| RELU                  |                                               |
| Dropout               | keep_prob = 0.75                              |
| Fully connected		| 43 output nodes								|
| Softmax				| Prediction probabilities						|

- My network is rather simple. It is the basic idea of 2 or 3 convolutional layers
with pooling layers followed by 2 or 3 fully connected layers. My network is a
modification of the LeNet Architecture.
- Based on my experimentation, Dropout layers helped to boost the accuracy
by ~1.5%.
- Tested out filter sizes 5x5, 4x4, and 3x3. 4x4 is better than 5x5 better
than 3x3.
- Overlapping strides for pooling layer is interesting. 3x3 filter with 2x2
strides produced promising results but I haven't experimented enough with this.
- Adding one more conv>>RELU>>max_pool>>dropout block did not help. Probably
because the output at the end of current convolutional block is 8x8 and there
isn't much useful to extract. An inception layer would have helped.
- 64, 128, 256 filter combination is better than 32, 64, 128.
- 2 Fully connected layers are better than 3.
- 1024, 512 hidden node combination is better than 1024, 1024 or 512, 512.

#### 3. Describe how you trained your model
Describe how you trained your model. The discussion can include the type of
optimizer, the batch size, number of epochs and any hyperparameters such as
learning rate.

- I used the same optimizer, cost, and cross entropy functions from
the LeNet lab.
- Learning rate of 0.00075 is better than 0.01, 0.0001, and 0.0005
- Initially I used a batch size of 128 and 50 epochs and got a testing
accuracy of ~98% (97.7 to 98.0). It took 40 minutes on the latest Titan Xp.
And the Validation accuracy was ~99%. The Validation accuracy did not improve
after 20 epochs in most cases. However, it produced a very consistent testing
accuracy.
- After some experimentation, I reduced the batch size to 64 to force more
frequent weight updates. Dropping it down to 32 produced bit more run to run
variance in testing accuracy.
- Keep probability of 0.75 is better than 0.5, 0.6, 0.85, and 0.9
- With epochs: 15, batch size: 64, learning rate: 0.00075, and kepp prob: 0.75
model training takes under 8 minutes on the Titan Xp.

#### 4. Describe the approach
Describe the approach taken for finding a solution and getting the validation
set accuracy to be at least 0.93. Include in the discussion the results on the
training, validation and test sets and where in the code these were calculated.
Your approach may have been an iterative process, in which case, outline the
steps you took to get to the final solution and why you chose those steps.
Perhaps your solution involved an already well known implementation or
architecture. In this case, discuss why you think the architecture is
suitable for the current problem.

- Like I mentioned above, my approach is iterative. First, I used LeNet and did
not get anything greater than ~89% validation accuracy. My approach is to get
maximum results with making as little changes to LeNet.
- Normalizing and converting the input to grayscale and adding more filters to
LeNet's Conv layers gave me better than 93% validation accuracy.
- Following the tips from my Deep Learning Nanodegree and stacking
Conv>>RELU>>Conv>>RELU>>max_pool gave me 95.5% testing accuracy.
- Data Augmentation and Dropout layer increased the validation accuracy to
~99% and testing accuracy to ~98%.

My final model results were:
* validation set accuracy of 98.4%
* test set accuracy of 97.6%

My model doesn't have a batch-norm layer, inception layer, or l2-regularization
with weight decay. However, it still gets less than 2.5% error and trains under
8 minutes. I am satisfied with this result as far as the project submission is
considered. I will continue my work until I achieve better than 99% accuracy.

### Test a Model on New Images

**1. Choose five German traffic signs found on the web and provide them in
the report. For each image, discuss what quality or qualities might be
difficult to classify.**

Here are 10 German traffic signs that I found on the web:

![alt text][test_images]

The 8th and 9th images might be difficult to classify because of the
"Getty Images" logo on them covering the features of the signs. The rest must be
trivial.

**2. Discuss the model's predictions on these new traffic signs and compare
the results to predicting on the test set. At a minimum, discuss what the
predictions were, the accuracy on these new predictions, and compare the
accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in
more detail as described in the "Stand Out Suggestions" part of the rubric).**

Here are the results of the prediction:

| Image			        |     Prediction		|
|:---------------------:|:---------------------:|
| Speed limit (30km/h)	| Speed limit (30km/h)	|
| General caution       | General caution       |
| Speed limit (70km/h)  | Speed limit (70km/h)	|
| Keep right     		| Keep right			|
| No entry              | No entry              |
| Yield					| Yield		            |
| Stop           		| Stop      		    |
| Children crossing		| Children crossing     |
| Roundabout mandatory  | Turn right ahead      |
| Right-of-way at the next intersection | Right-of-way at the next intersection |

The model was able to correctly guess 9 of the 10 traffic signs, which gives
an accuracy of 90%. This is way less than ~98% accuracy on the test set but 10
is a much smaller number. If lot more images are given, I would expect a
similar accuracy to the test set.

**3. Describe how certain the model is when predicting on each of the 10 new
images by looking at the softmax probabilities for each prediction. Provide the
top 5 softmax probabilities for each image along with the sign type of each
probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the
rubric, visualizations can also be provided such as bar charts)**

Below is the list of all 10 images and their softmax probabilities.
All images except for the `children crossing` and `roundabout mandatory` were
identified with 100% probability. `children crossing` got selected with a 99.8%
probability which is impressive considering the noise in the image. At the same
time `roundabout mandatory` should have got better than 0.1% prediction.
As a result, it is miss classified as `Turn right ahead`.    

**Image #1:**

![alt text][speed_limit_30] `Speed limit (30km/h)`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Speed limit (30km/h)							|
| 0.00     				| Speed limit (50km/h) 							|
| 0.00     				| Speed limit (20km/h) 							|
| 0.00     				| Speed limit (80km/h) 							|
| 0.00     				| Speed limit (70km/h) 							|

**Image #2:**

![alt text][general_caution] `General caution`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| General caution								|
| 0.00     				| Bicycles crossing                             |
| 0.00					| Traffic signals								|
| 0.00	      			| Children crossing				 				|
| 0.00				    | Pedestrians        							|

**Image #3:**

![alt text][speed_limit_70] `Speed limit (70km/h)`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Speed limit (70km/h)   						|
| 0.00     				| Speed limit (20km/h) 		    				|
| 0.00					| Speed limit (30km/h)							|
| 0.00	      			| Speed limit (120km/h)			 				|
| 0.00				    | Road narrows on the right     				|

**Image #4:**

![alt text][keep_right] `Keep right`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Keep right   									|
| 0.00    				| Roundabout mandatory 	    					|
| 0.00  				| Speed limit (30km/h)							|
| 0.00      			| Turn left ahead				 				|
| 0.00				    | Speed limit (20km/h)  						|

**Image #5:**

![alt text][no_entry] `No entry`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| No entry   									|
| 0.00    				| End of all speed and passing limits 			|
| 0.00					| Speed limit (80km/h) 							|
| 0.00	      			| Keep right					 				|
| 0.00				    | Stop               							|

**Image #6:**

![alt text][yield] `Yield`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Yield     									|
| 0.00    				| Priority road 								|
| 0.00					| Road work							    		|
| 0.00	      			| End of all speed and passing limits			|
| 0.00				    | Right-of-way at the next intersection			|

**Image #7:**

![alt text][stop] `Stop`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Stop      									|
| 0.00     				| Speed limit (120km/h) 						|
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Ahead only					 				|
| 0.00				    | General caution      							|

**Image #8:**

![alt text][children_crossing] `Children crossing`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.998        			| Children crossing   							|
| 0.001    				| Pedestrians 									|
| 0.001					| Road narrows on the right						|
| 0.000	      			| Ahead only					 				|
| 0.000				    | Right-of-way at the next intersection  		|

**Image #9:**

![alt text][roundabout] `Roundabout mandatory`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.640       			| Turn right ahead  							|
| 0.341    				| Stop 					     					|
| 0.008					| Keep left						    			|
| 0.007	      			| No entry					 		        	|
| 0.001				    | Roundabout mandatory      					|

**Image #10:**

![alt text][right_of_way] `Right-of-way at the next intersection`

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Right-of-way at the next intersection			|
| 0.00     				| Beware of ice/snow 							|
| 0.00					| Double curve					     			|
| 0.00	      			| Pedestrians					 				|
| 0.00				    | End of speed limit (80km/h)					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
