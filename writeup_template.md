#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[Pickle Training file]: ./train.p
[Pickle Validation file]: ./Valid.p
[Pickle Test File]: ./test.P
[image4]: ./Online-Test/Images/02131.ppm
[image5]: ./Online-Test/Images/04564.ppm
[image6]: ./Online-Test/Images/05713.ppm
[image7]: ./Online-Test/Images/09159.ppm
[image8]: ./Online-Test/Images/05261.ppm

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/akiyer73/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 34799
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

Here is an exploratory visualization of the data set.
A sample from each of the sign types (ClassID) are printed and the images have several limitations:
1. Located at different parts of the pictures.
2. not completely oriented towards the computer (non frontal).
3. There are other objects in the background.
4. Different brightness and contrast.
5. parts of other signboards are present.

![Image Samples][ ./examples/image_samples.jpg]

The Images per class ID are not uniform. 
A x-y plot of the various images based on the classid was made and we can see that the quantities per signboard type varies a lot. 
The class id is on the X axis (0-42) and the samples range from low hundreds to around two thousand.

![X-Y Plot][ ./examples/xyplot.jpg]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth to ninth code cells of the IPython notebook.

#Step-1 Grayscaling
As a first step, I decided to convert the images to grayscale because since traffic signs will not change with color, it will be lower load and provide better accuracy.

Here is an example of a traffic sign image before and after grayscaling.


![Example Color Traffic sign][.\examples\RGBsign.png]
![Converted Grayscale sign  ][.\examples\grayimg.png]

As one can see, the images do not have a good image histogram.
![Converted Grayscale Histograms  ][.\examples\gray_hist.png]

#Step-2 Histogram Equalization
To better classify the images, the images were Histogram Equalized.
Here is an example of a traffic sign image after Histogram Equalization.

![Histogram Equalized sign  ][.\examples\histEqimg.png]
As one can see, the images have a better image histogram.
![Equalized Histograms  ][.\examples\post_eq_hist.png]
Step 3: Normalization
As a last step, I normalized the image data because normalizing image provides uniform values (0 to1) and applies weights uniform across the images.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the (10-13) code cells of the IPython notebook.  
The training set uses X_train as the set of images to train and X_valid for validating the model.

To cross validate my model, I randomly split the training data into a training set and validation set. 

My final training set had X number of images. My validation set and test set had Y and Z number of images.


The 15th code cell of the IPython notebook contains the code for augmenting the data set. 
I decided to generate additional data because, this will again validate the model against newer image sets (traffic signs).

To add more data to the the data set, I used the following techniques , downloaded the tar file. Further this data is in pm format and hence resized and converted to grayscale. 
Here is an example of an original image and an augmented image:
I downloaded additional images from the German traffic sign website as below.
[image4]: ./Online-Test/Images/02131.ppm
[image5]: ./Online-Test/Images/04564.ppm
[image6]: ./Online-Test/Images/05713.ppm
[image7]: ./Online-Test/Images/09159.ppm
[image8]: ./Online-Test/Images/05260.ppm


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenthth cell of the ipython notebook. 
This follows a basic LeNet archotecture. 
[Lenet]: ./examples/lenet.png
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image, converted to grayscale		| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1X1 stride, Valid padding,					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flatten 
| Fully connected		| (256, 128)   									|
| Relu   				|         					    				|
| Fully connected		| (128, 86)   									|
| Relu   				|         						    			|
| Dropout		    	|	probability = 0.6							|
| Output    			|	43 outputs									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the twelfth cell of the ipython notebook. 

To train the model, I used an Epoch of 160, learning rate of 0.0008, batch size of 128

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 92%
* validation set accuracy of 100% 
* test set accuracy of 100%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used a proven LeNet model to start the process. I have retained the architecture as this has satisfied the requirements.

* What were some problems with the initial architecture?
The initial architecture does not give very high accuracy and hence the filter sizes were changes in the convolution layers.
Being RGB images, We had to first convert the image sto grayscale. We used Open Cv for the same. The accuracy was lower and hence the model was adjusted.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
To adjust the architecture, the depths of the convolution layers were increased.
Relu activatiuon was used in the model.
The final fully connected layer also had a dropout introduced to avoid overfitting. Keep Probability was kept at 0.6

* Which parameters were tuned? How were they adjusted and why?
Epochs, batch size and learning rate were adjusted primarily to get better accuracy.
The model seemed to have saddle points at around 90%. Due to this the epochs were considerably increased.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layers will work because of the fixed type of images, with limited variations.
A dropout layer will help in reducing the chances of overfiting.

If a well known architecture was chosen:
* What architecture was chosen? The Lenet architecture was chosen as a starting point.
* Why did you believe it would be relevant to the traffic sign application? The same was earlier used in a similar image cladsification activity.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The acuracy of the model 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image4]: ./Online-Test/Images/02131.ppm
[image5]: ./Online-Test/Images/04564.ppm
[image6]: ./Online-Test/Images/05713.ppm
[image7]: ./Online-Test/Images/09159.ppm
[image8]: ./Online-Test/Images/05260.ppm

The difference between the original data set and the augmented data set is the following:
Original images have varying sizes.
Original images are in 3 layers.
Final used images have 32X32 size and converted to grayscale, equalized histogram, normalized and used for testing.
The augmented grayscale images are below.

[image4]: ./examples/02131.jpg
[image5]: ./examples/04564.jpg
[image6]: ./examples/05713.jpg
[image7]: ./examples/09159.jpg
[image8]: ./examples/05260.jpg

These images have clas ID of 20,18,10,5,33 respectively.

These were correctly identified by the model.




The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.
The selected images were correctly predicted.

Here are the results of the prediction:



| Image			                                  |     Prediction	                 	| 
|:---------------------:                          |:---------------------------------------------:| 
| Dangerous curve to the right                    | Dangerous curve to the right   									| 
| General caution     			                  | General caution 										|
| No passing for vehicles over 3.5 metric tons	  | No passing for vehicles over 3.5 metric tons |
| Speed limit (80km/h)	      		              | Speed limit (80km/h)					 				|
| Turn right ahead			                      | Turn right ahead      							|





The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 'Dangerous Curve to the right sign) (probability of 99.9), and the image does contain a 'Dangerous curve to the right' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Dangerous curve to the right   									| 
| .00     				| Slippery road 										|
| .00					| No passing											|
| .00	      			| No passing for vehicles over 3.5 metric tons					 				|
| .00				    | End of no passing      							|

For the second image, the model is very sure that this is a 'General Caution' sign (probability of 100), and the image does contain a 'General Caution' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.         			| General Caution   									| 
| .00     				| Speed limit (30km/h) 										|
| .00					| Speed limit (60km/h)											|
| .00	      			| Road narrows on the right					 				|
| .00				    | Beware of ice/snow      							|


For the third image, the model is quite sure that this is a 'No passing for vehicles over 3.5 metric tons' sign (probability of 99.81), and the image does contain a 'No passing for vehicles over 3.5 metric tons' sign. This image looks blurred and has some part of another sign on top of this sign board visible. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.81        			| No passing for vehicles over 3.5 metric tons   									| 
| .10     				| Speed limit (80km/h) 										|
| .00					| Speed limit (60km/h)											|
| .00	      			| Speed limit (100km/h)					 				|
| .00				    | Yield      							|

For the fourth image, the model is  sure that this is a 'Speed limit (80km/h)' sign (probability of 100), and the image does contain a 'Speed limit (80km/h)' sign. Visually this looks the most difficult to identify, as the image is dark. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.         			| Speed limit (80km/h)   									| 
| .00     				| Speed limit (30km/h)) 										|
| .00					| Speed limit (60km/h)											|
| .00	      			| Speed limit (20km/h)					 				|
| .00				    | No passing for vehicles over 3.5 metric tons      							|

For the fifth image, the model is  sure that this is a 'Speed limit (80km/h)' sign (probability of 100), and the image does contain a 'Speed limit (80km/h)' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.         			| Turn right ahead   									| 
| .00     				| Ahead only 										|
| .00					| Go straight or left											|
| .00	      			| Keep left					 				|
| .00				    | General caution      							|