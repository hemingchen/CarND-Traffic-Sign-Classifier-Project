#**Traffic Sign Recognition** 

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/hemingchen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

As in the notebook.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Step 1: Images were converted to grayscale for faster computing and better robustness under various color distortions.

Step 2: Grayscaled images were then normalized to further enhance robustness under various exposure levels, etc. It could also accelerates the converging speed.

Step 3: Images from Step 2 were used as inputs to training.

An example of an image processed through above pipeline can be found in the notebook right above "Model Architecture" section.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					| 			  									|
| Max pooling			| 2x2 stride,  outputs 5x5x64       			|
| Flatten				| 5x5x64 flattened to (1600, 1) vector			|
| Fully connected 		| input 1600, output 120						|
| RELU					| 			  									|
| Fully connected 		| input 120, output 84							|
| RELU					| 			  									|
| Fully connected 		| input 84, output 43							|
| Softmax		 		| 												|
| Final output			| 												|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used softmax cross entropy as the loss function and trained model with Adam Optimizer.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.965
* test set accuracy of 0.943

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

-- I first tried training the LeNet model with default parameters quickly without grayscale the images, because I thought the colors in the signs should be a part of the feature. An accuracy of about 0.89 was achieved.

* What were some problems with the initial architecture?

-- The accuracy of the initial approach only improved a bit after applying both grayscale and normalization to the images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

-- To further increase accuracy/resolve the under fitting issue, I increased the 1st and 2nd convolution layer output size from 6 to 32 and 16 to 64, respectively, since the traffic sign has much more classes (43) than the default parameters that were tuned for digit recognition with only 10 classes.

* Which parameters were tuned? How were they adjusted and why?

-- I only tuned the batch size, 64 seemed a good one for the final model.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

-- Convolution layer worked well because the traffic signs have complex patterns and number of classes is large. Feeding the image directly to a fully connected NN will not work well. I didn't use dropout since my final model already satisfied the submission requirements. But if my model starts to overfit the data, I will use it.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

-- N/A

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five German traffic signs that I found on the web can be found in the notebook, which are:
1. Speed limit (30km/h)
2. Right-of-way
3. Road work
4. Yield
5. Ahead only

All images were recognized correctly.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Right-of-way 			| Right-of-way									|
| Road work				| Road work										|
| Yield					| Yield											|
| Ahead only      		| Ahead only					 				|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.943

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Image 1
The model easily identified it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (30km/h)  						|
| 0.0     				| Speed limit (50km/h) 							|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Speed limit (80km/h)			 				|
| 0.0				    | Roundabout mandatory 							|


Image 2
The model easily identified it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Right-of-way at the next intersection			|
| 0.0     				| Beware of ice/snow 							|
| 0.0					| End of no passing								|
| 0.0	      			| End of speed limit (80km/h)	 				|
| 0.0				    | Priority road 	 							|

Image 3
The model easily identified it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Road work										|
| 0.0     				| Dangerous curve to the right 					|
| 0.0					| No passing for vehicles over 3.5 metric tons	|
| 0.0	      			| Beware of ice/snow			 				|
| 0.0				    | Road narrows on the right 					|

Image 4
The model easily identified it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Yield											|
| 0.0     				| Road work			 							|
| 0.0					| No passing									|
| 0.0	      			| Turn right ahead				 				|
| 0.0				    | Priority road 	 							|

Image 5
The model easily identified it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Ahead only 									|
| 0.0     				| Speed limit (60km/h) 							|
| 0.0					| Dangerous curve to the right 					|
| 0.0	      			| Children crossing 			 				|
| 0.0				    | Yield			 	 							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

-- The trained network used the edges/shapes in the pictures to identify the traffic signs. For example, for the speed limit sign, the network searches for the circle and the numbers written in the circle in order to classify the sign.


