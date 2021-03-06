# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Writeup

In this project, I used my knowledge of deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the German Traffic Sign Dataset. After the model was trained, I then tried out my model on images of German traffic signs that I find on the web.

### Dependencies

This lab requires:
* Tensorflow
* Matplotlib
* numpy
* scikit-learn
* pickle
* random

### Dataset and Repository

I used a pickled dataset in which images are already resized to 32x32. It contains a training, validation and test set. The German Traffic Sign dataset consists of 43 unqiue traffic signs. 

Link to dataset:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

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

[image1]: ./Images/visualize_training.png "Visualization"
[image2]: ./Images/Gray_and_color.PNG "Grayscaling"
[image3]: ./Images/normalized.png "Normalized"
[image4]: ./test-data/NoEntry.jpg "Traffic Sign 1"
[image5]: ./test-data/roadWork.jpg "Traffic Sign 2"
[image6]: ./test-data/speed30.jpg "Traffic Sign 3"
[image7]: ./test-data/stop.jpg "Traffic Sign 4"
[image8]: ./test-data/yield.jpg "Traffic Sign 5"
[image9]: ./Images/test_data_proc.PNG "Data Processed"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

Here we can see that the training data is not well distributed among the 43 classes. This maybe intentionaly as in the real world there are some traffic signs that we see more often then the other. This may cause the classification results to be skewed. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this allows us to reduce complexity and allows the classifier to focus more on features that set each traffic sign apart. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Next, I normalized the image data because this create a similar data distribution among each input parameter. It also helps the network train faster by converging faster. Finally the data distribution would be centered around zero.

I didnt decided to generate additional data at this point to keep things simple. But we can apply image augmentation to the dataset such as rotation, shift or zoom. This will help the model generalize better on new images.

Finally, we one hot encode the labels. This allows for easy comparision between the classification results and truth labels. 

Here is what the label looks like before and after one_hot_encode:

Label: 41

one_hot: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 RGB image   							| 
| Convolution 2x2     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|			Activation Function									|
| Max pooling	2x2    	| 2x2 stride,  outputs 14x14x6 |
| Convolution 2x2     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|			Activation Function									|
| Max pooling	2x2    	| 2x2 stride,  outputs 5x5x16 |
| Flatten	    |       									|
| Fully connected		| 400 -> 200 outputs        									|
| Fully connected		| 200 -> 100 outputs        									|
| Fully connected		| 100 -> 10 outputs        									|
| Softmax				| Classifer        									|

The model I chose is based on the LeNet architecture. It has shown great success in similar classification tasks in the past. I tweaked the FC layers little to get desired performance.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Building the training pipeline.

First I passed the inputs through the network architecture undergoing few layers of convolution and max pooling followed by fully connected. This is the forward pass through the network.

Next I calculated the loss using the softmax cross-entropy loss function and input labels. Further we take the reduced mean of the loss to complete the loss function. 

Next, I added some regularization in the form of L2 regularization. This will help the model generalize better. 

To correct the performance of the model I calculated the cost of last result. This cost is used by the optimizer to propogate backwards through the network to adjust corresponding weights. For omptimization purposes I chose Adam optimizer as it is fairly new and has show great performance in similar applications. The goal of the optimizer is to minimize cost.

Finally, we evaluate the performance of the model. We start by comparing the result with truth labels and then take the reduced mean to perform accuracy operation.

I chose the following hyper parameters after adopting industry recommendations as a starting point and using a trial and error approach to refine further:

* Epochs = 50
* Batch_size = 128
* Learning_rate = Start with 0.01 and exponentially decays over time. This helps prevent the network from overfitting. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9912354
* validation set accuracy of 0.9539682 
* test set accuracy of 0.9275534

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?<br>
I used the LeNet architecture and tweaked the fully-connected layer outputs.

* What were some problems with the initial architecture?<br>
Didnt face any issue with regards to the architecture since I went with one that had been tested and proved.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. <br>
I didnt make any network changes apart from the outputs of FC layer. Having a larger number allows the model to have more features available for better classification.
I did add L2 regularization to prevent overfitting.

* Which parameters were tuned? How were they adjusted and why?<br>
I played around with batch_size and number of epochs. Having a larg number for epochs helped the model train longer and improve accuracy. I was cautious to not have it too large which may result in overfitting. I attempted batch size of 256 and 128 and decided to go with 128 due to better results. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?<br>
Convolutional networks excel in detecting visual patterns from images by creating combinations of pixels. L2 regularization was used to avoid over-fitting. 

If a well known architecture was chosen:
* What architecture was chosen?<br>
LeNet Architecture
* Why did you believe it would be relevant to the traffic sign application?<br>
LeNet has been used for other similar image classification purposes.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br>
The model acheive a validation accuracy of 0.91 which is slightly below the desired 0.93 and there could be many reasons for this. One of them being not enough epochs to train the model completely. Further the intialization of the weights being random, may not have started at the best point for this run. 
The model had a testing accuracy of 0.89 which may be due to its inability to generalize well on new images. Similar performance can be seen with new images from the web.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Preprocessed test images look as follows:
![alt_text][image9]

The first image might be difficult to classify because the no entry symbol is not cropped the same as the training images and since I have not augmented my training dataset with options such as zoom it doesnot generalize with different scaling of images. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No vehicles   									| 
| Road work     			| Road work 										|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| Stop	      		| Speed limit (100km/h)					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares similar to the accuracy on the test set of 92% keeping in mind the difference in number of images for each. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

For the first image, the model is not so sure that this is a No entry sign (probability of 0.5), but the image does contain a no No entry. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| No entry   									| 
| .31     				| Bumpy road 										|
| .05					| Traffic signals											|
| .04	      			| Speed limit (120km/h)					 				|
| .02				    | Bicycles crossing      							|


For the second image, the model is absolutly sure that this is a Road work sign (probability of 0.9), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Road work   									| 
| .00     				| General caution 										|
| .00					| Double curve											|
| .00	      			| Turn right ahead					 				|
| .00				    | Speed limit (30km/h)      							|

For the thrid image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.8), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .86         			| Speed limit (30km/h)   									| 
| .08     				| Wild animals crossing 										|
| .02					| Speed limit (50km/h)											|
| .00	      			| End of speed limit (80km/h)					 				|
| .00				    | Double curve      							|

For the fourth image, the model is relatively sure that this is a No passing for vehicles over 3.5 metric tons sign (probability of 0.9), and the image does contain a No passing for vehicles over 3.5 metric tons sign. It is infact a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			|  No passing for vehicles over 3.5 metric tons  									| 
| .03     				| No passing										|
| .02					| Stop 											|
| .01	      			| Speed limit (100km/h)					 				|
| .01				    | End of no passing by vehicles over 3.5 metric tons      							|

For the fifth image, the model is relatively sure that this is a yield sign (probability of 0.9), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Yield   									| 
| .00     				| Stop 										|
| .00					| No vehicles											|
| .00	      			| Keep left					 				|
| .00				    | No passing      							|



