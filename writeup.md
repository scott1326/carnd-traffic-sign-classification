# Traffic Sign Recognition Project


---

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: web_image0.png "Ahead Only"
[image1]: web_image1.png "Turn Right Ahead"
[image2]: web_image2.png "Go Straight or Left"
[image3]: web_image3.png "Right-of-Way at Next Intersection"
[image4]: web_image4.png "Speed Limit (30km/h)"

### Rubric Points

---


Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43


2. There is a visualization of the dataset in the notebook file.  I counted the number of instances of each class in the training, validation and test datasets.



Design and Test a Model Architecture

*1. Preprocessing*

To preprocess the data, I first converted the images from color to grayscale.  I used a multiplicative formula which had the added benefit of converting the data to the desired 32x32x1 shape.

I then normalized the data to ensure a mean of 0 and stddev of 1.0.

*2. Architecture*

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Layer 1: Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Layer 2: Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x64      	
|				RELU				|  |
| Max pooling | 2x2 stride, outputs 5x5x64
| Flatten to 1600 | |
| Layer 3: Fully connected		| 1600 outputs to 512 outputs        									|
| RELU				|         									|
|	Dropout					| keep_prob = 0.75												|
|	Layer 4: Fully connected					|	512 outputs to 256											|
| RELU ||
| Dropout |keep_prob = 0.75 |
| Layer 5, final layer: Fully connected | 256 outputs to 43 classes

I added dropout after the first two fully connected layers with a keep_probability of 0.75.  I initially trained on color images, but converting to grayscale and adding dropout enabled me to increase accuracy to around 95%.

I greatly increased the size of the convolution outputs and fully-connected layers.  This helped a great deal in increasing accuracy.

*3.Training*

As mentioned before, accuracy was a bit lower when training on color images.  Converting to grayscale helped a great deal.  After experimenting with several batch sizes, I settled on 128 as a good number.  A learning rate of 0.0008 seemed to be about right, and I trained for a total of 15 epochs.  I tried several variations of learning rates, epochs and batch sizes before settling on these, which seemed to perform best.


*4. Model Solution*

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 93.3%
* test set accuracy of 80.3%

Initially, I used the LeNet approach nearly identical to the one given in the online studies.  That model on color images never exceeded 85% training accuracy.  I realized converting to grayscale would enable faster processing and it increased accuracy as well.  However, obtaining an accuracy above 93% eluded me until I increased the depth of the convolution layers as well as the fully-connected layers.  Not until then, along with careful selection of the hyperparameters did I achieve the target accuracy rate.

I stuck with the LeNet model because it seems to perform well on classifying images.  I didn't really make any alterations other than increasing the size of the depth of the layers and adding dropout.


## Testing the Model on New Images


Here are five German traffic signs that I found on the web:

![alt text][image0]
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]


I resized these images to the same size as the dataset.  I performed the same preprocessing on them as I did with the initial dataset (grayscale, then normalization).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Ahead Only      		| Ahead Only  									|
| Turn Right Ahead     			|Turn Right Ahead										|
| Go Straight Or Right					| Traffic Signals											|
| Right-of-Way Next Intersection	      		| Right-of-Way Next Intersection				 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is almost identical to the accuracy of the test set, which was 80.3%.

### Predictions

Here are the softmax probabilities and predictions for the 5 web images.  The highest probability guess was correct in each case except for the third, in which the 3rd highest was the correct image (correct sign in *italics*). Interestingly, the model had a certainly of 85% for Road Work for the 3rd sign, while only a 5% certainty for the correct sign, Go Straight or Left.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|First Sign ||
| .99         			| *Ahead Only* 									|
| <.01     				| Speed Limit (60km/h) 										|
| <.01					| Wild Animals Crossing											|
| <.01	      			| Turn Right Ahead					 				|
| <.01				    | Right-of-Way Next Intersection      							|
|Second Sign||
| .99         			| *Turn Right Ahead*  									|
| <.01     				| Traffic Signals 										|
| <.01					| Go Straight or Left											|
| <.01	      			| Roundabout Mandatory					 				|
| <.01				    | Pedestrians      							|
|Third Sign||
| .85         			| Road Work   									|
| .09     				| Dangerous Curve to the Left 										|
| .05					| *Go Straight or Left*											|
| <.01	      			| Right-of-Way Next Intersection					 				|
| <.01				    | Keep Right      							|
|Fourth Sign||
| .99         			| *Right-of-Way at Next Intersection*   									|
| <.01     				| Double Curve									|
| <.01					|	Beware of Ice/Snow										|
| <.01	      			| Slippery Road					 				|
| <.01				    | Pedestrians      							|
|Fifth Sign||
| .99        			| *Speed Limit (30km/h)*   									|
| <.01    				| Speed Limit (50km/h) 										|
| <.01					| Speed Limit (70km/h)								|
| <.01	      			| Speed Limit (20km/h)					 				|
| <.01				    | Keep Left      							|
