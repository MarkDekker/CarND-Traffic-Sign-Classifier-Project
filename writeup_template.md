# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./writeup_assets/processed_image_smaple.png "Processing"
[image3]: ./writeup_assets/augmented_signs.png "Augmented Signs"
[image4]: ./writeup_assets/confusion_matrix.png "Confusion matrix for test data."
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
## **Reflection**

### Data Set Summary & Exploration

To start off, I got a feel for the data set by exploring its dimensions and having a look at some sample data.

The size of the data set is modest at __34799 training images__, with a further __12630 images for testing__ and __4410 images for validation__.

The data was already pre-processed into an easy-to-handle format as a numpy array of 32x32 pixel images with 3 colour channels.

In total there are 43 different sign classes, each of which is represented in the training, validation and test sets respectively.

Below is a sample of the images contained in the data set:

![alt text][image5]

#### 2. Include an exploratory visualization of the dataset.

In the graph below one can see the distribution of the images per class for each of the respective data sets:

![Bar chart of image distribution per class][image1]

From this, we can conclude that the data set is highly unbalanced with some classes being represented by as few as 200 signs, while others contain in excess of 2000 signs.  With such a large imbalance, there is a risk that the model will be heavily weighted towards identifying the classes that contain more training data.

### Design and Test a Model Architecture

#### Image pre-porcessing

1. According to [LeCun and Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) working in grayscale reduces the processing requirements and does not appear to impact classification significantly. Intuitively this seems strange though, since the colour information clearly carries information that aids in distinguishing between classes. (Take the priority road, stop and keep right signs which are yellow, red and blue respectively)

   For this reason, the number of colour channels is defined by a parameter in my solution, to be able to switch between colour and grayscale easily.

2. The next step was to scale the pixel data to have zero mean and unit norm for the entire image.  The tensor flow function `tf.image.per_image_standardization()` was used for this. The normalising is particularly useful for this data set, extracting a lot of additional detail from the images.

   Below are some examples of the conversion:

  ![Result after image preprocessing][image2]

Given that the data set is so imbalanced and it is only modestly sized, I created a subroutine to augment the data set. The function `augment_image_data(images_in, examples_required)` accepts the image set to be augmented and the total number of examples required for this image set. It then stretches, rotates, blurs and changes the brightness on the images with a randomised intensity until it reaches the required number of examples.  

For this, the training set was separated into bins of images of each respective class. These are then fed into the function and finally combined and reshuffled again.

It was decided to balance the image sets so that each contains at least 1000 images. 

Here are some examples of augmented images:

![Augmented Signs][image3]

The augmented data set now has a size of 61799 compared to 34799 for the original training data set.

#### Final Model Architecture

My final model consisted of the following layers:

| Layer         		    |     Description	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x1 RGB image   							            | 
|						            |												                        |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x16 				        |
|						            |												                        |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x32 				          |
|						            |												                        |
| Flatten	              | Flattens ConvNet outputs to 800x1     				|
| Fully connected		    | outputs 240        									          |
| RELU				          |         									                    |
|	Dropout               |	50% dropout probability during training       |
|						            |												                        |
| Flatten	              | Flattens ConvNet outputs to 800x1     				|
| Fully connected		    | outputs 240        									          |
| RELU				          |         									                    |
|	Dropout               |	50% dropout probability during training       |
|						            |												                        |
| Fully connected		    | outputs 84        									          |
| RELU				          |         									                    |
|	Dropout               |	50% dropout probability during training       |
|						            |												                        |
| Fully connected		    | outputs 43 to match number of classes         |


#### Training the model

A variety of techniques were used to improve the performance and accuracy when training the model.  The focus here was primarily on the hyperparameters.

- The batch size does not appear to influence the model that significantly, although the best values were obtained for __128 samples per batch__.
- Similarly, using the colour information of the images did not appear to improve the result.
- Increasing the learning rate at 0.01 leads to divergence of the model, while a value around 0.001 works reasonably well. However, it is commonly found that once the neural net has been trained to within the vicinity of an optimum, the learning rate should be decreased to improve the result further. Furthermore, [Loshchilov & Hutter](https://arxiv.org/abs/1608.03983) found that periodically raising the learning rate back to a higher level leads to more robust minima, which should also mean more generalisable models. This is why I opted to use stochastic gradient descent (SGDR) with restarts as described in their paper (tensor flow function `tf.train.cosine_decay_restarts()`).



 - Hyperparameters


#### Results and Training

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.981
* test set accuracy of 0.962

The network architecture is close to unchanged from LeNet-5 with more filters in the convolutional layers as well as more neurons in the first fully connected layer. Finally dropout was also added to the third and fourth fully connected layers for improved training performance.

This architecture was chosen because it is simple and small compared to more modern networks such as AlexNet or ResNet50 so it seemed like a good starting point to get a feel for the challenge.

To begin with, the LeNet-5 model was used as-is, without dropout or increasing the size of the hidden layers. The hyperparameters were tuned to see what the maximum performance of this model could be.

- The batch size does not appear to influence the model that significantly, although the best values were obtained for __128 samples per batch__.
- Similarly, using the colour information of the images did not appear to improve the result.
- Increasing the learning rate at 0.01 leads to divergence of the model, while a value around 0.001 works reasonably well. However, it is commonly found that once the neural net has been trained to within the vicinity of an optimum, the learning rate should be decreased to improve the result further. Furthermore, [Loshchilov & Hutter](https://arxiv.org/abs/1608.03983) found that periodically raising the learning rate back to a higher level leads to more robust minima, which should also mean more generalisable models. This is why I opted to use stochastic gradient descent (SGDR) with restarts as described in their paper (tensor flow function `tf.train.cosine_decay_restarts()`).
- To get the best performance, it was found that the learning rate should be kept high for the first 20 epochs after which decreasing the learning rate with SGDR results in the fine tuning required when it is needed.

This approach already yielded accuracy well above 0.9 for the validation set. After adding dropout to the fully connected layers, this already exceeded 0.95. This is because adding dropout reduces the tendency of the model to overfit the data since more neurons are forced to contribute to a successful classification when others are 'switched off'.

Finally, the size of the hidden layers was increased, increasing the complexity that the model can capture.  This ultimately yielded the final accuracy scores.

It was surprising to find that using the augmented test data resulted in significantly worse performance during training.  This could be because the distortions are too severe, which means that we are adding noise not data.  Nevertheless all the samples looked at from the augmentation set were still recognisable albeit cropped or severely blurred in some cases. This means that this accuracy was achieved without using the augmented data set.

I believe that while the model appears to work well as is, there is still potential for future improvements. On a large data set, misclassifying 4% of images leads to many mistakes.  A rate that would not be acceptable in an operational Autonomous vehicle. This is especially apparent when one looks at the confusion matrix for the results on the test set:

![Confusion Matrix for test results][image4] 

From this we see that there are clearly many misclassifications among the speed limits, which is understandable since these look so similar but it could also have disastrous consequences.

What is especially surprising is the relatively low accuracy for labels with many training images such as the "80km/h Speed Limit" when compared to other signs with far fewer associated images such as "End of all Speed and Passing Limits".

If we have a closer look at the misclassifications in each individual set, we gain more information on what is going wrong but also about the data in particular.





 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


