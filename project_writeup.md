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

[image1]: ./writeup_assets/raw_image_example.png "Raw image sample"
[image2]: ./writeup_assets/imge_distributions.png "Raw image sample"
[image3]: ./writeup_assets/processed_image_smaple.png "Processed image sample"
[image4]: ./writeup_assets/augmented_signs.png "Augmented Signs"
[image5]: ./writeup_assets/confusion_matrix.png "Confusion matrix for test data."
[image6]: ./writeup_assets/misclassified_priority_road.png "Misclassified 1"
[image7]: ./writeup_assets/misclassified_100kmh.png "Misclassified 2"
[image8]: ./writeup_assets/misclassified_20kmh.png "Misclassified 3"
[image9]: ./writeup_assets/misclassified_30kmh.png "Misclassified 4"
[image10]: ./writeup_assets/misclassified_traffic_signs.png "Misclassified 5"
[image11]: ./writeup_assets/misclassified_general_caution.png "Misclassified 6"
[image12]: ./writeup_assets/downloaded_images.png "Downloaded Images"
[image13]: ./writeup_assets/double_curve.png "Double Curve Examples"
[image14]: ./writeup_assets/softmax_double_curve.png "Double Curve Softmax"
[image15]: ./writeup_assets/softmax_right_turn_ahead.png "Turn Right Softmax"
[image16]: ./writeup_assets/softmax_no_entry.png "No Entry Softmax"
[image17]: ./writeup_assets/softmax_30kmh_2.png "No Entry Softmax"




---
## **Reflection**

### Data Set Summary & Exploration

To start off, I got a feel for the data set by exploring its dimensions and having a look at some sample data.

The size of the data set is modest at __34799 training images__, with a further __12630 images for testing__ and __4410 images for validation__.

The data was already pre-processed into an easy-to-handle format as a numpy array of 32x32 pixel images with 3 colour channels.

In total, there are 43 different sign classes, each of which is represented in the training, validation and test sets respectively.

Below is a sample of the images contained in the data set:

![alt text][image1]

### Exploratory visualization of the data set

In the graph below one can see the distribution of the images per class for each of the respective data sets:

![Bar chart of image distribution per class][image2]

From this, we can conclude that the data set is highly unbalanced with some classes being represented by as few as 200 signs, while others contain in excess of 2000 signs.  With such a large imbalance, there is a risk that the model will be heavily weighted towards identifying the classes that contain more training data.

## **Design and Test a Model Architecture**

### Image pre-processing

1. According to [LeCun and Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) working in grayscale reduces the processing requirements and does not appear to impact classification significantly. Intuitively this seems strange though, since the colour information clearly carries information that aids in distinguishing between classes. (Take the priority road, stop and keep right signs which are yellow, red and blue respectively)

   For this reason, the number of colour channels is defined by a parameter in my solution, to be able to switch between colour and grayscale easily.

2. The next step was to scale the pixel data to have zero mean and unit norm for each image.  The tensor flow function `tf.image.per_image_standardization()` was used for this. The normalising is particularly useful for this data set, extracting a lot of additional detail from the images.

   Below are some examples of the conversion:

  ![Result after image preprocessing][image3]

Given that the data set is so imbalanced and it is only modestly sized, I created a subroutine to augment the data set. The function `augment_image_data(images_in, examples_required)` accepts the image set to be augmented and the total number of examples required for this image set. It then stretches, rotates, blurs and changes the brightness on the images with a randomised intensity until it reaches the required number of examples.  

For this, the training set was separated into bins of images of each respective class. These are then fed into the function and finally combined and reshuffled again, after they have been augmented.

It was decided to balance the image sets so that each contains at least 1000 images. 

Here are some examples of augmented images:

![Augmented Signs][image4]

The augmented data set now has a size of 61799 compared to 34799 for the original training data set.

### Final Model Architecture

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
| Fully connected		    | outputs 240        									          |
| RELU				          |         									                    |
|	Dropout               |	50% dropout probability during training       |
|						            |												                        |
| Fully connected		    | outputs 84        									          |
| RELU				          |         									                    |
|	Dropout               |	50% dropout probability during training       |
|						            |												                        |
| Fully connected		    | outputs 43 to match number of classes         |


### Results and Training

My final model results were:
* Training set accuracy of 1.00
* Validation set accuracy of 0.981
* Test set accuracy of 0.962

The network architecture is close to unchanged from LeNet-5 with more filters in the convolutional layers as well as more neurons in the first fully connected layer. Finally dropout was also added to the third and fourth fully connected layers for improved training performance.

This architecture was chosen because it is simple and small compared to more modern networks such as AlexNet or ResNet50 so it seemed like a good starting point to get a feel for the challenge.

To begin with, the LeNet-5 model was used as-is, without dropout or increasing the size of the hidden layers. The hyperparameters were tuned to see what the maximum performance of this model could be.

- The batch size does not appear to influence the model that significantly, although the best values were obtained for __128 samples per batch__.
- Similarly, using the colour information of the images did not appear to improve the result.
- Increasing the learning rate to 0.01 leads to divergence of the model, while a value around 0.001 results in faster convergence without ultimately leading to divergence. However, it is commonly found that once the neural net has been trained to within the vicinity of an optimum, the learning rate should be decreased to improve the result further. Furthermore, [Loshchilov & Hutter](https://arxiv.org/abs/1608.03983) found that periodically raising the learning rate back to a higher level leads to more robust minima, which should also mean more generalisable models. This is why I opted to use stochastic gradient descent (SGDR) with restarts as described in their paper (tensor flow function `tf.train.cosine_decay_restarts()`).
- To get the best performance, it was found that the learning rate should be kept high for the first 20 epochs after which decreasing the learning rate with SGDR results in the fine tuning required when it is needed.

The initial results were not in line with what was expected, in many cases not even reaching an accuracy above 0.6. After playing with all conceivable hyperparameters, it was finally found that the initialisation of the model weights plays a crucial role. Changing the standard deviation to 0.05 had pronounced effects, yielding accuracy well above 0.9 for the validation set. 

After adding dropout to the fully connected layers, the accuracy exceeded 0.95. This is because adding dropout reduces the tendency of the model to overfit the data since more neurons are forced to contribute to a successful classification when others are 'switched off'.

Finally, the size of the hidden layers was increased, increasing the complexity that the model can capture.  This ultimately yielded the final accuracy scores.

It was surprising to find that using the augmented test data resulted in significantly worse performance during training.  This could be because the distortions are too severe, which means that we are adding noise as opposed to more data.  Nevertheless, all the samples looked at from the augmentation set were still recognisable albeit cropped or severely blurred in some cases. This means that all results described here were obtained without using the augmented data set.

I believe that while the model appears to work well as is, there is still potential for future improvements. On a large data set, misclassifying 4% of images leads to many mistakes.  A rate that would not be acceptable in an operational autonomous vehicle. This is especially apparent when one looks at the confusion matrix for the results on the test set:

![Confusion Matrix for test results][image5] 

From this we see that there are clearly many misclassifications among the speed limits, which is understandable since these look so similar but it could also have disastrous consequences.

What is especially surprising is the relatively low accuracy for labels with many training images such as the "80km/h Speed Limit" when compared to other signs with far fewer associated images such as "End of all Speed and Passing Limits".

If we have a closer look at the misclassifications in each individual set, we gain more information on what is going wrong but also about the data in particular.

The following images of the class "Roundabout Mandatory" were misclassified as follows:

__Priority Road__

![Misclassification - Roundabout Mandatory as Priority Road][image6] 

__100km/h Speed Limit__

![Misclassification - Roundabout Mandatory as 100km/h Speed limit][image7] 

From these examples it becomes clear that the "raw data" was already augmented and given that these images look so similar, the model effectively only misclassified one image which was repeated in the data. The flip side of this is that classifying a single image correctly that has multiple slightly augmented copies in the data set would artificially raise the accuracy.

It is nevertheless surprising that this image was misclassified since the shape and details of the sign are clearly visible.


| 20km/h Speed Limit    |  30km/h Speed Limit	|   Traffic Signals	|  General Caution	|
|:---------------------:|:-------------------:|:-----------------:|:--------------:| 
| ![Mis][image8]    |  ![Mis][image9]	|   ![Mis][image10]	|  ![Mis][image11]	|

The reason for the misclassification of these remaining images is clearer, given that they are even hardly recognisable to humans due to the distortions.


## **Testing the Model on New Images**

Here are 12 German traffic signs that I found on the web:

![''][image12]

Some images in the set were specifically chosen because they might be difficult to classify. The "Bumpy Road" sign is partially covered by a bikini, one of the "No Entry" signs is at a very oblique angle with difficult lighting, the "End of No Passing" sign is partially covered and surrounded by other signs that might confuse the classifier and finally the "Turn Right Ahead" sign is covered in stickers.

### Model Predictions

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:---------------------:|:---------------------------------:| 
| 30 km/h Speed Limit   | 30 km/h Speed Limit  							| 
| 30 km/h Speed Limit   | 30 km/h Speed Limit								|
| Bumpy Road      		  | Traffic Signals										|
| No Entry	            | Yield					 				            |
| No Entry              | No Entry      							      |
| Double Curve          | Dangerous Curve to the Right 			|
| End of No Passing     | End of No Passing  							  |
| General Caution       | General Caution      							|
| Priority Road         | Priority Road       							|
| Road Works            | Road Works          							|
| Turn Right Ahead      | Turn Left or Go Straight     			|
| Wild Animals Crossing | Wild Animals Crossing     				|


The model was able to correctly guess 8 of the 12 traffic signs, which gives an accuracy of 67%. This is below the level achieved for the validation and test sets.  Some of the images such as the "Bumpy Road", "Turn Right Ahead" and "No Entry" sign were relatively challenging examples however. What is surprising is that the "Double Curve" sign was misclassified, since this is as clear an example of such a sign as one could expect.

Upon closer inspection of the training data however, the reason for the misclassification of the double curve sign also becomes apparent. The training data does look different than this example:

![''][image13]

### A closer look at the classifications

For most signs, the model was close to 100% "confident" in its classification, even if it was incorrect.  The following three signs showed some more ambiguity though:

![''][image14]

![''][image15]

![''][image16]

As would be expected, the signs that deviate more from the training data result in lower confidence levels.  It is surprising and problematic that the other most likely classes are also incorrect, which means the model is not general enough to be reliable when used on new data.

## Potential For Improvements

While on paper the model appears to work well, a closer look reveals that it still misclassifies easily identifiable signs.  It was also found that the performance on new data is not as robust as would be desired.

While it has consistently been found that using colour information for training the models did not help significantly, this is highly counter intuitive.  It may be that the architecture of the neural network is something that is preventing it from extracting the relevant information from the colour channels.  Something that may be interesting here could be parallel convolutions for each colour channel, or a parallel fully connected layer that receives the image as an input.

It is expected that further improvements can be achieved with more careful and meticulous data augmentation. Examples of this include flipping images that are side agnostic, such as "Yield" and "Do Not Enter", using flipped signs containing arrows (keep right, right turn ahead ...) to augment the complimentary sign data sets (keep left, left turn ahead ...), using rotations and stretching judiciously and adding other distortions such as lens flares or obstructions.

Even with all these augmentations, it is important to understand the nature of the data set well. For this a closer inspection of a large part of the data set could help greatly.  Based on the results above it could be that the data is already substantially augmented, which would mean that the raw data set is even smaller than expected, which could mean that it is not suitable for training a sufficiently robust model.


