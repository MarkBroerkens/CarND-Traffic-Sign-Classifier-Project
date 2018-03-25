**Abstract**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar_chart_training_set]: ./figures/bar_chart_training_set.png "Distribution of training samples per label"
[labels_with_examples]: ./figures/labels_with_examples.png "Labels and example images"
[grayscale]: ./figures/grayscale.jpg "Grayscaling"
[traffic_signs_orig]: ./figures/traffic_signs_orig.png "Traffic Signs"
[traffic_signs_prediction]: ./figures/traffic_signs_prediction.png "Traffic Signs Prediction"
[learning]: ./figures/learning.png "Validation Accuracy per Epoche"
[prediction_probabilities_with_examples]: ./figures/prediction_probabilities_with_examples.png "Traffic Sign Prediction with Examples"
[prediction_probabilities_with_barcharts]: ./figures/prediction_probabilities_with_barcharts.png "Traffic Sign Prediction with Bar Charts"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Submitted Files

 1. README that includes all the rubric points and how you addressed each one. You're reading it!
 1. The [jupyter notebook](https://github.com/MarkBroerkens/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).
 1. [HTML output of the code](Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The following figure shows one example image for each label in the training set.

![alt text][labels_with_examples]

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples are contained in the training set per label.

![alt text][bar_chart_training_set]


### Design and Test a Model Architecture

#### 1 Preprocessing

As a first step, I decided to convert the images to grayscale because several images in the training were pretty dark and contained only little color und the grayscaling reduces the amount of features and thus reduces execution time. Additionally, several research papers have shown good results with grayscaling of the images. [Yann LeCun - Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

Then, I normalized the image using the formular `(pixel - 128)/ 128` which converts the int values of each pixel which range [0,255] to float values with range [-1,1]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer                  |     Description                                |
|:----------------------:|:----------------------------------------------:|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| Flatten                | outputs 400                                    |
| **Dropout**            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 53                                     |
| Softmax                |                                                |


To train the model, I used an Adam optimizer and the following hyperparameters:
* batch size: 128
* number of epochs: 150
* learning rate: 0.0005
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* keep probalbility of the dropout layer: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 97.5%
* test set accuracy of 95.1%

I used an iterative approach for the optimization of validation accuracy:
1. As an initial model architecture the original LeNet model from the course was chosen. In order to fit the new requirements I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and modified the number of output so that it fits to the number of unique labels in the training set. The training accuracy was **83.5%** and my pedestrian sign was not correctly classified. (used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1) 

1. preprocessing grayscale. The validation accuracy increased to **91%** (hyperparameter unmodified)

1. additional preprocessing normalization. Minor increase of validation accuracy: **91.8%** (hyperparameter unmodified)

1. reduced learning rate and increased number of epochs. validation accuracy = **94%** (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. overfitting. added dropout layer after relu of final fully connected layer: validation accuracy = **94,7%** (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. still overfitting. added dropout after relu of first fully connected layer. Overfitting reduced but still not good

1. added dropout before validation accuracy = 0.953 validation accuracy = **95,3%** ((EPOCHS = 50, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1)

1. further reduction of learning rate and increase of epochs. validation accuracy = **97,5%** ((EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0005, mu = 0, sigma = 0.1)

![alt text][learning]

Summary


### Test a Model on New Images

Here are five German traffic signs that I found on the web:
![alt text][traffic_signs_orig]

The first image might be difficult to classify because ...

Here are the results of the prediction:

![alt text][traffic_signs_prediction]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.1%

The code for making predictions on my final model is located in the 21th cell of the [Ipython notebook](https://github.com/MarkBroerkens/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

![alt text][prediction_probabilities_with_barcharts]

![alt text][prediction_probabilities_with_examples]
