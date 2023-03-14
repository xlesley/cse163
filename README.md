# CSE163 Final Project
**Lesley Xu, Hutiancong Wang, Sabrina Lin**

## The dataset
https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

## The project
In our project, we utilize the built-in dataset in tensorflow.keras.layers called `cifar-10` for classification task. We built our model using alexnet, and test our results by visualizing the loss, accuracy during the training, the predicted images. 

## The Jupytor Notebook
https://colab.research.google.com/drive/1ZoIDYsiNWJVMn8_TU2qlYLKMPRyYYhDt?usp=sharing

Above is a jupytor notebook version of our project, with all the output saved and printed. 


## The script
Please refer to the `cifar10_classifier.py` for the replication of our experiment. We are using batch size = 64, and epoch = 50 for our model training. The typical time of running 50 epochs would be around 10 minutes. `cifar10_classifier.py` is an integrated version of our code for the project. 

The `preprocessing.py` module implements the functions that preprocess the data with necessary steps before training. 

The `train_predict.py` module implements the definition of model, and the stakeholders may use it to test different outputs by adjusting the prameters inside the `alexnet()` function, as well as the batch size, epoch size. 

The necessary libraries for each module are listed at the top of the file, please refer to that to install the libraries for model training as well as train history, and prediction visualization. 
