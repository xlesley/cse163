# Using Deep Learning for Image Classification
**CSE 163 Project Final Report - Winter 2023**

(Albert) Hutiancong Wang, Lesley Xu, Sabrina Lin

Disclaimer: Find the most updated and formatted report in our gradescope submission (PDF format).

## Summary of Research Questions
1. How can we process images and extract content out of them?

- We investigated how to read images through the algorithm in order to get meaningful data out of it.
  - Answer: We found that we can process images and extract content out of them using the `tensorflow` and `numpy` library.

2. How can we predict the content of an image and translate them into words via an image classifier using deep learning?
- We investigated the tensorflow deep learning algorithm, which is very new compared to the introductory machine learning concepts we learned during class. We wanted to adopt new, advanced techniques to find predictions on the label.
  - Answer: We found that we can predict the content of an image and translate them into words by comparing the actual index in the category to the predicted index in the category, such that “true = [3], prediction = [7]”.

3. When applying the tensorflow deep learning model, how accurately can it predict objects in various different types of images, from an airplane to a frog? Are there certain types of objects the algorithm can predict more accurately? 
- We investigated the complex tensorflow deep learning library and compared accuracy scores of each class in the dataset to find which objects were more or less likely to be predicted correctly from the algorithm. We compared the accuracy scores of each object. 
  - Answer: We found that among the first 20 predictions, the ship was predicted accurately and all other categories were predicted inaccurately.  

4. What is the influence of batch size and epoch toward the performance of the trained model?
- Adjusting the batch size
  - Answer: We used 1, 32, 64, 500, and found out that the smaller the batch size is, the longer it takes to process for each epoch. However, does the smaller batch size mean higher accuracy? Not necessarily. During the experiment we found out that when batch size = 32 or batch size = 64 would be the optimal value.
- Adjusting the epoch number 
  - Answer: The higher the number of epochs is, the more times of training the model experienced, and higher accuracy accordingly.
- Compare the plot of accuracy and plot accordingly 
  - Answer: Still a low accuracy rate among the first 20 predictions even within only 10 categories.


## Motivation
According to Georgetown University and visionaustralia.org, 20 million people in the US have visual impairments and 39 million people across the world are blind (source 1, source 2). 23.2% of 1 million website homepages had missing alt text in 2022, making it inaccessible for people with visual disabilities (source). 

Our main motivation for pursuing this topic was because of the barriers that people with visual disabilities endure during their daily lives. People with disabilities are not able to enjoy the world the same way people without disabilities do. 

Knowing the answers to our research questions can provide a lot of insight into how to make images more accessible to people who are visually-impaired via image classification in automatic alternative text, and can open doors to more efficient ways to interpret images, enhancing the experience of those with visual disabilities. Image classification via deep learning is the gateway to more advanced techniques for automatic alternative text, which describes images to users who can’t see, and is an incredibly useful tool for accommodation. The auto-alternative text feature also provides a more effective way of automatically tagging the photos with captions, making it easier for social media users to search and get more precise results of what they want. When designing experiences inclusively, people with and without disabilities can benefit, and it significantly enhances the lives of people with disabilities.


## Dataset
We are using the following dataset for our classification: CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here are the following 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The reason we changed our dataset for the classification task from the Kaggle Caltech 256 categories dataset to the CIFAR-10 dataset is because the original 256 categories exceeds the capabilities of google colab. With google colab, the runtime is extremely long and it is hard to get a satisfying result. 


## Method
Below is a general overview of our method for this project. For a more comprehensive understanding, refer to the docstrings in the code. 

### Set up 
Addresses RQ #1
1. Load the image dataset using libraries such as `numpy`, convert them into the same sizes. Because there is not enough data in the training set,  we subset the data for validation from the test set. We also converted the data from `unint` type to `float32` type to increase the precision of the training process. 
2. Clean the dataset by normalizing the images in the dataset into numpy arrays. 
3. Divide the array by 255 to get a new pixel value between 0 and 1, and therefore increase the accuracy and precision of model training. 
4. After splitting the train set, the validation set, and the test set, convert the 10 labels for the image data set to one-hot vectors. (y_train, y_val, y_test)
5. Convert to vectors and transfer into the neural network. 
### Discovery process
Prepares us to answer RQ #2, 3, 4
Do research on deep learning and deep convolutional neural networks and choose which library will be the most challenging and uncover the most insight into our problem space.
Input dataset and train the models with different batch sizes and epoch numbers 
Addresses RQ #2 and RQ #4
After defining the alexnet model in a function by adding layers, input the cleaned training dataset into the model. The resources we found had a large kernel size and could not train the model in a limited amount of time. Therefore, we scaled the kernel to a smaller size. 
Compile the network, define loss function, optimizer, and accuracy as evaluation of the model. 
Start training the model by defining the train set and validation set, defining the batch size. Train the model over time and make improvements. 
### Test the model and look at accuracy scores for each object 
Addresses RQ #3
Test the set based on the tensorflow deep learning model we created. We evaluate the accuracy of our model by creating a visualization for the training history. 
Create a function to compare the loss difference between train set and validation set, and the accuracy difference between train set and validation set by plotting line plots over each epoch. 
As a supplemental method, we use the model to predict classification for the first 20 images. We visualize the predicted image. Add the true index label and the predicted index label, and compare the true label and the predicted label. 
### Evaluate the accuracy score of each object from the tensorflow deep learning model 
Addresses RQ #3 and RQ #4
Compare the output using accuracy score to find the appropriate one to use for the purpose of image captioning
Give feedback and suggestions
Recognize what has been achieved and what has not


## Results
After plotting losses and accuracy scores, we found that with 50 epochs, we finally stabilize our results to make losses around 1.22, accuracy 0.57, validation losses 1.39, and validation accuracy 0.51. As the number of epoch increases, the train accuracy increases, while the train loss decreases. 

The train accuracy has shown a mid-to-up level train accuracy for our model as 0.51. When it applied to the actual processes of image recognition, the test accuracy is only measured as 10% which could be a problem potentially involving overfitting, although we intently used `dropout` for regularization.

When looking at the predictions for the first 20 images with the alexnet model we made, only one out of 20 is correctly predicted (the ship), as seen in the image above.

We found that we can predict the content of an image and translate them into words by comparing the actual index in the category to the predicted index in the category, such that “true = [3], prediction = [7]”. In this example, referring to the list of categories, the actual image should depict a cat but the model predicts a horse, meaning the general classification is correct (detecting an animal rather than a mode of transport). This incorrect prediction might indicate issues associated with mislabelling or the layers within the model architecture since they filtered out certain conditions in the process of image recognition.

CNNs (Convolutional Neural Networks) are much more complicated than we had previously thought, and it needs extraordinary efforts to normalize and regularize the dataset inputs, as well as model parameters. The model complexity leads to problems like overfitting and declining accuracy, but we believe by improving on batch normalization and early stopping could greatly reduce the possibilities of having those problems. Referring to the file `CIFAR10 Classification` in our Github site, we have seen great changes by using different epochs, scales of training set, losses and accuracies.

In practice, with RQ #4, we found that increasing epochs and batch size would improve model performance. However, it takes more memory while processing. 

After implementing normalization, we surprisingly found that we improved training accuracy from 51% to 53% and ultimately have 55% test accuracy on a subset dataset of 20 pictures. Augmentation and other conditions to help the running processes and accuracies are also added. Results can be referred to below.


## Impact and Limitations
From our results, people who are visually-impaired, social media sites, and organizations with websites will benefit from our analysis. This is because people who are visually-impaired will be able to experience more inclusive web interface designs and social media posts by having their screen readers read the automated alt text created by our deep learning algorithm. Additionally, social media sites and organizations with websites will be able to increase their outreach and bring in revenue because they’ll be accommodating for people with visual impairments by utilizing automated alt text. It will also help employees and users avoid the burden of manually typing out alt text because of the automated predictions from the deep learning algorithm. Furthermore, companies will be able to see which objects our algorithm can accurately predict more than others by comparing the accuracy scores of each object, and make the necessary precautions (ex: suggesting users/employees to manually type out alt text for images if the algorithm doesn’t identify that specific image very well).

While visually-impaired people can benefit from this analysis, there is a possibility they may be harmed. Our deep learning algorithm may not predict an image correctly, which will lead visually-impaired users to a false interpretation of what they’re looking at. People who experience language barriers will also be harmed, because they are excluded from the analysis and thus will not be able to use this tool. The objects in the images will only be identified in English since different languages are not within the scope of this project.

Because there are millions of different objects in the world and we only looked at 10, our algorithm will not be able to predict millions of objects in images, thus making it more biased towards being able to predict certain images. Additionally, all of the object names were in English, so non-English speakers and people who experience language barriers will not be able to utilize our algorithm for automated alt text capabilities unless we expand it to work in multiple languages. People who experience language barriers should not use this algorithm unless they are comfortable with English reading comprehension.

However, the neural networks at a big-scale like this would require a strong machine to implement if using more epochs, larger-scale datasets, and/or other improving measures. To implement this plan would require huge servers and hardware from big tech companies, but companies with large monopolies would most likely only be money-motivated; if this solution doesn’t bring in enough money, it will likely not be implemented, regardless of the improvements in social equity.


## Challenge Goals
We dove deeper into the machine learning challenge goal for this project. We utilized machine learning techniques including what we learned in class (`Numpy` packages to utilize machine learning techniques) and even more advanced skills including creating a deep learning model (`tensorflow`) and graphing the deep learning algorithm’s accuracy scores and loss during the training process (`Plotly`). The algorithm might be based with reference to the representation learning process. Ultimately, the system uses deep learning algorithms to classify images based on their attributes.

We narrowed our scope from the list of possible new libraries from the proposal according to our research questions and dataset.

We imported the following from new libraries:
- import tensorflow as tf
- `Tensorflow` is a Python extension that is particularly used for Deep/Reinforcement Learning, constructing Neural Networks by processing large datasets and doing training & testing processes.
- from keras.utils import np_utils
- `keras.utils` provides a series of utility functions, and `np_utils` in particular helps functions dealing with arrays, here we used `to_categorical` to convert integers to metrics.
- from keras.models import Sequential
- `keras.models` contains useful high level API and training models. `Sequential` can create linear stacks of layers to build neural network models.
- import plotly.graph_objects as go
- `plotly.graph_objects` provides data visualizations and makes us capable of customizing our graphs to show the fluctuations and changes.


## Work Plan Evaluation 
Our proposed work plan was not very accurate due to the overestimates in timing and our revised project scope. We overestimated times because we predicted that we would work on the project code for three hours each day until the deadline. After starting to work on the code and consulting with our TA mentor, we realized this did not make much sense and wasn’t very feasible considering other commitments we have outside of this class and the length of time it took us to complete the take home assessments. Additionally, we didn’t begin working on the code until after our take home assessments were turned in and after we were introduced to the basics of images and machine learning in class, which set us back from our original work plan. If you’re interested in seeing our original work plan, please refer to our proposal submission.

**Below is the actual work plan we followed:** 
Overall, we worked on the project for 13 days (03/01/23 - 03/13/23. We spent most of our time setting up data and developing training models. The following is how we divided the project into discrete tasks over 13 days:

3/1 - 3/5 (6 hrs) → Understand the basics of machine learning related to images, read up on deep learning techniques, and discover which libraries we want to use
3/6 - 3/8 (5 hrs) → Set up, data loading
3/9 - 3/13 (12 hrs) → Input dataset, create tensorflow deep learning model, train the model
3/13 (1 hr) → Test model and calculate the accuracy_score for each category
3/13 (1 hr) → Plot the accuracy scores from Test model
3/13 (1 hr) → Analyze trends and key takeaways from our plots

We developed and tested our code asynchronously (on our own time) and synchronously (miscellaneous zoom and discord meetings as a team and with TA’s). We coordinated work over text message group chat. We developed python code on Google CoLaboratory because it was the most feasible considering how large our dataset was. It is also a good place to collaborate, as the code cells update in real time. We also utilized a shared GitHub repository to house our data files and push/pull the files into our local computers when needed.


## Testing 
To test our deep learning model, we went with a roughly 80/20 split where 80% of the dataset was put towards training and 20% of the dataset was put towards testing. This train test split method is a model validation process that allows us to simulate how our model would perform with new data rather than data that it has already looked at (source). Test accuracy is a much better evaluation of how the model will do in the future (compared to train accuracy) because it’s testing on data we haven’t tested before. It’s incredibly important to test on new data because the train accuracy will most likely be very high and not a very good representation of how the model will actually perform out in the real world with new data. 

In practice, we selected 20 random pictures and tested them using a training model which trains for 50 epoch times. It is totally feasible to increase the epoch time to repeatedly go through the dataset in order to improve accuracy. This reminds us that in big tech companies like Microsoft and Google, they always have huge-scale, powerful machines and strong, stable services to support their operations as well as updates. The attempts in our codes to reduce overfitting issues include using `dropout` and `validation set` are extraordinary but still leave spaces to be improved on.


## Collaboration 
With the guidance of our TA mentor, we consulted a variety of different resources since this project focused on a concept we have not learned in class. 

We utilized the following sources to write, debug, and understand our code:
- https://www.tensorflow.org/tutorials/images/cnn#download_and_prepare_the_cifar10_dataset
- https://www.tensorflow.org/api_docs/python/tf/keras/layers
- https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
- https://www.kaggle.com/code/vortexkol/alexnet-cnn-architecture-on-tensorflow-beginner
- https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98

We used the following references to write and support our report:
*Stats on visual impairments to inform our problem space sand the severity of the problem*
- https://hpi.georgetown.edu/visual/#:~:text=Almost%2020%20million%20Americans%20%E2%80%94%208,U.S.%20population%20%E2%80%94%20have%20visual%20impairments. 
- https://visionaustralia.org/news/2019-08-23/global-facts-blindness-and-low-vision#:~:text=According%20to%20WHO%20estimates%3A,million%20people%20have%20low%20vision 
- https://www.synaptiq.ai/library/the-pros-and-cons-of-artificial-intelligence-for-alt-text-generation?hsLang=en#:~:text=A%202022%20investigation%20by%20the,percent%20had%20missing%20alt%20text. 
*Importance of train test split*
- https://builtin.com/data-science/train-test--split#:~:text=Train%20test%20split%20is%20a%20model%20validation%20process%20that%20allows,would%20perform%20with%20new%20data. 

