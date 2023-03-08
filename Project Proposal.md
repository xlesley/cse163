# Descriptive Phrases for An Image

CSE 163 - Winter 2023

Albert Wang, Lesley Xu, Sabrina Lin

## **Summary of Research Questions**

1. How can we process images and extract content out of them?

2. 1. We will investigate how to read images through the algorithm in order to get meaningful data out of it.

3. How can we predict the content of an image and translate them into words and phrases?

4. 1. We will investigate machine learning algorithms and try to adopt new, advanced techniques to do predictions on the label.

5. When applying different machine learning models, there might be different styles of description. Evaluating from the potential stakeholders’ perspective, which model might provide the most accurate, precise information?

6. 1. We will investigate multiple different python machine learning libraries and compare the accuracy scores of each method with the same training sets. Evaluate the prediction accuracy by checking the accurate score or some other values. 

## **Motivation**

People with disabilities are not able to enjoy the world the same way people without disabilities do. Knowing the answers to our research questions can provide a lot of insight into how to make images more accessible to people who are visually-impaired via automatic alternative text, and can open doors to more efficient ways to interpret images, enhancing the experience of those with visual disabilities. Automatic alternative text describes images to users who can’t see, and is an incredibly useful tool for accommodation. The feature also provides a more effective way of automatically tagging the photos with captions, making it easier for social media users to search and get more precise results of what they want. When designing experiences inclusively, people with *and* without disabilities can benefit, and it significantly enhances the lives of people with disabilities.

## **Dataset**

https://www.kaggle.com/datasets/jessicali9530/caltech256

- There are 30,607 images in this dataset spanning 257 object categories. Object categories are extremely diverse, ranging from grasshopper to tuning fork. We will be able to use these categories to train the datasets using similar features for each set of images.

Backup: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

- The Flickr30k dataset has become a standard benchmark for sentence-based image description.

## **Challenge Goals**

We will be diving deeper into the **machine learning** challenge goal for this project. We will use machine learning techniques including what we learned in the class (to use `Scikit-learn` and `Numpy` packages to utilize machine learning techniques and to deal with particular mathematical problems) and even more advanced skills including graphing various machine learning algorithms we have not covered in the class, comparing the results and choosing the best one (Model Selection for Optimization). The algorithm might be based with reference to the representation learning process. Ultimately, the system uses machine learning algorithms to classify images based on their attributes.

We will also be challenging ourselves by looking into **new libraries** for this project. We are considering utilizing some new libraries including but not limited to the ones listed below:

1. `Scipy` for image manipulation
2. `TensorFlow` for reinforcement learning, optimization effects
3. `PyTorch` for deep learning, neural networks
4. `OpenCV` for image processing
5. `Pillow` for image processing

## **Method**

1. Set up (RQ #1: How can we process images and extract content out of them?)

2. 1. Load the image dataset using libraries such as `numpy`, convert them into same sizes
   2. Cleaning the dataset by removing punctuation marks, special characters, numbers, etc to normalize the tokens in the dataset
   3. Converted to vectors and transfer into the neural network

3. Discovery process

4. 1. Do research on deep learning and deep convolutional neural networks

5. Input dataset and train the models (RQ #2: How can we predict the content of an image and translate them into words and phrases?)

6. 1. Input the cleaned training dataset into the model
   2. Training the model over time and make improvements

7. Apply different models (RQ #3: When applying different machine learning models, there might be different styles of description. Evaluating from the potential stakeholders’ perspective, which model might provide the most accurate, precise information?)

8. 1. Compare the output using accuracy score and mean squared errors to find the appropriate one to use for the purpose of image captioning

9. Evaluate the predicted value from a human’s perspective

10. 1. Doing experiments on different datasets to see the results
    2. Giving feedbacks and suggestions
    3. Recognizing what has been achieved and what has not

## **Work Plan**

Overall, we have 22 days between the proposal due date (Thurs 2/16) and the final code deliverable due date on (Fri 3/10). We expect to spend most of our time setting up data, importing the model from libraries, and developing training models. The following is how we plan to divide the project into discrete tasks over the next 22 days:

1. Set up: 2.16 - 2.19 (~ 12 hours)

2. Discovery process/Looking into different libraries: 2.20 - 2.23 (~ 12 hours)

3. Input dataset and apply different models: 2.24 - 3.8 (~ 21 hours)

4. 1. Train the models on what we have gotten: 3.1 - 3.7 (~ 21 hours)

5. Test model and calculate the accuracy_score for each model: 3.8 (~ 3 hours)

6. Evaluate the predicted value: 3.9 - 3.10 (~ 6 hours)

We will develop and test our code asynchronously (on our own time) and synchronously (Mon/Tues 9-10 pm on Discord). We will coordinate work over text message group chat and if needed, we will discuss work coordination during our weekly meetings. If one task is unexpectedly challenging, team members will speak up in the text group chat (don’t suffer in silence!). That way, when needed, we can schedule meetings outside of our regular meeting times to go over the code as a group. If another meeting is not within our bandwidth, the team member can describe their difficulties over Discord or text, and we can work together to solve the issue asynchronously. As a last resort, we will go to TA office hours for assistance.

​	We will develop python on Visual Studio Code (VScode) using the LiveShare extension to assist in remote collaboration. We will also utilize a [shared GitHub repository](https://github.com/xlesley/cse163) to house our data files and push/pull the files into our local computers when needed. When we want to test out certain lines of code, we will use Google Colaboratory. For the code itself, we will install certain packages like `numpy`, `pandas`, and etc to keep everyone on the same page in terms of package usage, and use comments to indicate the meaning of lines of code. 