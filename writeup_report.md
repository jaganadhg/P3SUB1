## Project 3 Behavioural Cloning
### Objective
The goal of this project is to build a Machine Learning Model, which can mimic the driver behavior. In this project, we used Deep Learning (Convolution Network) architecture for building such a model. 

### Data Collection
We used the Udacity Self-Driving Car Simulator for collecting the data. We collected the data from total eight laps. Four laps the car was moving in the normal direction. The next four laps we reversed the direction of the car to collect the data. This step was performed to avoid data augmentation. In fact, the image and driving angle generated as part of reversing the car also included in the data. 


### Data Preprocessing

We tried different pre-processing techniques in the image, including re-sizing, grayscale conversion, and filtering, etc.. After testing it in a couple of models, we decided to move with a regular image normalization and crop the image y top 50 and bottom 20. This step is included in the model building process with the KEras Lambda layer. 

### Model Building

We have experimented with different models, which consumes image from all the cameras and only consumes the center camera. After evaluating the performance of the model in the simulator, we finally settled on a model which is based on the NVIDIA architecture. We have tried image augmentation too. 

In the final model, we didn't use any augmentation (because we have reverse track data in training). Image only from the center camera is used for training. 


Our model is adopted from the  NVIDIA architecture, which consists of 5 CNN layers, dropout, and four dense layers. Among the 5 CNN layers, three have 5x5 filter sizes, and 2 have 3x3 filter size. These layers include RELU layers to introduce nonlinearity. Then a dropout layer is subsequent to reduce overfitting.The model used an adam optimizer to tune the learning rate.


The training process was controlled by an early stopping callback in the Keras library. This helped us to control the number of the epoch. The final model converged at the 8th iteration. To adopt the code for much larger data we used a generator based training data loader. We adopted an Object Oriented Generator in Python so that we can multi-process capability in the Keras fir_generator. It helped us to use an 8GB RAM 4 Core MacOSX machine for our training (Even though it growled like a wild animal ;-)). 

### Future Steps

The model testing simulator didn't provide any facility to fetch the images from all the three cameras. It might have been a lot more fun. We found a very interesting way to use the image and some meta-properties to build an interesting model. Waiting for the simulator enhancement to get all the three camera images. 

### Output Videos 
I have uploaded the ouput videos in this repo. But to avoid check-out of learge file, it is available in the Youtube link https://www.youtube.com/watch?v=qoPksQio1gA 
