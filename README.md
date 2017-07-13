# SelfDrivingCar_CNN
## Introduction
This is an application of [Convolution Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) which is trained to drive a car. The very first problem in having to build such a model is to have a dataset to train the model, this in itself was a very tedious task which was not so practical until recently when the games were used to collect the artificial data. This project uses one such [simulator](https://github.com/udacity/self-driving-car-sim) by [Udacity](https://www.udacity.com) for their **Self-Driving Car Nanodegree** (cheers to **Udacity** for opening thier simulator and the wrapper programs to set up the simulator for everyone to use).

The model to train the *CNN* is built using [Keras](https://keras.io) with [Tensorflow](https://www.tensorflow.org) at the back-end, hence it's a no brainer that we need these two librarires for the project, other than these some of the other librarires include **opencv3** (for image transformations), **PIL** (for loading, saving and basic manipulation). Also one cool aspect (which was done by udacity by default) is the [Client-Server model](https://en.wikipedia.org/wiki/Client–server_model) between the simulator and my python scripts, so that one can control the car in the simulator with the ouput genrated by the python scripts. An exhaustive list of all the dependencies can be found in the *yml* files in the **src**.

The model implemented for the self driving car in **src** is the model proposed by **NVIDIA** in thier research work [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) It consists of **9 layers** the first being the layer performing normalisation, the next **5** being the *convolutional layers* and **3** *fully connected layers* there is also an intermediate layer to flaateen out the state matrix after all the convolutional layes and there is a final fully connected layer, which gives just one output (the steering angle).

## Some Analysis
Certain subtle issues and analysis of the data is discussed in this [blog post](https://udionblog.wordpress.com) (coming soon) by me.

## Usage
* One has to download the simulator mentioned above by **Udacity**, however if this repository is cloned one should find these simulator in **beta_simulator_linux** directory (only for linux).
  * To open the simulator shift to the directory containing the simulator and do ```sudo chmod +x beta_simulator.*``` (to make them executables). then open the simulator by  ```./beta_simulator.x86_64```
  * Simulator allows for the two mode the `Traing Mode` and `Autonomous Mode` and offeres to tracks. Training mode can be used to collect the data as one drives the car manually. (*My python scripts by default look for the data in the* **data** *directory, so it is suggest to select that directory for storing the data, else one might pass the data directory manually while training the model, check the source code to figure out how.*). Autonomous mode can be used to check the trained model, to see how the model is performing.
* Life is alot easier if one is using [Anaconda](https://www.continuum.io/downloads) which is a great data science ecosystem for python which also happens to manage virtual environments. I assume that *Anaconda* is installed in the system.(**Note**: One can also set up the virtual environment manually and install every dependency mentioned in the *yml* file manually using [pip](https://pypi.python.org/pypi/pip)). I did not use the *Tensorflow* framework running on *GPU* however one can do that too (if one has necessary hardware and software setup)
  * shift to the **src** directory and do the following to set up the enivironment, see the list of present environments and activating the environment created by the *yml* file.
  ```
  conda env create -f environments.yml
  conda env list
  source activate SelfDrivingCar_CNN
  ```
  * If the data is collected using training mode in the **data** directory then do the following to build and store the model in *h5* format in the **models** directory
  ```
  python model.py
  ```
  This will run the script in the default mode, although script is capable of accepting various parameters in arguement check the source code to figure out or do `python model.py --help`
  * To test the model built and trained using the data, open the simulator in the autonomous mode and run the `drive.py` as follows, it can also accept an arguement which will represent the name of the directory present inside the **testRunRecording** to record the images from the drive (by default script is written to read the models stored in **models** directory and to **not save** the drives of the model)
  ```
  python drive.py ../models/model-xxx.h5 testDrive1 # for recording the drive
  python drive.py ../models/model-xxx.h5 # for not recording the drive
  ```
  model-xxx.h5 represents the model stored in **models** directory
## Notes and Results
* The **data** directory present in the repo is the truncated ones because of the size restriction of the github, but it gives an idea about the format in which images are stored (well maybe a more memory efficient way would be to read the source code)
* The 2 models in the **models** are from the same data set, one from the starting epoch and the other from the last epoch. They serve as a very good example of our model improvoing as one can clearly see the poor performance for the first epoch model (car sinks in water for most of the time :P) and the model obtained after the last epoch can successfully drive the car completing laps.
  * result visuals for model just after the first epoch (notice how it goes in water in the end)

  ![results for model-000.h5](https://github.com/udion/SelfDrivingCar_CNN/blob/master/results/summary_model-000.gif)

  * results for the final model

  ![results for model-009.h5](https://github.com/udion/SelfDrivingCar_CNN/blob/master/results/summary_model-009.gif)
