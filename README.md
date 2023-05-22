#  Muscle exhaustion detection based on ultrasound images

Based on ultrasound images the model provides an estimate of the level of exhaustion of the muscle.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)

## Project Description

This project data was collected by performing a series of squats until exhaustion was reached. For every squat the image of the ultrasound probe was saved and then was assigned a category based on the level of exhaustion (1 through 5).
The model can detect some level of exhaustion in the muscle as the accuracy is not completely random (around 31%) when it uses the EffientnetV2 architecture. 

## Installation

Follow the steps below after the repository has been cloned. 

- Create a virtual environment, ie 'py -3.10 -m venv .venv'
- Run 'pip install -r requirements.txt' to install all necessary packages

# Usage
To train and test the model run 'py model\main.py'.

The following arguments can be added to the command above:
- nb_epochs (determines numbers of epochs to run. Default = 10)
- batch_size (defines the number of samples that will be propagated through the network. Default = 2) 
- model_type (chooses with model to go with. Options are resnet, efficientnet, custom. Default = efficientnet)

Custom model_type can be changed in the model\net.py file.
Arguments can be added to the 'py model\main.py' by adding '--model_type resnet', so 'py model\main.py --model_type resnet'.
