# MNIST - From Python to C#

# What is this project about?
A sample end to end project demonstrating how to train MNIST images using Keras/Tensforflow and then write a C# application using the TensorFlowSharp nuget package to load the trained model and evaluate images at runtime.

# My article on CodeProject
The code in this repo is used by my CodeProject article here.
https://www.codeproject.com/Articles/5164135/TensorFlow-Creating-Csharp-Applications-using


## MNISTpng
Full dump of MNIST in png format (60K+10K files, ZIP format)

## Python Trainer
Python scripts which will load the training PNG files from the MNISTpng folder and created a trained model.

## ConsoleAppTester
C# console exe project. This exe will load the trained model and then test each of the 10,000 images from the test set. The model will emit a 10 dimension vector. .NET 4.6)


