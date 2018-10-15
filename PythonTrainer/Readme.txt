This is a set of Python scripts which will use Keras and TF train MNIST using PNG files
***************************************************************************************

What does this script produce?
------------------------------
A Keras model which can be consumed by other applications (e.g. a C# application for classifying images at runtime)


How to train?
-------------

Step 1
------
	Extract all the MNIST training and testing images to a folder

Step 2
------
	The script TrainMnistFromFolder.py will initiate the training by reading the training images. Depending on 
the location of the folder in Step 1, you will need to change the variable values.

Outcome of Step 2
-----------------
	Step 2 will product 
		1)Model file (JSON)
		2)Weights file (H5)
Step 3
------
	Convert the Keras model to a file that TF can understand. 
	Execute the script ConvertKerasToTF.bat
	This will produce a .PB file - which contains both the neural network structure and weights.

Step 4
------
	Move on to the step of consuming the model file in a C# console application.
