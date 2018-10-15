This is a Python script which will use Keras and TF train MNIST using PNG files


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
		1)Model file (json)
		2)Weights file (h5)
Step 3
------
	Convert the Keras model to a file that TF can understand. 
	Execute the script ConvertKerasToTF.bat

Step 4
------
	Run the C# console application which will consume the trained model and then evaluate the test images