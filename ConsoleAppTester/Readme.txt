What does this project folder contain?
**************************************
C# console app using .NET Framework 4.7. This executable will consume a trained model file (PB5) and evaluate MNIST images. The code will produce a 10 dimensional vector. Each bit of this vector denotes a digit.

Prerequisites
--------------
TensorflowSharp page
Trained model
MNIST test images extracted to a folder

Other requirements
------------------
64 bit.




Change to Readme.md
Steps for extracting the MNIST test and train images

Next steps
-----------
Clean up TrainMnistFromFolder.py
Prompt for path


Install Module
---------------

py -m pip list
py -m pip install opencv-python
You will need to install Keras
py -m pip install keras
py -m pip install tensorflow-gpu
py -m pip install tensorflow-gpu==1.10 

ImportError: Could not find 'cudart64_100.dll'. TensorFlow requires that this DLL be installed in a directory that is named in your %PATH% environment variable. Download and install CUDA 10.0 from this URL: https://developer.nvidia.com/cuda-90-download-archive


Steps
------

	Step 1
	------
	Did not work
Installing CUDA 10.1 from https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
cuda_10.1.168_425.25_win10.exe

	Step 2
	------
	Did not work
Add the following directory to PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\

	Step 3
	------
	Download v9 from https://developer.nvidia.com/cuda-90-download-archive (this is the link presented in the error message)
	this did not work

	Step 4
	-------
	Uninstall CUDA 10.1  (Control Panel, removed all 10.1 references, Samples, Documentation,etc.)
	and install CUDA 10.0

	Step 5
	-------
	Copy cudnn64_7.dll
	from
		https://www.tensorflow.org/install/gpu  (cuDNN 7.4.1)
		https://developer.nvidia.com/rdp/cudnn-download  (cuDNN 7.6.2, CUDA 10)
	to
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
	Status: CUDA driver version is insufficient for CUDA runtime version
	Problem - Driver version was not uptodate

	Step 6
	-------
	Go to device manager -->Display adapters --> Update driver (24.21.14.1131 worked)


	Step 7
	------
	Testing fails with error
	TensorFlow.TFException: NodeDef mentions attr 'explicit_paddings' not in Op<name=Conv2D; signature=input:T, filter:T -> output:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]; attr=strides:list(int); attr=use_cudnn_on_gpu:bool,default=true; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]; attr=dilations:list(int),default=[1, 1, 1, 1]>; NodeDef: {{node conv1/convolution}} = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], explicit_paddings=[], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true](conv1_input, conv1/kernel/read). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
		at TensorFlow.TFStatus.CheckMaybeRaise(TFStatus incomingStatus, Boolean last)
		at TensorFlow.TFGraph.Import(TFBuffer graphDef, TFImportGraphDefOptions options, TFStatus status)
		at TensorFlow.TFGraph.Import(Byte[] buffer, TFImportGraphDefOptions options, TFStatus status)
		at TensorFlow.TFGraph.Import(Byte[] buffer, String prefix, TFStatus status)
		at ConsoleAppTester.Program.TestMnistImagesUsingTrainedModel(String modelfile, String folderWithMNIST) in C:\Users\saurabhd\MyTrials\MachineLearnings-2\MNISTpng\ConsoleAppTester\Program.cs:line 101
		at ConsoleAppTester.Program.Main(String[] args) in C:\Users\saurabhd\MyTrials\MachineLearnings-2\MNISTpng\ConsoleAppTester\Program.cs:line 23


	Step 8
	------
	Added PY script to save Keras session to TF
		https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

	Step 9
	------
	Update libtensorflow.dll
		(1)Download latest TF dll from https://www.tensorflow.org/install/lang_c
		(2)Add to ConsoleAppTester and set "Copy to output directory"

	Convert Keras to TF
	-------------------
	https://github.com/keras-team/keras/issues/3223
	https://github.com/amir-abdi/keras_to_tensorflow



How to find a list of all versions of a package?
-------------------------------------------------
pip install yolk3k
yolk -V tensorflow-gpu
