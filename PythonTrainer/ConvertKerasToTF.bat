echo off
rem This BAT file will invoke the Python script keras_to_tensorflow.py
rem This step is neccessary when we want to consume the model in a C# console application built using TensorflowSharp library
rem
python keras_to_tensorflow.py --input_model=TrainedMnistModelWts.h5  --input_model_json=TrainedMnistModel.json --output_model=TrainedMnistTFModel.pb
