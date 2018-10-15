using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleAppTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Specify full path to the trained TF model");
            you ere here,
            //string modelfile = Console.ReadLine();
            string modelfile = @"C:\Users\saurabhd\MyTrials\MachineLearnings-2\MNISTpng\PythonTrainer\TrainedMnistTFModel.pb";

            Console.WriteLine("Specify the top level folder which contains the MNIST test images");
            //string folderWithMNIST = Console.ReadLine();
            string folderWithMNIST = @"C:\Users\saurabhd\MyTrials\MachineLearnings\MNIST101\mnist_png\testing";
            TestMnistImagesUsingTrainedModel(modelfile, folderWithMNIST);
        }

        private static void TestMnistImagesUsingTrainedModel(string modelfile, string folderWithMNIST)
        {
            string[] files = System.IO.Directory.GetFiles(folderWithMNIST, "*.png", System.IO.SearchOption.AllDirectories);
            string[] files_2_3 = files.Where(file => file.Contains("\\2\\") || file.Contains("\\3\\")).ToArray();

            byte[] buffer = System.IO.File.ReadAllBytes(modelfile);
            int countOfFailedClassifications = 0;
            using (var graph = new TensorFlow.TFGraph())
            {
                graph.Import(buffer);
                using (var session = new TensorFlow.TFSession(graph))
                {
                    foreach (string file in files_2_3)
                    {
                        Stopwatch sw = new Stopwatch();
                        sw.Start();
                        var runner = session.GetRunner();
                        var tensor = Utils.ImageToTensorGrayScale(file);
                        runner.AddInput(graph["conv1_input"][0], tensor);
                        runner.Fetch(graph["activation_4/Softmax"][0]);

                        var output = runner.Run();
                        var vecResults = output[0].GetValue();
                        float[,] results = (float[,])vecResults;
                        sw.Stop();
                        ///
                        /// Evaluate the results
                        ///
                        int[] quantized = Utils.Quantized(results);
                        string parentFolder = System.IO.Directory.GetParent(file).Name;
                        int[] expected = (parentFolder == "2") ? Utils.OUTPUT_2 : Utils.OUTPUT_3;
                        bool success = quantized.SequenceEqual(expected);
                        if (!success) countOfFailedClassifications++;
                        string message = $"Directory={parentFolder}    File={System.IO.Path.GetFileName(file)}    Bit1={results[0, 0]} Bit2={results[0, 1]}   Elapsed={sw.ElapsedMilliseconds} ms, Success={success}";
                        Console.WriteLine(message);
                    }
                }
            }
            double totalfiles = files_2_3.Length;
            double overallsuccess = 100.0 * (totalfiles - countOfFailedClassifications) / totalfiles;
            Console.WriteLine($"total failures={countOfFailedClassifications} out of {totalfiles} files, success%={overallsuccess}");
        }
    }
}
