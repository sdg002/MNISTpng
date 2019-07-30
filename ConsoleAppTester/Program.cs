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
            string modelfile = null;
            string folderWithMNIST = null;
            try
            {
                modelfile = PromptUserForTrainedModel();
                folderWithMNIST = PromptUserForMNISTTestImages();

                //modelfile = @"C:\Users\saurabhd\MyTrials\MachineLearnings-2\MNISTpng\PythonTrainer\TrainedMnistTFModel.pb";
                //folderWithMNIST = @"C:\Users\saurabhd\MyTrials\MachineLearnings\MNIST101\mnist_png\testing";
                TestMnistImagesUsingTrainedModel(modelfile, folderWithMNIST);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private static string PromptUserForMNISTTestImages()
        {
            string lastKnownTestImagesFolder = Properties.Settings.Default.dir_testing;
            string folderTestImages = null;
            string yesno = null;
            if (
                (string.IsNullOrWhiteSpace(lastKnownTestImagesFolder) == false) &&
                (System.IO.Directory.Exists(lastKnownTestImagesFolder) == true)
                )
            {
                Console.WriteLine("------------------------");
                Console.WriteLine("Last known test images folder:");
                Console.WriteLine(lastKnownTestImagesFolder);
                Console.WriteLine("------------------------");
                Console.WriteLine("Do you want to use the last selected Test images directory (y/n)?");
                yesno = Console.ReadLine();
                if (yesno == "y")
                {
                    folderTestImages = lastKnownTestImagesFolder;
                    return folderTestImages;
                }
            }
            Console.WriteLine("Specify full path to Test images directory");
            folderTestImages = Console.ReadLine();
            Properties.Settings.Default.dir_testing = folderTestImages;
            Properties.Settings.Default.Save();
            return folderTestImages;
        }

        private static string PromptUserForTrainedModel()
        {
            string lastModelFile = Properties.Settings.Default.file_trainedmodel;
            string yesno = null;
            string modelfile = null;
            if (
                (string.IsNullOrWhiteSpace(lastModelFile) == false) &&
                (System.IO.File.Exists(lastModelFile) == true)
                )
            {
                Console.WriteLine("------------------------");
                Console.WriteLine("Last known model file:");
                Console.WriteLine(lastModelFile);
                Console.WriteLine("------------------------");
                Console.WriteLine("Do you want to use the last known model file (y/n)?");
                yesno = Console.ReadLine();
                if (yesno.ToLower() == "y")
                {
                    modelfile = lastModelFile;
                    return modelfile;
                }
            }
                        
            Console.WriteLine("Specify absolute path to the trained TF model:");
            Console.WriteLine("----------------------------------------------");
            modelfile = Console.ReadLine();
            Properties.Settings.Default.file_trainedmodel = modelfile;
            Properties.Settings.Default.Save();
            return modelfile;
        }

        private static void TestMnistImagesUsingTrainedModel(string modelfile, string folderWithMNIST)
        {
            Random rnd = new Random(DateTime.Now.Second);
            string[] filesTesting = System.IO.Directory.GetFiles(folderWithMNIST, "*.png", System.IO.SearchOption.AllDirectories);
            string[] filesTestingRandomized = filesTesting.OrderBy(f=> rnd.Next()).ToArray();// files.Where(file => file.Contains("\\2\\") || file.Contains("\\3\\")).ToArray();

            byte[] buffer = System.IO.File.ReadAllBytes(modelfile);
            int countOfFailedClassifications = 0;
            using (var graph = new TensorFlow.TFGraph())
            {
                graph.Import(buffer);
                using (var session = new TensorFlow.TFSession(graph))
                {
                    foreach (string file in filesTestingRandomized)
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
                        ///
                        /// Use the parent folder name to deterimine the expected digit
                        ///
                        string parentFolder = System.IO.Directory.GetParent(file).Name;
                        int iParentFolder = int.Parse(parentFolder);
                        int[] expected = Utils.GetQuantizedExpectedVector(iParentFolder);
                        bool success = quantized.SequenceEqual(expected);
                        if (!success) countOfFailedClassifications++;
                        string message = $"Directory={parentFolder}    File={System.IO.Path.GetFileName(file)}    Bit1={results[0, 0]} Bit2={results[0, 1]}   Elapsed={sw.ElapsedMilliseconds} ms, Success={success}";
                        Console.WriteLine(message);
                    }
                }
            }
            double totalfiles = filesTestingRandomized.Length;
            double overallsuccess = 100.0 * (totalfiles - countOfFailedClassifications) / totalfiles;
            Console.WriteLine($"total failures={countOfFailedClassifications} out of {totalfiles} files, success%={overallsuccess}");
        }
    }
}
