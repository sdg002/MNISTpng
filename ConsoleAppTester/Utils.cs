using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleAppTester
{
    public class Utils
    {
        /// <summary>
        /// Load a PNG and create a TF tensor out of it
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public static TensorFlow.TFTensor ImageToTensorGrayScale(string file)
        {
            using (System.Drawing.Bitmap image = (System.Drawing.Bitmap)System.Drawing.Image.FromFile(file))
            {
                var matrix = new float[1, image.Size.Height, image.Size.Width, 1];
                for (var iy = 0; iy < image.Size.Height; iy++)
                {
                    for (int ix = 0, index = iy * image.Size.Width; ix < image.Size.Width; ix++, index++)
                    {
                        System.Drawing.Color pixel = image.GetPixel(ix, iy);
                        matrix[0, iy, ix, 0] = pixel.B / 255.0f;
                        //matrix[0, iy, ix, 1] = pixel.Green() / 255.0f;
                        //matrix[0, iy, ix, 2] = pixel.Red() / 255.0f;
                    }
                }
                TensorFlow.TFTensor tensor = matrix;
                return tensor;
            }
        }
        /// <summary>
        /// Converts the TF result into a 10 dimensional vector
        /// Silly repetitions here! I was running out of time.
        /// </summary>
        /// <param name="results"></param>
        /// <returns></returns>
        internal static int[] Quantized(float[,] results)
        {
            int[] q = new int[]
            {
                results[0,0]>0.5?1:0,
                results[0,1]>0.5?1:0,
                results[0,2]>0.5?1:0,
                results[0,3]>0.5?1:0,
                results[0,4]>0.5?1:0,
                results[0,5]>0.5?1:0,
                results[0,6]>0.5?1:0,
                results[0,7]>0.5?1:0,
                results[0,8]>0.5?1:0,
                results[0,9]>0.5?1:0,
            };
            return q;
        }
        /// <summary>
        /// Given a digit , it returns the vectorized output in 10 dimensions
        /// e.g. 
        ///     0->0000000000
        ///     1->0100000000
        ///     2->0010000000
        /// </summary>
        /// <param name="iParentFolder"></param>
        /// <returns></returns>
        internal static int[] GetQuantizedExpectedVector(int iParentFolder)
        {
            int[] output = new int[10] {
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    0
            };
            output[iParentFolder] = 1;
            return output;
        }
    }
}
