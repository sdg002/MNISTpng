using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleAppTester
{
    public class Utils
    {
        public static int[] OUTPUT_2 = new int[2] { 1, 0 };
        public static int[] OUTPUT_3 = new int[2] { 0, 1 };
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
        internal static int[] Quantized(float[,] results)
        {
            int[] q = new int[]
            {
                results[0,0]>0.5?1:0,
                results[0,1]>0.5?1:0,
            };
            return q;
        }

    }
}
