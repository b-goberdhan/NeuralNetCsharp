using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet;

namespace NeuralNetTester
{
   class Program
   {
      static void Main(string[] args)
      {
        
         //Network network = new Network(2, 3, 1);
         /*network.BuildNetwork(network.Sigmoid);
         network.SetupInputValues(1, 1);
         network.SetupTargetValues(0);
         network.ForwardPropagation();
         */

         Matrix mat1 = new Matrix(4, 2, 
                                    0, 0, 
                                    0, 1, 
                                    1, 0, 
                                    1, 1);
         Matrix mat2 = new Matrix(4, 1,
                                    0,
                                    1,
                                    1,
                                    0);

         //Matrix mat3 = Matrix.Multiply(mat1, mat2);
         NetworkMatrix netMat = new NetworkMatrix(NetworkMatrix.Sigmoid, mat1, 5, 1, mat2);
         netMat.Learn(10000);
         Console.ReadLine();
         netMat.RunBackgroundLearn();


         while (true)
         {
             Matrix temp = netMat.Use(new Matrix(1, 2,
                                  1, 0));
             Console.WriteLine(temp.ToString());
             if (Console.ReadLine() == "stop")
                 break;
         }

         netMat.StopBackgroundLearn();


      }
   }
}
