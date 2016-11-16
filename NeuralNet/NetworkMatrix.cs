using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace NeuralNet
{
   public delegate double ActivationFunction(double x);
   
   public class NetworkMatrix
   {

      private Matrix inputLayer, inputHiddenWeights, hiddenOutputWeights, hiddenResult, hiddenSum, outputResult, outputSum, errorOutput, targetOutput;
      private ActivationFunction activationFunction;
      private Random random;
      private Thread thread;

      public bool LearnBackground { get; set; }
      public struct HiddenLayer
      {

         public List<Matrix> wieghts;
         public List<Matrix> sums;
         public List<Matrix> results;
         public int neuronsPerLayer;
         public int layers;

      }

      private HiddenLayer hidLayer;

      private int inputNum;
      private int outputNum;
      private double learningRate = 1;
      public NetworkMatrix(Func<double,double> actFucntion, Matrix input, int hiddenNeurons, int hiddenLayers ,Matrix output)
      {
         random = new Random();
         activationFunction = new ActivationFunction(actFucntion);
         this.inputLayer = input;
         this.targetOutput = output;
         inputNum = input.Columns;
         outputNum = output.Columns;
         InitLayers(input, output, hiddenLayers, hiddenNeurons);
            //init(hiddenNeurons);
         thread = new Thread(() => BackgroundLearn(this));
      }

      public void RunBackgroundLearn()
      {
         thread.Start();
      }

      public void StopBackgroundLearn()
      {
         thread.Abort();
         thread.Join();
      }

      public static double Sigmoid(double x)
      {
         return (1 / (1 + Math.Pow(Math.E, -x)));
      }
      public static double SigmoidPrime(double x)
      {
         return (Sigmoid(x) * (1-Sigmoid(x)));
      }   

      private void InitLayers(Matrix input, Matrix output, int layers, int neurons)
      {
         inputHiddenWeights = new Matrix(input.Columns, neurons);
         for (int i = 0; i < inputHiddenWeights.Rows; i++)
         {
            for (int j = 0; j < inputHiddenWeights.Columns; j++)
            {
               inputHiddenWeights[i, j] = random.NextDouble();
            }
         }
         InitHiddenLayer(layers, neurons, input * inputHiddenWeights);
         InitOutputLayer(output);

      }
      private void InitHiddenLayer(int layers, int neurons, Matrix firstSum)
      {
         hidLayer = new HiddenLayer() { wieghts = new List<Matrix>(), sums = new List<Matrix>(), results = new List<Matrix>()};
         hidLayer.layers = layers;
            hidLayer.neuronsPerLayer = neurons;
         for (int i = 0; i < layers - 1; i++)
         {
            Matrix tempWieght = new Matrix(neurons, neurons);
            for (int x = 0; x < tempWieght.Rows; x++)
            {
               for (int y = 0; y < tempWieght.Columns; y++)
               {
                  tempWieght[x, y] = random.NextDouble();
               }
            }
            hidLayer.wieghts.Add(tempWieght);
         }

         hidLayer.sums.Add(firstSum);
         hidLayer.results.Add(Matrix.Transform(Sigmoid, firstSum));

         for (int i = 0; i < layers - 1; i++)
         {
            Matrix temp = hidLayer.sums[i] * hidLayer.wieghts[i];
            hidLayer.sums.Add(temp);
            hidLayer.results.Add(Matrix.Transform(Sigmoid, temp));
         }
         


      }
      private void InitOutputLayer(Matrix output)
      {
         //take number of columns on this layer (numper of output neurons)
         //need also the number of neurons in a hidden layer
         hiddenOutputWeights = new Matrix(hidLayer.neuronsPerLayer, output.Columns);
         for (int i = 0; i < hiddenOutputWeights.Rows; i++)
         {
            for (int j = 0; j < hiddenOutputWeights.Columns; j++)
            {
               hiddenOutputWeights[i, j] = random.NextDouble();
            }
         }

         //take last hidden layer
         outputSum = hidLayer.results[hidLayer.results.Count - 1] * hiddenOutputWeights;
         outputResult = Matrix.Transform(Sigmoid, outputSum);


      }
      private void ForwardPropagation()
      {
         //Everything is initilized, random wieghts sums and results are computed.
         //We need the wieght from sum to first hidden layer

         hidLayer.sums[0] = inputLayer * inputHiddenWeights;
         hidLayer.results[0] = Matrix.Transform(Sigmoid, hidLayer.sums[0]);

         for (int i = 1; i < hidLayer.layers; i++)
         {   
            hidLayer.sums[i] = hidLayer.sums[i - 1] * hidLayer.wieghts[i - 1];
            hidLayer.results[i] = Matrix.Transform(Sigmoid, hidLayer.sums[i]);
         }

         outputSum = hidLayer.results[hidLayer.results.Count - 1] * hiddenOutputWeights;
         outputResult = Matrix.Transform(Sigmoid, outputSum);



      }
      private void BackwardPropagation()
      {

         int last = hidLayer.results.Count - 1;
         errorOutput = targetOutput - outputResult;
            //do initial fix for out put to last hidden layer
         Matrix deltaLayer = Matrix.HadamardProd(Matrix.Transform(SigmoidPrime, outputSum), errorOutput);
         Matrix layerChanges = Matrix.Transpose(hidLayer.results[last]) * deltaLayer * learningRate;
         Matrix previousWieght = hiddenOutputWeights;
         hiddenOutputWeights += layerChanges;

         for (int i = 0; i < last ; i++)
         {
            deltaLayer = Matrix.HadamardProd(deltaLayer * Matrix.Transpose(previousWieght), Matrix.Transform(SigmoidPrime, hidLayer.sums[last - i]));
            layerChanges = Matrix.Transpose(hidLayer.results[last - i]) * deltaLayer * learningRate;
            previousWieght = hidLayer.wieghts[last - 1 - i];
            hidLayer.wieghts[last - 1 - i] += layerChanges;
         }

         deltaLayer = Matrix.HadamardProd(deltaLayer * Matrix.Transpose(previousWieght), Matrix.Transform(SigmoidPrime, hidLayer.sums[0]));
         layerChanges = Matrix.Transpose(inputLayer) * deltaLayer * learningRate;
         inputHiddenWeights += layerChanges;
          
       
           

      }

      public void Learn(int iterations)
      {
         while (iterations > 0)
         {
            BackwardPropagation();
            ForwardPropagation();
                  
            Console.WriteLine("\r"+outputResult.ToString());
            iterations--;
         }

         Console.WriteLine("Input Hidden Wieghts:");
         Console.WriteLine(inputHiddenWeights.ToString());

         Console.WriteLine("Hidden Output Wieghts:");
         Console.WriteLine(hiddenOutputWeights.ToString());

         Console.WriteLine("With " + hidLayer.layers + " hidden layers.");
         

      }

      public void Learn()
      {
          BackwardPropagation();
          ForwardPropagation();
      }
    
      public static void BackgroundLearn(NetworkMatrix net)
      {
         while (true)
         {
            net.Learn();
         }
      }


      public Matrix Use(Matrix useInput)
      {
         //get product of all wieghts
         Matrix newOutput = Matrix.Transform(Sigmoid, useInput * inputHiddenWeights);
         for (int i = 0; i < hidLayer.wieghts.Count; i++)
         {
            newOutput *= hidLayer.wieghts[i];
            newOutput = Matrix.Transform(Sigmoid, newOutput);
         }
        
         newOutput = Matrix.Transform(Sigmoid, (newOutput * hiddenOutputWeights));

         return newOutput;
      }

   }
}
