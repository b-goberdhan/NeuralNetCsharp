using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    /*
   public class Network
   {
      private double[] target;
      private double calculatedVal;
      private Random random;

      private List<Neuron> inputLayer, hiddenLayer, outputLayer;



      public Network(int inputSize, int hiddenSize, int outputSize)
      {
         inputLayer = new List<Neuron>(inputSize);
         hiddenLayer = new List<Neuron>(hiddenSize);
         outputLayer = new List<Neuron>(outputSize);
         target = new double[outputSize];
         random = new Random();
         //Build Network
      }

      private void initLayers(Func<double, double> actFunction)
      {
         for (int i = 0; i < hiddenLayer.Capacity; i++)
         {
            hiddenLayer.Add(new Neuron(actFunction, NeuronType.Hidden));
         }

         for (int i = 0; i < inputLayer.Capacity; i++)
         {
            inputLayer.Add(new Neuron(actFunction, NeuronType.Input));
         }

         for (int i = 0; i < outputLayer.Capacity; i++)
         {
            outputLayer.Add(new Neuron(actFunction, NeuronType.Output));
         }
      }

      public void BuildNetwork(Func<double,double> actFunction)
      {

         initLayers(actFunction);
         //connect each input to a hidden neuron
         foreach (Neuron neuron in inputLayer)
         {
            foreach(Neuron hiddenNeuron in hiddenLayer)
            {
               double randomNum = random.NextDouble();
               hiddenNeuron.AddNeuronBefore(neuron, randomNum);
               neuron.AddNeuronAfter(hiddenNeuron, randomNum);

            }
         }

         foreach (Neuron neuron in outputLayer)
         {
            foreach (Neuron hiddenNeuron in hiddenLayer)
            {
               double randomNum = random.NextDouble();
               hiddenNeuron.AddNeuronAfter(neuron, randomNum);
               neuron.AddNeuronBefore(hiddenNeuron, randomNum);
            }
         }


      }

      public bool SetupInputValues(params double[] input)
      {
         if (input.Length != inputLayer.Capacity)
            return false;
         else
         {
            for (int i = 0; i < input.Length; i++)
            {
               inputLayer[i].Result = input[i];
            }
            return true;
         }
      }

      public bool SetupTargetValues(params double[] output)
      {
         if (output.Length != outputLayer.Capacity)
            return false;
         else
         {
            for (int i = 0; i < output.Length; i++)
            {
               target[i] = output[i];
            }
            return true;
         }
      }


      public void ForwardPropagation()
      {
         //take each hidden layer and have the sum equal to the value of the input times the wieght for each edge
         //start with a hidden neuron
         Console.WriteLine("Hidden Layer values");
         foreach (Neuron neuron in hiddenLayer)
         {
            double sum = 0;
            //find everthing before
            foreach (KeyValuePair<Neuron, double> entry in neuron.NeuronsBefore)
            {
               sum += (entry.Key.Result * entry.Value);
            }
            neuron.Result = sum;
            Console.WriteLine(sum);
         }

         //finally do the same for the output layer
         Console.WriteLine("Output Layer values");
         foreach(Neuron neuron in outputLayer)
         {
            double sum = 0;
            //find everthing before
            foreach (KeyValuePair<Neuron, double> entry in neuron.NeuronsBefore)
            {
               sum += (entry.Key.Result * entry.Value);
            }
            
            neuron.Result = sum;
            Console.WriteLine(sum);
         }

      }

      public double Sigmoid(double x)
      {
         return (1 / (1 + Math.Pow(Math.E, -x)));
      }


   }
   */
}
