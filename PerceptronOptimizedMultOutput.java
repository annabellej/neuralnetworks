/*
 * Name: Annabelle Ju
 * Creation date: 11 October 2019
 * Updated: 8 January 2020
 *
 * Description: A simple, multilayer Perceptron connectivity model with
 * configurable activations.
 */

import java.io.*;
import java.util.Scanner;
import java.util.StringTokenizer;

public class PerceptronOptimizedMultOutput
{
   double[][] activations, thetaVals;
   double[][][] weights;
   int numLayers, numRows, numInputs, numOutputs;
   double[] trainingValues;
   double learningFactor;

   /*
    * Constructor for a simple, multilayer Perceptron.
    *
    * @param numInputs          the number of input values.
    * @param numOutputs         the number of outputted values.
    * @param hiddenLayerNodes   the number of activation units in each layer.
    * @param rows               the max number of rows in any layer.
    * @param lambda             the learning factor for training (not adaptive).
    * @param weightsLow         the lower bound for randomizing weight values.
    * @param weightsHigh        the higher bound for randomizing weight values.
    */
   public PerceptronOptimizedMultOutput(int numInputs, int numOutputs, int[] hiddenLayerNodes, double lambda,
                                        double weightsLow, double weightsHigh)
   {
      this.numInputs      = numInputs;
      this.numOutputs     = numOutputs;
      this.numLayers      = hiddenLayerNodes.length+2;

      this.learningFactor = lambda;

      activations = new double[numLayers][];
      activations[numLayers-1] = new double[numOutputs];

      for (int n = 1; n < numLayers-1; n++)
      {
         activations[n] = new double[hiddenLayerNodes[n-1]];
      }

      int rows = numInputs;
      for (int num: hiddenLayerNodes)
      {
         rows = Math.max(rows, num);
      }
      rows = Math.max(rows, numOutputs);

      this.numRows = rows;

      weights = new double[numLayers-1][rows][rows];

      randomizeWeightValues(weightsLow, weightsHigh);

      thetaVals = new double[numLayers][numRows];
   }

   /*
    * Generates a random number between a given range.
    *
    * @param low  the lower bound for the random number.
    * @param high the upper bound for the random number.
    *
    * @return a random number between the range low and high.
    */
   public double randNum (double low, double high)
   {
      return (Math.random()*(high-low))+low;
   }

   /*
    * Sets the weight values to randomized numbers.
    * The range of weight values is set by the user.
    *
    * @param low   the lower bound of the randomized weight values.
    * @param high  the higher bound of the randomized weight values.
    */
   public void randomizeWeightValues(double low, double high)
   {
      for (int m = 0; m < weights.length; m++)
      {
         for (int source = 0; source < weights[m].length; source++)
         {
            for (int dest = 0; dest < weights[m][source].length; dest++)
            {
               weights[m][source][dest] = randNum(low, high);
            }
         }
      }
   }

   /*
    * For the purpose of this simple model, the propagation rule
    * calculates the dot product of all the connections into
    * the unit in question.
    *
    * @precondition 0<layer<total number of layers and row<number of rows in this layer
    *
    * @param layer   specifies the layer number of the unit in question.
    * @param row     specifies the row number of the unit in question.
    *
    * @return the dot product of all the connections into the specified unit.
    */
   public double propagationFunction(int layer, int row)
   {
      double result = 0.0;
      int prevLayer = layer-1;

      for (int r = 0; r < activations[prevLayer].length; r++) //Iterate over each unit in the preceding layer
      {
         result += activations[prevLayer][r] * weights[prevLayer][r][row];
      }

      return result;
   }

   /*
    * Applies the sigmoid function as a threshold function for a given net input.
    *
    * @param  netInput the value calculated by the propagation rule,
    *                 to be manipulated by the activation function
    *
    * @return the new state of activation for a unit
    */
   public double activationFunction(double netInput)
   {
      return 1.0/(1.0 + Math.exp(-netInput));
   }

   /*
    * Applies the derivative of the activation function for
    * a given net input. f'(x)
    *
    * @param  netInput the value to be manipulated by f'(x)
    *
    * @return the value of f'(x).
    */
   public double derivActivationFunction(double netInput)
   {
      double activationVal = activationFunction(netInput);

      return activationVal *(1.0-activationVal);
   }

   /*
    * Loops through the neural network to produce the final outputs,
    * Also saves the theta values for training.
    *
    * @precondition  the input values and weight values have already been inputted.
    * @postcondition the activation 2D array is filled, theta values are saved
    *
    * @return the array of final outputs.
    */
   public double[] findOutputs()
   {
      for (int layer = 1; layer < numLayers; layer++)
      {
         for (int row = 0; row < activations[layer].length; row++)
         {
            double netInput = propagationFunction(layer, row);

            thetaVals[layer][row] = netInput;
            activations[layer][row] = activationFunction(netInput);
         }
      }

      return activations[numLayers-1];
   }

   /*
    * Calculates the total error of this case over all outputs.
    *
    * @precondition the final output has already been determined.
    *
    * @return the calculated error.
    */
   public double errorCalculation()
   {
      double[] finalOutputs = findOutputs();

      double totalError = 0.0;

      for (int row = 0; row<numOutputs; row++)
      {
         double currError = 0.5*((trainingValues[row]-finalOutputs[row])*(trainingValues[row]-finalOutputs[row]));
         totalError += currError*currError;
      }

      return Math.sqrt(totalError);
   }

   /*
    * Trains the network by running steepest descent and back propagation to recalculate each weight.
    *
    * Terminates when:
    * maxError < threshold, or
    * lambda = 0, or
    * change in error <= threshold, or
    * number of iterations > threshold
    *
    * @postcondition: error for this network is minimized.
    *
    * @param errorThreshold     the threshold for the max error over all test cases.
    * @param iterationThreshold threshold for number of iterations of this function.
    * @param changeThreshold    threshold for change in error, function terminates when change in error is below it.
    * @param initError          the initial error from the first test case with randomized weights.
    * @param inputVals          the input values for each test case.
    * @param trainingVals       the target values for each test case.
    * @param numTestCases       the number of test cases to go through.
    * @param outputFile         to print out termination conditions.
    *
    * @return the termination message, tells user why training terminated.
    */
   public String training(double errorThreshold, int iterationThreshold, double changeThreshold, double initError,
                                 double[][] inputVals, double[][] trainingVals, int numTestCases, PrintWriter outputFile)
   {
      int iterations     = 0;
      double errorChange = 0.0;
      double prevError   = initError;
      double maxError    = initError; //the max error over all test cases; if below the threshold, training can stop.

      while (iterations == 0 || (maxError>errorThreshold && iterations < iterationThreshold &&
              learningFactor!=0.0 && errorChange>changeThreshold))
      {
         for (int t = 0; t < numTestCases; t++) //iterate through each test case
         {
            //run network for this test case:
            activations[0] = inputVals[t];
            trainingValues = trainingVals[t];
            findOutputs();

            //recalculate weight values: output layer
            double[][] seniorOmegaVals = new double[numLayers][numRows];

            for (int j = 0; j < activations[numLayers-2].length; j++)
            {
               for (int i = 0; i < numOutputs; i++)
               {
                  double fI = activationFunction(thetaVals[numLayers-1][i]);
                  double juniorOmegaI = trainingValues[i] - fI;
                  double psiI = juniorOmegaI * derivActivationFunction(thetaVals[numLayers-1][i]);

                  weights[numLayers-2][j][i] += learningFactor * activations[numLayers-2][j] * psiI;

                  seniorOmegaVals[numLayers-2][j] += psiI * weights[numLayers-2][j][i];
               }
            }

            //recalculate weight values: second hidden layer
            for (int k = 0; k < activations[1].length; k++)
            {
               for (int j = 0; j < activations[numLayers-2].length; j++)
               {
                  double psiJ = seniorOmegaVals[numLayers-2][j] * derivActivationFunction(thetaVals[numLayers-2][j]);

                  weights[1][k][j] += learningFactor * activations[1][k] * psiJ;

                  seniorOmegaVals[1][k] += psiJ * weights[1][k][j];
               }
            }

            //recalculate weight values: first hidden layer
            for (int m = 0; m < numInputs; m++)
            {
               for (int k = 0; k < activations[1].length; k++)
               {
                  double psiK = seniorOmegaVals[1][k] * derivActivationFunction(thetaVals[1][k]);

                  weights[0][m][k] += learningFactor * activations[0][m] * psiK;
               }
            }

            //run network with new weights and calculate new error:
            findOutputs();
            double newError = errorCalculation();

            errorChange = Math.abs(newError-prevError);

            //compare new error w/ previous error:
            if (newError<=prevError)
            {
               prevError = newError;
            }
            else
            {
               if (newError>errorThreshold)
               {
                  t--;
               }
            }
         } //for (int t = 0; t < numTestCases; t++)

         //find max error over all test cases using current weights:
         for (int t = 0; t < numTestCases; t++)
         {
            activations[0] = inputVals[t];
            trainingValues = trainingVals[t];
            double curError = errorCalculation();
            if (t==0 || curError > maxError)
            {
               maxError = curError;
            }
         }

         iterations++;
      } //while (iterations == 0 || (maxError>errorThreshold && iterations < iterationThreshold &&
        // learningFactor!=0.0 && errorChange>changeThreshold))


      //determine the termination condition that was met and print it for the user:
      outputFile.println("Max error over all test cases: "+maxError);
      outputFile.println("Iterations: "+iterations);
      if (iterations==iterationThreshold)
      {
         outputFile.println("Threshold: "+iterationThreshold);

         return "Terminated because iterations exceeded threshold.";
      }
      else if (maxError<=errorThreshold)
      {
         outputFile.println("Error Threshold: "+errorThreshold);

         return "Terminated because error is adequately minimized, reached threshold.";
      }
      else
      {
         outputFile.println("Change in error: "+errorChange);
         outputFile.println("Threshold: "+changeThreshold);

         return "Terminated because change in error is too small, below threshold.";
      }

   }

   /*
    * The main method that runs the Perceptron.
    */
   public static void main(String[] args) throws IOException
   {
      Scanner scan = new Scanner(System.in);

      //System.out.println("Please input the name of the control file: ");
      //String controlFile = scan.nextLine();
      String controlFile = "input2.in";
      //System.out.println("Please input the name of the output file: ");
      //String outputFile = scan.nextLine();
      String outputFile = "output.out";

      BufferedReader in = new BufferedReader(new FileReader(controlFile));
      PrintWriter out2 = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)));

      int numInputs  = Integer.parseInt(in.readLine());         //read in number of inputs
      int numOutputs = Integer.parseInt(in.readLine());         //read in number of outputs
      int numHidden  = Integer.parseInt(in.readLine());         //read in number of hidden layers

      int[] hiddenLayerNodes = new int[numHidden];

      StringTokenizer stk = new StringTokenizer(in.readLine()); //read in number of activations in each hidden layer
      for (int n = 0; n < numHidden; n++)
      {
         int numAct = Integer.parseInt(stk.nextToken());

         hiddenLayerNodes[n] = numAct;
      }

      double errorThreshold  = Double.parseDouble(in.readLine());
      int iterationThreshold = Integer.parseInt(in.readLine());
      double changeThreshold = Double.parseDouble(in.readLine());
      int numTestCases       = Integer.parseInt(in.readLine());
      double initialLambda   = Double.parseDouble(in.readLine());
      double weightsLow      = Double.parseDouble(in.readLine());
      double weightsHigh     = Double.parseDouble(in.readLine());

      PerceptronOptimizedMultOutput testNet = new PerceptronOptimizedMultOutput(numInputs, numOutputs, hiddenLayerNodes,
              initialLambda, weightsLow, weightsHigh);

      //System.out.println("Please type in the name of the input activation file: ");
      //String inputsFile = scan.nextLine();
      String inputsFile = "inputActivations.in";

      in = new BufferedReader(new FileReader(inputsFile));
      scan = new Scanner(in);

      double[][] inputVals = new double[numTestCases][numInputs];
      for (int t = 0; t < numTestCases; t++)
      {
         for (int curInputIndex = 0; curInputIndex<numInputs; curInputIndex++)
         {
            inputVals[t][curInputIndex] = scan.nextDouble();
         }
      }

      //System.out.println("Please type in the name of the target values file: ");
      //scan = new Scanner(System.in);
      //String targetsFile = scan.nextLine();
      String targetsFile = "targetValues.in";

      in = new BufferedReader(new FileReader(targetsFile));
      scan = new Scanner(in);

      double[][] trainingVals = new double[numTestCases][numOutputs];
      for (int index = 0; index < numOutputs; index++)
      {
         for (int t = 0; t < numTestCases; t++)
         {
            trainingVals[t][index] = scan.nextDouble();
         }
      }

      //use the first test case to calculate an initial error:
      testNet.activations[0] = inputVals[0];
      testNet.trainingValues = trainingVals[0];
      testNet.findOutputs();
      double initError = testNet.errorCalculation();

      String terminationReason=testNet.training(errorThreshold, iterationThreshold, changeThreshold, initError, inputVals,
              trainingVals, numTestCases, out2);

      out2.println(terminationReason+"\n");

      //print out final outputs and errors to given output file:
      for (int t = 0; t < numTestCases; t++)
      {
         testNet.activations[0] = inputVals[t];
         testNet.trainingValues = trainingVals[t];

         out2.println("Outputs for test case "+t+": ");

         double[] outputs = testNet.findOutputs();
         for (double outputVal: outputs)
         {
            out2.println(outputVal);
         }
         out2.print("\n");

         out2.println("Error for test case "+t+": "+"\n"+testNet.errorCalculation());

         out2.println("\n");
      }

      out2.close();
   }
}