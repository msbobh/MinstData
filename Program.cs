// See https://aka.ms/new-console-template for more information


using Accord.DataSets;
using Accord.Statistics.Analysis;
using TestDataSets;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning;

Console.WriteLine("Getting MNIST Dataset");

var MNISTData = new MNISTDataset();


// Generate always same random numbers
Accord.Math.Random.Generator.Seed = 0;

// Let's say we would like to learn a classifier for the famous Iris
// dataset, and measure its performance using a GeneralConfusionMatrix

// Download and load the Iris dataset and configure a model and learning algorithm
var iris = new Iris();
double[][] inputs = iris.Instances;
int[] outputs = iris.ClassLabels;
Console.WriteLine("Creating IRIS data");
// Create the multi-class learning algorithm for the machine
var teacher = new MulticlassSupportVectorLearning<Linear>()
{
    // Configure the learning algorithm to use SMO to train the
    //  underlying SVMs in each of the binary class subproblems.
    Learner = (param) => new SequentialMinimalOptimization<Linear>()
    {
        // If you would like to use other kernels, simply replace
        // the generic parameter to the desired kernel class, such
        // as for example, Polynomial or Gaussian:

        Kernel = new Linear() // use the Linear kernel
    }
};
Console.WriteLine("Training Model with IRIS Data");
// Estimate the multi-class support vector machine using one-vs-one method
MulticlassSupportVectorMachine<Linear> ovo = teacher.Learn(inputs, outputs);

// Compute classification error
GeneralConfusionMatrix cm = GeneralConfusionMatrix.Estimate(ovo, inputs, outputs);
Console.WriteLine(" Error = {0}", cm.Error);
Console.WriteLine(" Accuracy = {0}", cm.Accuracy);
Console.WriteLine(" Kappa = {0}", cm.Kappa);
Console.WriteLine(" ChiSquare = {0}", cm.ChiSquare);
double error = cm.Error;         // should be 0.066666666666666652
double accuracy = cm.Accuracy;   // should be 0.93333333333333335
double kappa = cm.Kappa;         // should be 0.9
double chiSquare = cm.ChiSquare; // should be 248.52216748768473 

// Done processing the IRIS dataset
Console.WriteLine("Training model with MNIST data");



double[][] Minputs = MNISTData.TrainingData;
int[] Moutputs = MNISTData.Labels;

// We will use mini-batches of size 32 to learn a SVM using SGD
// var batches = MiniBatches.Create(batchSize: 32, maxIterations: 1000,
// shuffle: ShuffleMethod.EveryEpoch, input: Minputs, output: Moutputs);
// *********************************


// Now, we can create a multi-label teaching algorithm for the SVMs
/*var Mteacher = new MultilabelSupportVectorLearning<Linear, double[]>
{
    // We will use SGD to learn each of the binary problems in the multi-class problem
    Learner = (p) => new StochasticGradientDescent<Linear, double[], LogisticLoss>()
    {
        LearningRate = 1e-3,
        MaxIterations = 1 // so the gradient is only updated once after each mini-batch
    }
};

Mteacher.Learn(Minputs, Moutputs);*/

// Now, we can start training the model on mini-batches:
/*foreach (var batch in batches)
{
    teacher.Learn(batch.Inputs, batch.Outputs);
}*/
//******************************
var smo = new SequentialMinimalOptimization<Spline>()
{
    // Force a complexity value C or let be 
    // determined automatcially by a heuristic
    // Complexity = 1.5
};
var SMOsvm = smo.Learn(Minputs, Moutputs);

bool[] predicted = SMOsvm.Decide(Minputs);
double Merror = new ZeroOneLoss(Moutputs).Loss(predicted);

// Get the final model:
var svm = teacher.Model;

// Now, we should be able to use the model to predict 
// the classes of all flowers in Fisher's Iris dataset:
int[] prediction = svm.ToMulticlass().Decide(Minputs);

// And from those predictions, we can compute the model accuracy:
cm = new GeneralConfusionMatrix(expected: Moutputs, predicted: prediction);
double Maccuracy = cm.Accuracy; // should be approximately 0.913

GeneralConfusionMatrix MCM = GeneralConfusionMatrix.Estimate(svm, Minputs, Moutputs);
Console.WriteLine(" Error = {0:p2}", MCM.Error);
Console.WriteLine(" Accuracy = {0:p2}", MCM.Accuracy);
Console.WriteLine(" Kappa = {0}", MCM.Kappa);
Console.WriteLine(" ChiSquare = {0}", MCM.ChiSquare);

