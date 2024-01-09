using Accord.DataSets;
using Accord.Math;
using System.Runtime.CompilerServices;

namespace TestDataSets
{
    //class CreateMNISTData
    // Modified National Institute of Standards and Technology database of handwritten digits
    /* The MNIST database contains 60,000 training images and 10,000 testing images.Half of the 
     * training set and half of the test set were taken from NIST's training dataset, while the 
     * other half of the training set and the other half of the test set were taken from NIST's 
     * testing dataset.[9] The original creators of the database keep a list of some of the 
     * methods tested on it.[7] In their original paper, they use a support-vector machine to 
     * get an error rate of 0.8% 
     * 
     * Example of MNIST usage https://github.com/accord-net/framework/issues/1089
      
                    
    */


    class MNISTDataset
    {
        public double[][]? TrainingData { get; }
        public int[] Labels { get; }
        public MNISTDataset(in string _downloadedFilename = "\\Data")
        {
            // Setup destination for downloaded files
            string CurrentDir = System.IO.Directory.GetCurrentDirectory();
            string TargetDir = CurrentDir + _downloadedFilename;
            Directory.CreateDirectory(_downloadedFilename);
            // Download the MNIST dataset to a temporary dir:
            var mnist = new MNIST(path: TargetDir);


            // Get the training inputs and expected outputs:
            Sparse<double>[] xTrain = mnist.Training.Item1;
            Labels = mnist.Training.Item2.ToInt32();
            TrainingData = xTrain.ToDense<double>();

        }
    }
    class IrisData
    {
        public double[][] TrainingData { get { return _trainingData; } }
        public int[] Labels { get { return _labels; } }
        private double[][] _trainingData;
        private int[] _labels;
        public IrisData(in string _downloadedFilename = "DownLoadData")
        {
            
            // Download and load the Iris dataset
            Iris iris = new Iris();
            _trainingData = iris.Instances;
            _labels = iris.ClassLabels;
        }
    }
}
