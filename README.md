A basic neural network for classifying handwritten digits from the MNIST database.
Trains a random neural network with the gradient descent / backpropogation algorithm.
Loosely based on the neural network described in this tutorial: http://neuralnetworksanddeeplearning.com/

This requires the training and test csv files from here: http://www.pjreddie.com/projects/mnist-in-csv/
They are larger than the github max file size limit. Put them in the resources folder.

An example of basic usage:

(use 'neural-network.training)
(use 'neural-network.inspect)

(init-mnist) ;;loads MNIST samples from CSV files into memory
(def trained (train (random-network))) ;;Trains a random network. This will take a few minutes.

(test-accuracy trained) ;; Shows you how many times it correctly classifies an MNIST digit in 10000 trials.

