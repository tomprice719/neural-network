A basic neural network for classifying handwritten digits from the MNIST database.
Trains a random neural network with the gradient descent / backpropogation algorithm.
Loosely based on the neural network described in this tutorial: http://neuralnetworksanddeeplearning.com/
By default, uses one hidden layer with 30 neurons.
Uses weight decay and a cross-entropy cost function (these are explained in the tutorial in chapter 3).

This requires the training and test csv files from here: http://www.pjreddie.com/projects/mnist-in-csv/
They are larger than the github max file size limit. Put them in the resources folder.

I'll give a very brief overview here of how feedforward neural networks work in case you're not familiar with them, but I recommend reading the neuralnetworksanddeeplearning.com tutorial which gives a much better and more thorough explanation than mine.

A feedforward neural network consists of one or more layers of artificial neurons, stacked on top of an input layer.
Each neuron in each layer takes the outputs of the previous layer, processes them, and feeds that into the next layer.
Specifically, it takes a weighted sum of the outputs in the previous layer, adds a "bias" to that value, and applies a function (the "activation function") to the result. For the network here, the activation function is the logistic sigmoid function.

The first layer of neurons is a bit different in that it doesn't process the outputs of other neurons. Instead, it processes data that is supplied to the neural network. In the case of this neural network, that would be the brightness values of the pixels in an image of a handwritten digit from the MNIST database.

The output of the last layer is interpreted as the neural network's classification of the input. For this neural network, the last layer consists of 10 neurons, each of which corresponds to one of the numerals 0 - 9. The networks's classification of the input is the corresponding numeral of whichever neuron in the final layer has the largest output value.

In order for the neural network to classify inputs accurately, you need to determine a good set of weights and biases for each layer. In other words, you need to train the network.

You can imagine a cost function which takes as input a set of weights and biases for the neural network, and outputs the average error of the network when it uses these weights and biases. The goal of training then is to find a set of weights and biases that minimizes this cost function. This is done by gradient descent.

The idea of gradient descent is to find a local minimum of a function by starting at a random point, and repeatedly moving a small amount in the direction of the negative gradient of the function. The gradient is a vector calculus operation; it is a vector where each coordinate is the partial derivative in that direction, and it always points in the direction that the function increases most quickly. The negative of the gradient is the direction in which the function decreases most quickly, and if you keep moving in that direction you are eventually going to get to a local minimum of the function. You may not arrive at a global minimum, but in practice the local minima tend to be pretty good with neural networks.

The gradient of the cost function is determined with the backpropogation algorithm. The backpropogation algorithm lets you determine the gradient of the error of the neural network's classification of an individual sample, and by averaging this over a large number of samples (a "batch") you can get an approximation of the gradient of the cost function.

Briefly, this is how the backpropagation algorithm determines the gradient of the error for an individual sample:
For each layer you can determine how much each weight and bias effects the error if you already have this information for the next layer, basically by applying the chain rule for derivatives. So starting at the top layer and moving down, you can figure out the gradient for each layer. A much better explanation is given in the neuralnetworksanddeeplearning.com tutorial.

The backpropogation algorithm is fairly computationally intensive. However, it can mostly be expressed in terms of vector and matrix operations, and so I've avoided doing most of the computation in Clojure by using a Java linear algebra library, la4j. The weights and biases of the neural network are represented as la4j matrices and vectors, and I'm able to do most of the backpropogation computations by doing operations like taking sums and products of vectors and matrices, as well as some other operations like hadamard products, outer products, and applying the activation function to each coordinate of a vector.

Here is an example of basic usage of this neural network:

(use 'neural-network.training)

(use 'neural-network.inspect)

(init-mnist) ;;loads MNIST samples from CSV files into memory

(def trained (train (random-network))) ;;Trains a random network. This will take a few minutes.

(test-accuracy trained) ;; Shows you how many times it correctly classifies an MNIST digit in 10000 trials.

The "train" function shown above takes a few optional parameters, specifically batch-size (this is the number of samples in a batch, as explained above), num-batches (the number of batches used to train the network), learning-rate (this determines the size of the steps you take in gradient descent), and weight-decay (after each batch, the weights in the network get multiplied by this constant).
For example, if you are impatient and want the network to train more quickly, you can use a smaller number of batches (the default is 10000):

(def trained (train (random-network) :num-batches 1000))

This will have an effect on the accuracy of the network, however.

You can also use the "perform" function to see an example of the outputs of the network given a random MNIST sample.
For example, (perform trained) may print the following to the console:

\#\<BasicVector [ 0.000,  0.000,  0.000,  0.000,  1.000,  0.000,  0.000,  0.000,  0.000,  0.000 ]\>

\#\<BasicVector [ 0.000,  0.199,  0.034,  0.001,  0.215,  0.007,  0.005,  0.003,  0.014,  0.013 ]\>

The first vector is the label of the randomly chosen MNIST sample. It is the target that the neural network is trying to guess. The 1.0 in the 5th column means that the randomly chosen sample was the 5th of the digits 0 - 9, in other words, a 4. The second vector is the outputs of the neural network when it is fed the pixels in the image of the 4. As you can see, the output is highest in the 5th column, which means the network guessed correctly. There is also a relatively large value in the 2nd column, which means that it was ambiguous to the network whether the digit was a 1 or a 4. But it still gave the larger value for the 4 and therefore guessed correctly.

You can also see a visualization of the weights for each neuron in the hidden layer:

(draw-weights trained)

This will draw 30 png files into the images folder, one image for each neuron in the hidden layer. Each pixel represents the weight of the connection from that neuron to that pixel in the input images. The red pixels represent negative weights, the blue pixels represent positive weights, and the brightness represents the magnitude of the weight.

One thing I realized while making this neural network, that was not covered in the tutorial, is that you can use gradient descent / backpropogation not just to train a neural network to classify inputs accurately, but also to train an input to be classified by the network as a certain target. So for example, we can use gradient descent and backpropogation to create an image that the network classifies as a "3". One way to interpret this image is what the network "thinks" a 3 looks like.

Here is an example of how to do that:

;;Using a lower weight-decay gives worse classification accuracy but makes for better recreated inputs.

(def trained (train (random-network) :weight-decay 0.99)) 

;;Make an la4j vector of a recreated 3 (the provided vector is the target output)

(def ri (recreate-input trained [0 0 0 1 0 0 0 0 0 0]))

;;Draw the recreated input to a 28 by 28 png image

(neural-network.draw/draw-vect ri 28 28 "images/recreated.png")

You can now find a recreated 3 in the images folder.






