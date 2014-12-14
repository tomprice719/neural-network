(ns neural-network.training)
(use 'neural-network.backprop)
(use 'neural-network.utils)
(use 'neural-network.draw)

;;Stuff for training a neural network
;;Neural networks are represented here as a vector [weights biases],
;;where "weights" is a sequence of la4j matrices representing the weights at each layer,
;;and "biases" is a sequence of la4j vectors representing the biases at each layer

(defn init-mnist []
  (mnist/initialize))

(def num-pixels (* 28 28))

(defn add-networks [network1 network2]
  "Add together all weights and biases in two networks"
  (map #(map (memfn add v) %1 %2)
       network1 network2))

(defn scale-network [network scalar]
  "Multiplies all weights and biases in a network by a given scalar"
  (map (fn [piece]
         (map #(.multiply % scalar) piece))
       network))

(defn random-network
  "Creates a network with random weights and biases, for the beginning of training."
  ([layer-sizes]
   [(map random-matrix (partition 2 1 layer-sizes))
    (map random-vector (rest layer-sizes))])
  ([] (random-network [num-pixels 30 10])))

(defn training-sample []
  "Gives a random sample from the MNIST training data, represented as a vector [pixels labels]."
  (let [jSample (mnist/getTraining)]
    [(.pixels jSample)
     (.labels jSample)]))

(defn test-jsample []
  "Gives a random sample from the MNIST test data, represented as a Sample object."
  (mnist/getTest))

(defn test-sample []
   "Gives a random sample from the MNIST test data, represented as a vector [pixels labels]."
  (let [jSample (test-jsample)]
    [(.pixels jSample)
     (.labels jSample)]))

(defn batch-gradient [network batch-size learning-rate]
  "Determines how to shift the network weights and biases based on how it performs in a batch."
  (scale-network
   (reduce add-networks
           (repeatedly batch-size
                       #(backprop network (training-sample))))
   (/ (- learning-rate) batch-size)))

(defn train [network
             & {:keys [num-batches batch-size learning-rate weight-decay]
                :or {num-batches 10000 batch-size 30 learning-rate 1.0 weight-decay 0.999}}]
  "Does gradient descent starting with network 'network'."
  (nth-iteration num-batches
                 #(scale-network
                   (add-networks % (batch-gradient % batch-size learning-rate))
                   weight-decay)
                 network))





