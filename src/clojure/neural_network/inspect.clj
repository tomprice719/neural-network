(ns neural-network.inspect)
(use 'neural-network.utils)
(use 'neural-network.draw)
(use 'neural-network.backprop)
(use 'neural-network.training)

;;Functions to see what's going on in a neural network

(defn print-piece [piece]
  (doseq [p piece] (println (.toString p))))

(defn print-weights [[weights biases]]
  (print-piece weights))

(defn print-biases [[weights biases]]
  (print-piece biases))

(defn print-network [network]
  (print-weights network)
  (print-biases network))

(defn guess [input weights biases]
  "Feeds the input 'input' into the network specified by the given weights and biases. Returns the activations of the final layer."
  (last (infer input weights biases)))

(defn perform [[weights biases]]
  "See the output the network generates when given a random MNIST test sample."
  (let [[pixels labels] (test-sample)]
    (println labels)
    (println (guess pixels weights biases))))

(defn test-accuracy [[weights biases]]
  "See how many times the network guesses correctly in 10000 tries."
  (nth-iteration 10000
                 #(+ %
                     (let [jSample (test-jsample)]
                       (if (.checkGuess jSample
                                        (guess (.pixels jSample) weights biases))
                         1 0)))
                 0))

(defn draw-weights [[weights biases]]
  "Makes visual representations of the hidden-layer weights in the provided network."
  (doseq [cindex (range 30)]
    (draw-vect (.getColumn (first weights) cindex) 28 28 (str "images/weights" cindex ".png"))))

(defn improved-input [[weights biases] target input]
  "Returns an input vector that is closer to being classified as 'target' by the network. Used by recreate-input."
  (let [a (infer input weights biases)
        bg (first (bias-gradient target weights a))
        ig  (.multiply bg (.transpose (first weights)))] ;;input gradient
    (.multiply (.add input (.multiply ig -0.1)) 0.99)))

(defn recreate-input [network target]
  "Uses backpropogation / gradient descent to create an input vector that the network classifies as 'target'."
  (let [jtarget (make-jvector target)] ;; convert target from clojure vector to la4j vector
    (nth
     (iterate (partial improved-input network jtarget) (random-vector (* 28 28)))
     10000)))

(defn perform2 [[weights biases] labels]
  "See how the network classifies an artificial input."
  (let [pixels (recreate-input [weights biases] labels)]
    (println labels)
    (println (last (infer pixels weights biases)))))
