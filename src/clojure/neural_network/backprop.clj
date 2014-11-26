(ns neural-network.backprop)
(use 'neural-network.utils)

;;Functions to perform the backpropogation algorithm

(def act-function (LogSig.)) ;;la4j transformer for activation function.
(def act-derivative (LogSigDerivative.)) ;;Another la4j transformer. Suppose activation function is f(x). This outputs f'(x) given f(x).

(defn infer [input weights biases]
  "Feeds the input 'input' into the network specified by the given weights and biases. Returns a list of the activations at every layer."
  (reductions
   (fn [i [w b]]
     (.transform (.add (.multiply i w) b) act-function))
   input (zip weights biases)))

(defn top-gradient [target activations]
  "Computes the gradient of the cost function as a function of the top-layer biases."
  (.subtract (last activations) target))

(defn bias-gradient [target weights activations]
  "Computes the gradient of the cost function as a function of the biases."
  (reverse
   (reductions
    (fn [d [weights activations]]
      (.hadamardProduct
       (.multiply d (.transpose weights))
       (.transform activations act-derivative)))
    (top-gradient target activations)
    (zip (-> weights rest reverse)
            (-> activations rest reverse rest)))))

(defn weight-gradient [bg activations]
  "Computes the gradient of the cost function as a function of the weights. Requires bias gradient and activations."
  (map (memfn outerProduct v) activations bg))

(defn backprop [[weights biases] [input target]]
  "Computes the gradient of the cost function as a function of the biases and weights."
  (let [activations (infer input weights biases)
        bg (bias-gradient target weights activations)
        wg (weight-gradient bg activations)]
  [wg bg]))
