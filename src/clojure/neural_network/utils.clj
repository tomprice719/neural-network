(ns neural-network.utils)

(import org.la4j.matrix.dense.Basic2DMatrix)
(import org.la4j.vector.dense.BasicVector)
(import java.util.Random)
(import org.la4j.vector.functor.VectorFunction)

;;Utility functions

(def jrand (Random.))

(defn gaussian
  "Ignores arguments and generates random double with Gaussian distribution, with mean 0 and variance 1."
  [& args]
  (.nextGaussian jrand))

(defn multi-get [c & indices]
  "Gets an item from the multidimensional vector 'c' at location determined by 'indices'"
  (reduce get c indices))

(defn nth-iteration [n f x]
  "Result of iterating f n times, starting with x."
  (nth (iterate f x) n ))

(defn dims [c]
  "Returns the dimensions of a multidimensional vector."
  (map count
       (take-while coll?
                   (iterate #(if (coll? %) (first %))
                            c))))

(defn make-matrix
  ([dims f] "Makes an la4j matrix with the specified dimensions, from a function that outputs matrix entries given the row and column"
   (let [[h w] dims
         matrix (Basic2DMatrix. h w)]
     (doseq
       [y (range h) x (range w)]
       (.set matrix y x (f y x)))
     matrix))
  ([c] "Makes a matrix from a given multidimensional vector"
   (make-matrix (dims c) (partial multi-get c))))

(defn random-matrix [dims]
  "Returns a random matrix with dimensions given by 'dims' and Gaussian entries."
  (make-matrix dims gaussian))

(defn make-jvector [c]
  "Makes an la4j BasicVector with entries provided by c"
  (BasicVector. (into-array Double/TYPE c)))

(defn random-vector [length]
  "Makes an la4j BasicVector with random entries."
  (make-jvector (repeatedly length gaussian)))

(defn zip [& colls]
  "Analogous to Python zip. Returns a collection where the first item is a vector of all the first items of colls,
  the second item is a vector of the second items, etc."
  (apply map vector colls))

(defn transform [vector f]
  "Replaces each entry of la4j vector 'vector' with (f index entry)"
  (.transform vector
              (proxy [VectorFunction] []
                (evaluate [arg0 arg1] (f arg0 arg1)))))

(defn dovect [vector f]
  "Calls (f index entry) for side effects for each entry of la4j vector 'vector'."
  (transform vector #(do (f %1 %2) %2)))

(defn to-cvect [jvect]
  "Converts an la4j vector to a Clojure vector."
  (let [cvect (transient [])]
    (dovect jvect #(conj! cvect %2))
    (persistent! cvect)))

(defn vect-abs [v]
  "Applies abs to elements of v"
  (transform v #(Math/abs %2)))

(defn coords [i width]
  "Converts an index in a one-dimensional array representing two-dimensional data into a pair of coordinates."
  [(quot i width) (mod i width)])
