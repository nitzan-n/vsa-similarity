(ns org.nn.similarity
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]))

;-------------------- helper functions --------------------------

(defn as-comparable
  [v]
  (if (instance? Comparable v)
    v
    (into [] v)))

(defn sorted-map-by-values
  [m]
  (into (sorted-map-by (fn [key1 key2]
                         (cond
                           (= key1 key2)
                           0

                           (and (= (class (m key1)) (class (m key2)))
                                (= (class key1) (class key2)))
                           (compare [(as-comparable (m key2)) (as-comparable key2)]
                                    [(as-comparable (m key1)) (as-comparable key1)])

                           (= (class (m key1)) (class (m key2)))
                           (compare [(as-comparable (m key2)) (.getName (class key2))]
                                    [(as-comparable (m key1)) (.getName (class key1))])

                           (= (class key1) (class key2))
                           (compare [(.getName (class (m key2))) (as-comparable key2)]
                                    [(.getName (class (m key1))) (as-comparable key1)])

                           :default
                           (compare [(.getName (class (m key2))) (.getName (class key2))]
                                    [(.getName (class (m key1))) (.getName (class key1))]))))
        m))


;------------------- hyper-dimensional binary vectors -----------------------

(m/set-current-implementation :vectorz)

(def sz 100000)
(def sparsness 0.1)


(defn rand-hv
  "generate a random hyper-dimensional vector"
  []
  (let [hv (m/new-sparse-array [sz])
        n (* sparsness sz)]
    (dotimes [i n]
      (m/mset! hv (rand-int sz) 1))
    hv))

(defn mean-add [& hvs]
  "'add' vectors: combines vectors' elements but preserves similarity to each original vector.
  element-wise one/zero count majority."
  (m/emap #(Math/round (double %))
          (m/div (apply m/add hvs) (count hvs))))

(defn xor-mul [v1 v2]
  "'multiply' vectors: randomizes vectors' elements but preserves distance between vectors randomized with same multiplicand.
  element-wise xor."
  (m/emap #(mod % 2) (m/add v1 v2)))

(defn hamming-dist
  "calculate distance between 2 vectors"
  [v1 v2]
  (/ (m/esum (xor-mul v1 v2)) sz))

(defn cosine-sim [v1 v2]
  "calculate similarity between 2 vectors"
  (/ (m/dot v1 v2)
     (* (ml/norm v1) (ml/norm v2))))

(def distance-fn cosine-sim) ;hamming-dist

;----------------- associative memory implementation ------------------

(defn get-item
  "get vector encoding in the DB of item x"
  [db x]
  (get-in db [:index x]))

(defn add-vector
  [db val-vec]
  (assoc db :vecs (conj (:vecs db) val-vec)))

(defn insert-item
  "update DB with item and its vector"
  [db new-val]
  (if (get-item db new-val)
    db
    (let [new-val-vec (rand-hv)]
      (-> db
          (assoc-in [:index new-val] new-val-vec)
          (add-vector new-val-vec)))))

(defn update-db
  "update DB with vectors of all data objects"
  [db]
  (assoc db :db (apply mean-add (:vecs db))))

;-------------------- associative memory interface ---------------------------

(defn encode-item
  [db x]
  (insert-item db x))

(defn encode-assoc-entry
  ([db [k v]]
   (let [updated-db (-> db (insert-item k) (insert-item v))]
     (add-vector updated-db (xor-mul (get-item updated-db k) (get-item updated-db v)))))
  ([db k v]
   (let [updated-db (-> db (insert-item k) (insert-item v))]
     (add-vector updated-db (xor-mul (get-item updated-db k) (get-item updated-db v))))))

(defn encode
  "encode items into the DB"
  [item]
  (update-db
    (cond
      (or (keyword? item) (string? item) (number? item) (boolean? item))
      (encode-item {} item)

      (map? item)
      (reduce encode-assoc-entry {} item)

      (or (vector? item) (list? item))
      (reduce encode-assoc-entry {} item))))

(defn compare-using
  "calculate similarity between data objects from memory"
  [db x y]
  (let [v-x (get-item db x)
        v-y (get-item db y)]
    (if (and v-x v-y)
      (distance-fn v-x v-y)
      0.0)))

(defn test-using
  "get data objects from the memory that are closely associated with given data object x"
  [db x]
  (let [v-x (get-item db x)
        db-vector (db :db)]
    (if (and v-x db-vector)
      (let [unbound (xor-mul v-x db-vector)]
        (sorted-map-by-values
          (into {}
                (comp
                  (filter #(not= x (key %)))
                  (map #(vector (key %) (distance-fn (val %) unbound))))
                (:index db))))
      nil)))

(comment
  (let [db (encode {:x 1 :y 2 :z 5 "hello" "world"})]
    (test-using db :x))

  (let [db (encode [[:x 1] [:y 2] [:z 5] ["hello" "world"]])]
    (test-using db :x))

  (let [db (encode [[:x 1] [:y 2] [:z 5] [:x 4]])]
    (test-using db :x)))