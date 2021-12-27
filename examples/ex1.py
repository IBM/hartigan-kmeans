#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import os
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

from hkmeans import HKMeans
from clustering_utils import fetch_20ng, save_report_and_heatmap


# This example compares Scikit Learn's Lloyd's K-Means to the Hartigan's K-Means
# delivered in this distribution. We will use  the 20 News Groups dataset as a
# benchmark (about 19K docs, 20 clusters).

# step 0 - create an output directory if it does not exist
output_path = os.path.join("output", "ex1")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# step 1 - read the dataset
texts, gold_labels_array, n_clusters, topics, n_samples = fetch_20ng('all')
print("Clustering dataset contains %d texts from %d topics" % (n_samples, n_clusters))

# The following settings are meant for comparison purposes and should be adjusted
# based on the real-world use-case.
# The default for Lloyd's K-Means in sklearn is n_init=10, max_iter=300;
# For Hartigan's K-Means it is enough to use max_iter=15.
# Here we use max_iter=15 for both to be able to compare run-time
# We set kmeans algorithm to 'full' to apply lloyd's k-means
n_init = 10
max_iter = 15
setups = [
    ("Scikit-Learn Lloyd's K-Means", lambda: KMeans(n_clusters=n_clusters, n_init=n_init,
                                                    max_iter=max_iter, algorithm='full')),
    ("Hartigan's K-Means", lambda: HKMeans(n_clusters=n_clusters, n_init=n_init,
                                           max_iter=max_iter))
]

# step 2 - represent the clustering data using bow of the 10k most frequent
# unigrams in the dataset, excluding stop words. Note that if you wish to
# apply some text pre-processing like stemming - that's the place to do that.
print("Vectorization starts...", end=' ')
vectorization_start_t = time()
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
vectors = vectorizer.fit_transform(texts)
vectorization_end_t = time()
print("ended in %.3f secs." % (vectorization_end_t - vectorization_start_t))
print("Clustering settings: n_init=%d, max_iter=%d:" % (n_init, max_iter))
for algorithm_name, factory in setups:
    print("Running with %s:" % algorithm_name)

    # step 3 - cluster the data
    print("\tClustering starts...", end=' ')
    clustering_start_t = time()
    algorithm = factory()
    algorithm.fit(vectors)
    clustering_end_t = time()
    print("ended in %.3f secs." % (clustering_end_t - clustering_start_t))

    predictions_array = algorithm.labels_

    # measure the clustering quality
    homogeneity = metrics.homogeneity_score(gold_labels_array, predictions_array)
    completeness = metrics.completeness_score(gold_labels_array, predictions_array)
    v_measure = metrics.v_measure_score(gold_labels_array, predictions_array)
    ami = metrics.adjusted_mutual_info_score(gold_labels_array, predictions_array)
    ari = metrics.adjusted_rand_score(gold_labels_array, predictions_array)
    print("\tClustering measures: AMI: %.3f, ARI: %.3f" % (ami, ari))

    save_report_and_heatmap(gold_labels_array, predictions_array, topics,
                            algorithm, algorithm_name, output_path,
                            ami, ari, homogeneity, completeness, v_measure,
                            n_samples, vectorization_end_t-vectorization_start_t,
                            clustering_end_t-clustering_start_t)
