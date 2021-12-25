#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import os
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from hkmeans.hkmeans_main import HKMeans
from examples.clustering_utils import fetch_20ng, create_heatmap


# by default, this test is meant for evaluating the clustering
# quality and it runs with 4 random initializations. however,
# if we are interested in evaluating speed, we will use a single
# initialization.
speed_test_mode = True

# step 0 - create an output directory if it does not exist
output_path = os.path.join("output", "ex1")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# step 1 - read the dataset
texts, gold_labels, n_clusters, topics, n_samples = fetch_20ng('all')
print("Clustering dataset contains %d texts from %d topics" % (n_samples, n_clusters))
print()

# step 2 - represent the clustering data using bow of the 10k most frequent
# unigrams in the dataset
vectorizer = TfidfVectorizer(max_features=10000)
vectors = vectorizer.fit_transform(texts)
vectors = vectors.toarray()

# step 3 - create an instance of Hartigan K-Means and run the actual clustering
# n_init = the number of random initializations to perform
# max_ter = the maximal number of iteration in each initialization
# n_jobs = the maximal number of initializations to run in parallel
clustering_start_t = time()
n_init = 1 if speed_test_mode else 4
hkmeans = HKMeans(n_clusters=n_clusters, random_state=128, n_init=n_init,
                  n_jobs=-1, max_iter=5, verbose=True, optimizer_type='B')
hkmeans.fit(vectors)
clustering_end_t = time()

print("Clustering time: %.3f secs." % (clustering_end_t - clustering_start_t))

# step 4 - some evaluation
homogeneity = metrics.homogeneity_score(gold_labels, hkmeans.labels_)
completeness = metrics.completeness_score(gold_labels, hkmeans.labels_)
v_measure = metrics.v_measure_score(gold_labels, hkmeans.labels_)
ami = metrics.adjusted_mutual_info_score(gold_labels, hkmeans.labels_)
ari = metrics.adjusted_rand_score(gold_labels, hkmeans.labels_)
print("Homogeneity: %0.3f" % homogeneity)
print("Completeness: %0.3f" % completeness)
print("V-measure: %0.3f" % v_measure)
print("Adjusted Mutual-Information: %.3f" % ami)
print("Adjusted Rand-Index: %.3f" % ari)

# save a heatmap
create_heatmap(gold_labels, hkmeans.labels_,
               topics, 'Hartigan K-Means clustering heatmap',
               os.path.join(output_path, 'hkmeans_heatmap'))

# save a report
with open(os.path.join(output_path, "hkmeans_report.txt"), "wt") as f:
    f.write(str(hkmeans) + "\n")
    f.write("Size: %d vectors\n" % vectors.shape[0])
    f.write("Time: %.3f seconds\n" % (clustering_end_t - clustering_start_t))
    f.write("Measures:\n")
    f.write("\tHomogeneity: %0.3f\n" % homogeneity)
    f.write("\tCompleteness: %0.3f\n" % completeness)
    f.write("\tV-measure: %0.3f\n" % v_measure)
    f.write("\tAdjusted Mutual Information: %.3f\n" % ami)
    f.write("\tAdjusted Rand Index: %.3f\n" % ari)
    f.write("\n\n")
