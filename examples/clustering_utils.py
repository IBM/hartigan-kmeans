#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import os
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot as plt
import seaborn as sns


def create_heatmap(gold_array, predicted_array, names, title, file_name,
                   threshold=0.05, use_png=False, use_svg=True):
    gold_labels = np.unique(gold_array)
    gold_labels_len = len(gold_labels)
    heatmap_matrix = contingency_matrix(gold_array, predicted_array)
    reordered_array = np.zeros_like(predicted_array)
    for i in range(gold_labels_len):
        gold_label, predicted_label = np.unravel_index(np.argmax(heatmap_matrix, axis=None), heatmap_matrix.shape)
        heatmap_matrix[gold_label, :] = -1
        heatmap_matrix[:, predicted_label] = -1
        predicted_indices = np.where(predicted_array == predicted_label)[0]
        np.put(reordered_array, predicted_indices, [gold_label], mode='wrap')
    heatmap_matrix = contingency_matrix(gold_array, reordered_array)
    sums_vector = heatmap_matrix.sum(axis=1)
    heatmap_matrix = np.divide(heatmap_matrix, sums_vector[:, np.newaxis])
    heatmap_matrix = np.around(heatmap_matrix, decimals=2)
    mask = np.isclose(heatmap_matrix, np.zeros_like(heatmap_matrix), atol=threshold)
    plt.ioff()
    plt.figure(figsize=(gold_labels_len * 0.667, gold_labels_len * 0.667))
    ax = sns.heatmap(heatmap_matrix,
                     cmap="BuGn",
                     yticklabels=names,
                     fmt=".2f",
                     mask=mask,
                     annot=True)
    ax.set_title(title)
    ax.figure.tight_layout()
    if use_png:
        plt.savefig(file_name + ".png", format='png', dpi=300)
    if use_svg:
        plt.savefig(file_name + ".svg", format='svg')
    plt.close()


def get_alignment(gold_array, predicted_array):
    alignment = {}
    for predicted_label in np.unique(predicted_array):
        predicted_label_indices = np.where(predicted_array == predicted_label)
        gold_array_labels = gold_array[predicted_label_indices]
        most_comon_gold_label = np.argmax(np.bincount(gold_array_labels)).item()
        alignment[predicted_label] = most_comon_gold_label
    return alignment


def fetch_20ng(subset):
    dataset = fetch_20newsgroups(subset=subset, categories=None,
                                 shuffle=True, random_state=256)
    texts = dataset.data
    gold_labels_array = dataset.target
    unique_labels = np.unique(gold_labels_array)
    n_clusters = unique_labels.shape[0]
    topics = dataset.target_names
    n_samples = len(texts)
    return texts, gold_labels_array, n_clusters, topics, n_samples


def save_report_and_heatmap(gold_labels_array, predictions_array, topics,
                            algorithm, algorithm_name, output_path,
                            ami, ari, homogeneity, completeness, v_measure,
                            n_samples, vectorization_time, clustering_time):
    # save a heatmap
    create_heatmap(gold_labels_array, predictions_array,
                   topics, algorithm_name + ' heatmap',
                   os.path.join(output_path, algorithm_name + '_heatmap'))
    with open(os.path.join(output_path, algorithm_name + "_report.txt"), "wt") as f:
        f.write("Clustering:\n")
        f.write(str(algorithm) + "\n")
        f.write("Dataset size: %d texts\n" % n_samples)
        f.write("Vectorization time: %.3f seconds\n" % vectorization_time)
        f.write("Clustering time: %.3f seconds\n" % clustering_time)
        f.write("Measures:\n")
        f.write("\tHomogeneity: %0.3f\n" % homogeneity)
        f.write("\tCompleteness: %0.3f\n" % completeness)
        f.write("\tV-measure: %0.3f\n" % v_measure)
        f.write("\tAdjusted Mutual Information: %.3f\n" % ami)
        f.write("\tAdjusted Rand Index: %.3f\n" % ari)
        f.write("\n\n")


