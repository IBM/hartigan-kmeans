/*
 *
 * Copyright 2021- IBM Inc. All rights reserved
 * SPDX-License-Identifier: Apache2.0
 *
 */

#include "hkmeans_optimizer.h"


// Constructor
HKMeansOptimizer::HKMeansOptimizer(int32_t n_clusters, int32_t n_features)
    : n_clusters(n_clusters), n_features(n_features) {}

// Destructor
HKMeansOptimizer::~HKMeansOptimizer() {}

// Partition initialization
void HKMeansOptimizer::init_partition(
        int32_t n_samples, const int32_t *xy_indices,
        const int32_t *xy_indptr, const double *xy_data,
        const double *x_squared_norm, int32_t *labels,
        int32_t *t_size, double *t_centroid_sum, double *t_centroid_avg,
        double *t_squared_norm, double* inertia) {

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const double* x_data;
    double x_squared_norm_x;

    bool sparse = xy_indices != NULL;

    // sum the vectors of each cluster
    for (int x=0; x<n_samples ; x++) {
        int t = labels[x];
        t_size[t] += 1;

        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &xy_indices[x_start];
            x_data = &xy_data[x_start];
            x_squared_norm_x = x_squared_norm[x];
        } else {
            x_data = &xy_data[x * n_features];
        }

        double *t_centroid_sum_t = &t_centroid_sum[t * n_features];
        if (sparse) {
            for (int j=0 ; j<x_size ; j++) {
                t_centroid_sum_t[x_indices[j]] += x_data[j];
            }
        } else {
            for (int j=0 ; j<x_size ; j++) {
                t_centroid_sum_t[j] += x_data[j];
            }
        }
    }

    // calculate the centroids and their squared norms
    for (int t=0; t<this->n_clusters ; t++) {
        double *t_centroid_avg_t = &t_centroid_avg[t * this->n_features];
        double *t_centroid_sum_t = &t_centroid_sum[t * this->n_features];
        double inv_t_size = 1.0 / t_size[t];
        double t_squared_norm_t = 0.0;
        for (int j=0 ; j<this->n_features ; j++) {
            double value = t_centroid_sum_t[j] * inv_t_size;
            t_centroid_avg_t[j] = value;
            if (sparse) {
                t_squared_norm_t += value * value;
            }
        }
        t_squared_norm[t] = t_squared_norm_t;
    }

    // calculate the inertia (sum of distances from centroids)
    *inertia = 0.0;
    for (int x=0; x<n_samples ; x++) {
        int t = labels[x];
        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &xy_indices[x_start];
            x_data = &xy_data[x_start];
            x_squared_norm_x = x_squared_norm[x];
        } else {
            x_data = &xy_data[x * n_features];
        }
        double *t_centroid_avg_t = &t_centroid_avg[t * this->n_features];
        if (sparse) {
            double dot_product = 0.0;
            for (int j=0 ; j<x_size ; j++) {
                dot_product += x_data[j] * t_centroid_avg_t[x_indices[j]];
            }
            *inertia += x_squared_norm[x] + t_squared_norm[t] - 2 * dot_product;
        } else {
            for (int j=0 ; j<x_size ; j++) {
                double delta = x_data[j] - t_centroid_avg_t[j];
                *inertia += delta * delta;
            }
        }
    }
}

// Iteration over n samples for clustering / classification.
void HKMeansOptimizer::iterate(bool clustering_mode,        // clustering / classification mode
        int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
        const int32_t *xy_indptr, const double *xy_data,
        const double *x_squared_norm,
        int32_t* x_permutation,                             // order of iteration
        int32_t *t_size, double *t_centroid_sum,            // centroids
        double *t_centroid_avg, double *t_squared_norm,
        int32_t *labels, double* costs, double* total_cost, // assigned labels and costs
        double* inertia, double* change_rate) {             // stats on updates

    int n_changes = 0;

    if (!clustering_mode) {
        *total_cost = 0;
    }

    int32_t x_start = 0;
    int32_t x_end = n_features;
    int32_t x_size = x_end - x_start;
    const int32_t* x_indices;
    const double* x_data;
    double x_squared_norm_x;

    bool sparse = xy_indices != NULL;

    for (int32_t i=0; i<n_samples ; i++) {
        int32_t x = clustering_mode ? x_permutation[i] : i;
        int32_t old_t = labels[x]; // when clustering mode = false, this value is garbage

        if (clustering_mode && t_size[old_t] == 1) {
            // skip elements from singleton clusters
            continue;
        }

        // obtain local pointers
        if (sparse) {
            x_start = xy_indptr[x];
            x_end = xy_indptr[x + 1];
            x_size = x_end - x_start;
            x_indices = &xy_indices[x_start];
            x_data = &xy_data[x_start];
            x_squared_norm_x = x_squared_norm[x];
        } else {
            x_data = &xy_data[x * n_features];
        }

        if (clustering_mode) {
            // withdraw x from its current cluster
            int32_t t_size_old_t = t_size[old_t];
            double *t_centroid_sum_old_t = &t_centroid_sum[old_t * n_features];
            if (sparse) {
                double t_squared_norm_old_t = t_squared_norm[old_t];
                double dot_product = 0.0;
                for (int j=0 ; j<x_size ; j++) {
                    t_centroid_sum_old_t[x_indices[j]] -= x_data[j];
                    dot_product += x_data[j] * t_centroid_sum_old_t[x_indices[j]];
                }
                t_squared_norm[old_t] = (t_squared_norm_old_t * (t_size_old_t * t_size_old_t)
                                         - x_squared_norm_x - 2 * dot_product)
                                        / ((t_size_old_t - 1) * (t_size_old_t - 1));
            } else {
                double* t_centroid_avg_old_t = &t_centroid_avg[old_t * n_features];
                double inv_t_size_old_t = 1.0 / (double)(t_size_old_t - 1);
                for (int j=0 ; j<x_size ; j++) {
                    t_centroid_sum_old_t[j] -= x_data[j];
                    t_centroid_avg_old_t[j] = t_centroid_sum_old_t[j] * inv_t_size_old_t;
                }
            }
            t_size[old_t] -= 1;
        }

        // pointer to the costs array (used only for classification)
        double* x_costs = clustering_mode ? NULL : &costs[n_clusters * x];

        double min_cost = 0;
        int32_t min_cost_t = -1;
        double cost_old_t = 0;

        for (int32_t t=0 ; t<n_clusters ; t++) {
            double t_size_t = t_size[t];
            double cost = 0;
            if (sparse) {
                double inv_t_size_t = 1.0 / t_size_t;
                double* t_centroid_sum_t = &t_centroid_sum[t * n_features];
                double t_squared_norm_t = t_squared_norm[t];
                for (int j=0 ; j<x_size ; j++) {
                    cost += x_data[j] * t_centroid_sum_t[x_indices[j]] * inv_t_size_t;
                }
                cost *= -2;
                cost += t_squared_norm_t + x_squared_norm_x;
            } else {
                double* t_centroid_avg_t = &t_centroid_avg[t * n_features];
                for (int j=0 ; j<x_size ; j++) {
                    double delta = x_data[j] - t_centroid_avg_t[j];
                    cost += delta * delta;
                }
            }
            cost *= t_size_t / (t_size_t + 1);

            if (min_cost_t == -1 || cost < min_cost) {
                min_cost_t = t;
                min_cost = cost;
            }

            if (clustering_mode) {
                if (t == old_t) {
                    cost_old_t = cost;
                }
            } else {
                x_costs[t] = cost;
            }
        }

        int32_t new_t = min_cost_t;

        if (clustering_mode) {

            // add x to its new cluster
            double *t_centroid_sum_new_t = &t_centroid_sum[new_t * n_features];
            int t_size_new_t = t_size[new_t];
            if (sparse) {
                double dot_product = 0.0;
                for (int32_t j=0 ; j<x_size ; j++) {
                    dot_product += x_data[j] * t_centroid_sum_new_t[x_indices[j]];
                    t_centroid_sum_new_t[x_indices[j]] += x_data[j];
                }
                t_squared_norm[new_t] = ((t_size_new_t * t_size_new_t) * t_squared_norm[new_t]
                                         + x_squared_norm_x + 2 * dot_product)
                                        / ((t_size_new_t + 1) * (t_size_new_t + 1));
            } else {
                double *t_centroid_avg_new_t = &t_centroid_avg[new_t * n_features];
                double inv_t_size_new_t = 1.0 / (double)(t_size_new_t + 1);
                for (int32_t j=0 ; j<x_size ; j++) {
                    t_centroid_sum_new_t[j] += x_data[j];
                    t_centroid_avg_new_t[j] = t_centroid_sum_new_t[j] * inv_t_size_new_t;
                }
            }
            t_size[new_t]++;

            if (new_t != old_t) {
                // update the changes counter
                n_changes++;

                // count the decrease in cost (distance)
                *inertia -= cost_old_t - min_cost;
            }

        } else {
            *total_cost += min_cost;
        }

        labels[x] = new_t;
    }

    if (clustering_mode) {
        // calculate the change rate
        *change_rate = n_samples > 0 ? n_changes / (double)n_samples : 0;
    }
}
