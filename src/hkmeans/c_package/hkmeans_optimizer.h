/*
 *
 * Copyright 2021- IBM Inc. All rights reserved
 * SPDX-License-Identifier: Apache2.0
 *
 */

#ifndef HKMEANS_OPTIMIZER_H
#define HKMEANS_OPTIMIZER_H

#include <stdint.h>

class HKMeansOptimizer {
    public:
        HKMeansOptimizer(int32_t n_clusters, int32_t n_features);
        virtual ~HKMeansOptimizer();

        void init_partition(
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const double *xy_data,
                const double *x_squared_norm, int32_t *labels,
                int32_t *t_size, double *t_centroid_sum, double *t_centroid_avg,
                double *t_squared_norm, double* inertia);

        void iterate(bool clustering_mode,                          // clustering / classification mode
                int32_t n_samples, const int32_t *xy_indices,       // data to cluster / classify
                const int32_t *xy_indptr, const double *xy_data,
                const double *x_squared_norm,
                int32_t* x_permutation,                             // order of iteration
                int32_t *t_size, double *t_centroid_sum,            // centroids
                double *t_centroid_avg, double *t_squared_norm,
                int32_t *labels, double* costs, double* total_cost, // assigned labels and costs
                double* inertia, double* change_rate);              // stats on updates

    private:
        int32_t n_clusters;
        int32_t n_features;
};

#endif // HKMEANS_OPTIMIZER_H
