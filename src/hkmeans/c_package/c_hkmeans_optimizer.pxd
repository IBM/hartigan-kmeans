#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# cython: language_level=3, boundscheck=False

cdef extern from "hkmeans_optimizer.cpp":
    pass

from libcpp cimport bool
from libc.stdint cimport int32_t


# Declare the class with cdef
cdef extern from "hkmeans_optimizer.h":
    cdef cppclass HKMeansOptimizer:
        HKMeansOptimizer(int32_t n_clusters, int32_t n_features);

        void init_partition(
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const double *xy_data,
                const double *x_squared_norm, int32_t *labels,
                int32_t *t_size, double *t_centroid_sum, double *t_centroid_avg,
                double *t_squared_norm, double* inertia);

        void iterate(
                bool clustering_mode,
                int32_t n_samples, const int32_t *xy_indices,
                const int32_t *xy_indptr, const double *xy_data,
                const double *x_squared_norm,
                int32_t* x_permutation,
                int32_t *t_size, double *t_centroid_sum, double *t_centroid_avg,
                double *t_squared_norm, int32_t *labels, double* costs, double* total_cost,
                double* inertia, double* change_rate);
