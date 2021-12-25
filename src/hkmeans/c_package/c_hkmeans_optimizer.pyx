#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# distutils: language = c++
# cython: language_level=3

from .c_hkmeans_optimizer cimport HKMeansOptimizer

from libc.stdint cimport int32_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)


# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods

# Python extension type.
cdef class CHKMeansOptimizer:
    cdef HKMeansOptimizer* c_hkmeans_optimizer  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int n_clusters, int n_features):
        self.c_hkmeans_optimizer = new HKMeansOptimizer(n_clusters, n_features)

    def __dealloc__(self):
        del self.c_hkmeans_optimizer

    def optimize(self, int32_t n_samples, const int32_t[::1] x_indices,
                 const int32_t[::1] x_indptr, const double[::1] x_data,
                 const double[::1] x_squared_norm,
                 int32_t[::1] x_permutation,
                 int32_t[::1] t_size, double[:,::1] t_centroid_sum,
                 double[:,::1] t_centroid_avg, double[::1] t_squared_norm,
                 int32_t[::1] labels, double inertia):
        cdef double change_rate = 0
        self.c_hkmeans_optimizer.iterate(True, n_samples,
                                         &x_indices[0] if x_indices is not None else NULL,
                                         &x_indptr[0] if x_indptr is not None else NULL,
                                         &x_data[0], &x_squared_norm[0],
                                         &x_permutation[0],
                                         &t_size[0], &t_centroid_sum[0, 0], &t_centroid_avg[0, 0],
                                         &t_squared_norm[0], &labels[0], NULL, NULL,  # costs and total cost
                                         &inertia, &change_rate)
        return change_rate, inertia

    def infer(self, int32_t n_samples, const int32_t[::1] x_indices,
              const int32_t[::1] x_indptr, const double[::1] x_data,
              const double[::1] x_squared_norm,
              int32_t[::1] t_size, double[:,::1] t_centroid_sum,
              double[:,::1] t_centroid_avg, double[::1] t_squared_norm,
              int32_t[::1] labels, double[:,::1] costs):
        cdef double inertia
        self.c_hkmeans_optimizer.iterate(False, n_samples,
                                         &x_indices[0] if x_indices is not None else NULL,
                                         &x_indptr[0] if x_indptr is not None else NULL,
                                         &x_data[0], &x_squared_norm[0],
                                         NULL,  # permutation
                                         &t_size[0], &t_centroid_sum[0, 0], &t_centroid_avg[0, 0],
                                         &t_squared_norm[0],  &labels[0], &costs[0, 0], &inertia,
                                         NULL, NULL) # inertia and change_rate
        return inertia

    def init_partition(self, int32_t n_samples, const int32_t[::1] x_indices,
                       const int32_t[::1] x_indptr, const double[::1] x_data,
                       const double[::1] x_squared_norm, int32_t[::1] labels,
                       int32_t[::1] t_size, double[:,::1] t_centroid_sum,
                       double[:,::1] t_centroid_avg, double[::1] t_squared_norm):
        cdef double inertia
        self.c_hkmeans_optimizer.init_partition(n_samples,
                                                &x_indices[0] if x_indices is not None else NULL,
                                                &x_indptr[0] if x_indptr is not None else NULL,
                                                &x_data[0], &x_squared_norm[0], &labels[0],
                                                &t_size[0], &t_centroid_sum[0, 0],
                                                &t_centroid_avg[0, 0], &t_squared_norm[0], &inertia)
        return inertia
