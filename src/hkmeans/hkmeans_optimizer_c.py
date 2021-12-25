#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

from scipy.sparse import issparse
from hkmeans.c_package.c_hkmeans_optimizer import CHKMeansOptimizer as CHKMeansOptimizerExtension


class CHKMeansOptimizer:

    def __init__(self, n_clusters, n_features, n_samples, x, x_squared_norm):
        self.c_hkmeans_optimizer = CHKMeansOptimizerExtension(n_clusters, n_features)
        self.n_samples = n_samples
        self.x = x
        self.x_squared_norm = x_squared_norm
        self.sparse = issparse(x)
        if self.sparse:
            self.x_indices = self.x.indices
            self.x_indptr = self.x.indptr
            self.x_data = self.x.data
        else:
            self.x_indices = None
            self.x_indptr = None
            self.x_data = self.x.ravel()

    def init_partition(self, labels, t_size, t_centroid_sum, t_centroid_avg, t_squared_norm):
        return self.c_hkmeans_optimizer.init_partition(self.n_samples, self.x_indices, self.x_indptr,
                                                       self.x_data,  self.x_squared_norm, labels, t_size,
                                                       t_centroid_sum, t_centroid_avg, t_squared_norm)

    def optimize(self, x_permutation, t_size, t_centroid_sum, t_centroid_avg, t_squared_norm, labels, inertia):
        return self.c_hkmeans_optimizer.optimize(self.n_samples, self.x_indices, self.x_indptr, self.x_data,
                                                 self.x_squared_norm, x_permutation, t_size, t_centroid_sum,
                                                 t_centroid_avg, t_squared_norm, labels, inertia)

    def infer(self, n_samples, x, x_squared_norm, t_size, t_centroid_sum,
              t_centroid_avg, t_squared_norm, labels, costs):
        if self.sparse:
            x_indices = x.indices
            x_indptr = x.indptr
            x_data = x.data
        else:
            x_indices = None
            x_indptr = None
            x_data = x.ravel()
        return self.c_hkmeans_optimizer.infer(n_samples, x_indices, x_indptr, x_data, x_squared_norm,
                                              t_size, t_centroid_sum, t_centroid_avg, t_squared_norm,
                                              labels, costs)
