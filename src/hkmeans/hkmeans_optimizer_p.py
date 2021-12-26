#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import row_norms


class PHKMeansOptimizer:

    def __init__(self, n_clusters, n_features, n_samples, x, x_squared_norm):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.x = x
        self.x_squared_norm = x_squared_norm
        self.sparse = issparse(x)

    def init_partition(self, labels, t_size, t_centroid_sum, t_centroid_avg, t_squared_norm):
        sparse = issparse(self.x)
        for i in range(self.n_samples):
            t = labels[i]
            t_size[t] += 1
            if sparse:
                i_start = self.x.indptr[i]
                i_end = self.x.indptr[i + 1]
                v_indices = self.x.indices[i_start:i_end]
                v_data = self.x.data[i_start:i_end]
                t_centroid_sum[t, v_indices] += v_data
            else:
                t_centroid_sum[t, :] += self.x[i, :]

        np.multiply(t_centroid_sum, (1 / t_size)[:, None], out=t_centroid_avg)
        if sparse:
            t_squared_norm[:] = row_norms(t_centroid_avg, squared=True)
        else:
            t_squared_norm[:] = 0

        # calculate inertia
        inertia = 0
        for i in range(self.n_samples):
            t = labels[i]
            if sparse:
                i_start = self.x.indptr[i]
                i_end = self.x.indptr[i + 1]
                v_indices = self.x.indices[i_start:i_end]
                v_data = self.x.data[i_start:i_end]
                inertia += (t_squared_norm[t] + self.x_squared_norm[i]
                            - 2 * np.dot(t_centroid_avg[t, v_indices], v_data))
            else:
                subtraction = t_centroid_avg[t, :] - self.x[i, :]
                inertia += np.dot(subtraction, subtraction)

        return inertia

    def optimize(self, x_permutation, t_size, t_centroid_sum, t_centroid_avg,
                 t_squared_norm, labels, inertia, ref_labels=None):
        return self.iterate(True, self.n_samples, self.x, self.x_squared_norm,
                            x_permutation, t_size, t_centroid_sum, t_centroid_avg,
                            t_squared_norm, labels, None, inertia, ref_labels)

    def infer(self, n_samples, x, x_squared_norm, t_size, t_centroid_sum,
              t_centroid_avg, t_squared_norm, labels, costs, ref_labels=None):
        return self.iterate(False, n_samples, x, x_squared_norm, None, t_size,
                            t_centroid_sum, t_centroid_avg, t_squared_norm,
                            labels, costs, None, ref_labels)

    def iterate(self, clustering_mode, n_samples, x, x_squared_norm, x_permutation,
                t_size, t_centroid_sum, t_centroid_avg, t_squared_norm,
                labels, costs, inertia, ref_labels=None):

        n_changes = 0

        total_cost = 0

        if not self.sparse:
            tmp_delta = np.empty_like(t_centroid_avg)
        else:
            tmp_delta = None

        for i in range(n_samples):
            x_id = x_permutation[i] if x_permutation is not None else i
            old_t = labels[x_id]

            if clustering_mode and t_size[old_t] == 1:
                continue  # if t is a singleton cluster we do not reduce it any further

            # obtain local references
            if self.sparse:
                x_start = x.indptr[x_id]
                x_end = x.indptr[x_id + 1]
                x_indices = x.indices[x_start:x_end]
                x_data = x.data[x_start:x_end]
                x_squared_norm_x = x_squared_norm[x_id]
            else:
                x_indices = None
                x_data = x[x_id, :]
                x_squared_norm_x = None

            if clustering_mode:
                # withdraw x from its current cluster
                if self.sparse:
                    t_centroid_sum[old_t, x_indices] -= x_data
                    dot_product = np.dot(x_data, t_centroid_sum[old_t, x_indices])
                    t_squared_norm[old_t] = (t_squared_norm[old_t] * (t_size[old_t] ** 2)
                                             - x_squared_norm_x - 2 * dot_product) / ((t_size[old_t] - 1) ** 2)
                else:
                    t_centroid_sum[old_t, :] -= x_data
                    np.multiply(t_centroid_sum[old_t, :], 1 / (t_size[old_t] - 1), out=t_centroid_avg[old_t, :])
                t_size[old_t] -= 1

            # select new_t
            if self.sparse:
                dot_product = (t_centroid_sum[:, x_indices] @ x_data) / t_size
                tmp_costs = t_squared_norm + x_squared_norm_x - 2 * dot_product
            else:
                np.subtract(t_centroid_avg, x_data[np.newaxis, :], out=tmp_delta)
                tmp_costs = (tmp_delta[:, None, :] @ tmp_delta[..., None]).ravel()
            tmp_costs *= t_size / (t_size + 1)
            new_t = np.argmin(tmp_costs).item()

            if ref_labels is not None:
                ref_t = ref_labels[x_id]
                if new_t != ref_t and not np.isclose(tmp_costs[new_t], tmp_costs[ref_t]):
                    print("override t of cost=%.8f, with cost=%.8f" % (tmp_costs[new_t], tmp_costs[ref_t]))
                    new_t = ref_t

            if clustering_mode:

                # update membership
                if self.sparse:
                    dot_product = np.dot(x_data, t_centroid_sum[new_t, x_indices])
                    t_squared_norm[new_t] = (t_squared_norm[new_t] * (t_size[new_t] ** 2)
                                             + x_squared_norm_x + 2 * dot_product) / ((t_size[new_t] + 1) ** 2)
                    t_centroid_sum[new_t, x_indices] += x_data
                else:
                    t_centroid_sum[new_t, :] += x_data
                    np.multiply(t_centroid_sum[new_t, :], 1 / (t_size[new_t] + 1), out=t_centroid_avg[new_t, :])
                t_size[new_t] += 1

                # update stats
                if new_t != old_t:
                    n_changes += 1
                    inertia -= tmp_costs[old_t] - tmp_costs[new_t]

            else:
                total_cost += tmp_costs[new_t]
                costs[x_id, :] = tmp_costs

            labels[x_id] = new_t

        if clustering_mode:
            return n_changes / self.n_samples if self.n_samples > 0 else 0, inertia
        else:
            return total_cost
