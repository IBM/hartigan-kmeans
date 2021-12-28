#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import copy
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from joblib import Parallel, delayed, effective_n_jobs

from hkmeans.hkmeans_optimizer_p import PHKMeansOptimizer
from hkmeans.hkmeans_optimizer_c import CHKMeansOptimizer


class HKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Hartigan K-Means clustering.

    Parameters
    ----------

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    n_init : int, default=10
        Number of times the algorithm will be run with different
        centroid seeds. The final result will be the initialization
        with highest mutual information between the clustering
        analysis and the vocabulary.

    max_iter : int, default=15
        Maximum number of iterations of the algorithm for a single run.

    tol : float, default=0.02
        Relative tolerance with regards to number of centroid updates
        to declare convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``-1`` means using all processors.


    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples)
        Labels of each point

    score_ : float
        Mutual information between the cluster analysis and the vocabulary.

    inertia_ : float
        The score value negated

    n_iter_ : int
        Number of iterations ran

    costs_ :  ndarray of shape (n_samples, n_clusters)
        The input samples transformed to o cluster-distance space

    """

    def __init__(self, n_clusters, random_state=None, n_jobs=1,
                 n_init=10, max_iter=15, tol=0.02, verbose=False,
                 optimizer_type='C'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.optimizer_type = optimizer_type

        self.n_samples = -1
        self.n_features = -1

        self.x = None
        self.x_squared_norm = None

        self.partition_ = None
        self.score_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.costs_ = None

    def __str__(self):
        param_values = [("n_clusters", self.n_clusters), ("n_jobs", self.n_jobs),
                        ("n_init", self.n_init), ("max_iter", self.max_iter),
                        ("tol", self.tol), ("random_state", self.random_state),
                        ("optimizer_type", self.optimizer_type),
                        ("verbose", self.verbose)]

        return "HKmeans(" + ", ".join(name + "=" + str(value) for name, value in param_values) + ")"

    def fit(self, x):
        """Compute Hartigan K-Means clustering.

        Parameters
        ----------
        x : sparse matrix, shape=(n_samples, n_features)
            It is recommended to provide count vectors (un-normalized)

        Returns
        -------
        self
            Fitted estimator.
        """
        self.n_samples, self.n_features = x.shape

        if not self.n_samples > 1:
            raise ValueError("n_samples=%d should be > 1" % self.n_samples)

        if self.n_samples < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d"
                             % (self.n_samples, self.n_clusters))

        random_state = check_random_state(self.random_state)

        if x.dtype == np.float32:
            x = x.astype(np.float64)

        self.x = x
        self.x_squared_norm = row_norms(x, squared=True)

        if self.verbose:
            print("Initialization complete")

        # Main (restarts) loop
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        if effective_n_jobs(self.n_jobs) == 1 or self.n_init == 1:
            # For a single thread, less memory is needed if we just store one set
            # of the best results (as opposed to one set per run per thread).
            best_partition = None
            for i, seed in enumerate(seeds):
                # run Hartigan K-Means once
                tmp_partition = self.hkmeans_single(seed, run_id=(i if self.n_init > 1 else None))
                if best_partition is None or tmp_partition.score > best_partition.score:
                    best_partition = tmp_partition
        else:
            # parallelization of runs
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self.hkmeans_single)(random_state=seed, job_id=job_id)
                for job_id, seed in enumerate(seeds))
            scores = np.fromiter((T.score for T in results), float, self.n_init)
            best_partition = results[np.argmax(scores)]

        if self.verbose:
            print("H-KMeans inertia in best partition: %.2f" % -best_partition.score)

        # Last updates
        self.partition_ = best_partition
        self.score_ = best_partition.score
        self.inertia_ = -self.score_
        self.n_iter_ = best_partition.n_iter
        self.cluster_centers_ = best_partition.t_centroid_sum / best_partition.t_size[:, None]
        self.labels_, self.costs_, _ = self.infer_labels_costs_score(self.n_samples, self.x,
                                                                     self.x_squared_norm)
        return self

    def hkmeans_single(self, random_state, job_id=None, run_id=None):
        # initialization: random generator, partition and optimizers
        random_state = check_random_state(random_state)
        optimizer, v_optimizer = self.create_optimizers()
        partition = Partition(self.n_samples, self.n_features, self.n_clusters,
                              self.x, random_state, optimizer, v_optimizer)

        # main loop of optimizing the partition
        self.report_status(partition, job_id, run_id)
        while not self.converged(partition):
            self.optimize(partition, optimizer, v_optimizer)
            self.report_status(partition, job_id, run_id)
            # partition.dump()

        self.report_convergence(partition, job_id, run_id)

        # final calculations
        partition.score = -partition.inertia

        # return the partition
        return partition

    def create_c_optimizer(self):
        return CHKMeansOptimizer(self.n_clusters, self.n_features, self.n_samples, self.x, self.x_squared_norm)

    def create_p_optimizer(self):
        return PHKMeansOptimizer(self.n_clusters, self.n_features, self.n_samples, self.x, self.x_squared_norm)

    def create_optimizers(self):
        if self.optimizer_type == 'C':
            optimizer = self.create_c_optimizer()
            v_optimizer = None
        elif self.optimizer_type == 'P':
            optimizer = self.create_p_optimizer()
            v_optimizer = None
        else:
            optimizer = self.create_c_optimizer()
            v_optimizer = self.create_p_optimizer()
        return optimizer, v_optimizer

    def report_status(self, partition, job_id, run_id):
        if self.verbose:
            print((("Job %2d, " % job_id) if job_id is not None else "") +
                  (("Run %2d, " % run_id) if run_id is not None else "") +
                  ("Iteration %2d, inertia=%.2f" % (partition.n_iter, partition.inertia)) +
                  ((", Updates=%.2f%%" % (partition.change_ratio * 100))
                   if partition.n_iter > 0 else ""))

    def report_convergence(self, partition, job_id, run_id):
        if self.verbose:
            print((("Job %2d, " % job_id) if job_id is not None else "") +
                  (("Run %2d, " % run_id) if run_id is not None else "") +
                  partition.convergence_str)

    def optimize(self, partition, optimizer, v_optimizer):
        x_permutation = partition.random_state.permutation(self.n_samples).astype(np.int32)

        v_partition = None
        if v_optimizer:
            v_partition = copy.deepcopy(partition)

        partition.change_ratio, partition.inertia = optimizer.optimize(
            x_permutation, partition.t_size, partition.t_centroid_sum, partition.t_centroid_avg,
            partition.t_squared_norm, partition.labels, partition.inertia)

        if v_optimizer:
            v_partition.change_ratio, v_partition.inertia = v_optimizer.optimize(
                x_permutation, v_partition.t_size, v_partition.t_centroid_sum, v_partition.t_centroid_avg,
                v_partition.t_squared_norm, v_partition.labels, v_partition.inertia)
            assert np.allclose(partition.change_ratio, v_partition.change_ratio)
            assert np.allclose(partition.inertia, v_partition.inertia)
            assert np.allclose(partition.t_size, v_partition.t_size)
            assert np.allclose(partition.t_centroid_sum, v_partition.t_centroid_sum)
            assert np.allclose(partition.t_centroid_avg, v_partition.t_centroid_avg)
            assert np.allclose(partition.t_squared_norm, v_partition.t_squared_norm)
            assert np.allclose(partition.labels, v_partition.labels)

        partition.n_iter += 1
        if v_optimizer:
            v_partition.n_iter += 1

    def converged(self, partition):
        if partition.n_iter > 0 and partition.change_ratio <= self.tol:
            partition.convergence_str = "Hartigan K-Means converged in iteration %d with change=%.2f%%" \
                                        % (partition.n_iter, 100 * partition.change_ratio)
            return True
        elif partition.n_iter >= self.max_iter:
            partition.convergence_str = "Hartigan K-Means did NOT converge (change=%.2f%%), " \
                                        "stopped due to max_iter=%d" \
                                        % (100 * partition.change_ratio, self.max_iter)
            return True
        else:
            return False

    def infer_labels_costs_score(self, n_samples, x, x_squared_norm):
        optimizer, v_optimizer = self.create_optimizers()
        labels = np.empty(n_samples, dtype=np.int32)
        costs = np.empty((n_samples, self.n_clusters))
        score = optimizer.infer(n_samples, x, x_squared_norm,
                                self.partition_.t_size,
                                self.partition_.t_centroid_sum,
                                self.partition_.t_centroid_avg,
                                self.partition_.t_squared_norm,
                                labels, costs)
        if v_optimizer:
            v_labels = np.empty(n_samples, dtype=np.int32)
            v_costs = np.empty((n_samples, self.n_clusters))
            v_score = v_optimizer.infer(n_samples, x, x_squared_norm,
                                        self.partition_.t_size,
                                        self.partition_.t_centroid_sum,
                                        self.partition_.t_centroid_avg,
                                        self.partition_.t_squared_norm,
                                        v_labels, v_costs)
            assert np.isclose(score, v_score)
            assert np.allclose(costs, v_costs)
            assert np.allclose(labels, v_labels)
        return labels, costs, score

    def fit_new_data(self, x):
        n_samples, _ = x.shape

        if not self.partition_:
            raise ValueError("Estimator HKmeans must be fitted before being used")

        if not self.n_samples > 1:
            raise ValueError("n_samples=%d should be > 1" % self.n_samples)

        x_squared_norm = np.dot(x, x)

        return self.infer_labels_costs_score(n_samples, x, x_squared_norm)

    def fit_transform(self, x, y=None, sample_weight=None):
        """Compute clustering and transform x to cluster-distance space.

        Equivalent to fit(x).transform(x) but more efficient.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array, shape [n_samples, n_clusters]
            X transformed in the new space.
        """
        self.fit(x)
        return self.costs_

    def fit_predict(self, x, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Equivalent to fit(x).predict(x) but more efficient.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        self.fit(x)
        return self.labels_

    def transform(self, x):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  The array returned is always dense.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        labels, costs, score = self.fit_new_data(x)
        return costs

    def predict(self, x):
        """Predict the closest cluster each sample in x belongs to.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        labels, costs, score = self.fit_new_data(x)
        return labels

    def score(self, x):
        """The value of x on the algorithm objective. This is the sum
        of distances between each sample in x and the centroid of the
        cluster predicted for it.

        Parameters
        ----------
        x : sparse matrix of shape (n_samples, n_features)
            New data.

        Returns
        -------
        score : float
            The value of x on the algorithm objective.
        """

        labels, costs, score = self.fit_new_data(x)
        return score


class Partition:
    def __init__(self, n_samples, n_features, n_clusters, x, random_state, optimizer, v_optimizer):

        # Produce a random partition as an initialization point
        self.labels = random_state.permutation(np.linspace(0, n_clusters, n_samples,
                                                           endpoint=False).astype(np.int32))

        # initialize the data structures based on the labels and the vector values
        self.t_size = np.zeros(n_clusters, dtype=np.int32)
        self.t_centroid_sum = np.zeros((n_clusters, n_features), dtype=x.dtype)
        self.t_centroid_avg = np.empty((n_clusters, n_features), dtype=np.float64)
        self.t_squared_norm = np.empty(n_clusters, dtype=np.float64)
        self.inertia = optimizer.init_partition(self.labels, self.t_size, self.t_centroid_sum,
                                                self.t_centroid_avg, self.t_squared_norm)
        if v_optimizer is not None:
            v_t_size = np.zeros(n_clusters, dtype=np.int32)
            v_t_centroid_sum = np.zeros((n_clusters, n_features), dtype=x.dtype)
            v_t_centroid_avg = np.empty((n_clusters, n_features), dtype=np.float64)
            v_t_squared_norm = np.empty(n_clusters, dtype=np.float64)
            v_inertia = v_optimizer.init_partition(self.labels, v_t_size, v_t_centroid_sum,
                                                   v_t_centroid_avg, v_t_squared_norm)
            assert np.allclose(self.t_size, v_t_size)
            assert np.allclose(self.t_centroid_sum, v_t_centroid_sum)
            assert np.allclose(self.t_centroid_avg, v_t_centroid_avg)
            assert np.allclose(self.t_squared_norm, v_t_squared_norm)
            assert np.allclose(self.inertia, v_inertia)

        # more initializations
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_features = n_features
        self.n_iter = 0
        self.change_ratio = 0
        self.score = None
        self.convergence_str = None

    def __str__(self):
        return " size: %d\n counter: %d\n convergence_str: %s" % (
            self.n_clusters, self.n_iter, self.convergence_str)
