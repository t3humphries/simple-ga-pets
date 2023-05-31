"""K-medoids clustering"""
# Modified by [Redacted for blind review]
# Authors: Timo Erkkilä <timo.erkkila@gmail.com>
#          Antti Lehmussola <antti.lehmussola@gmail.com>
#          Kornel Kiełczewski <kornel.mail@gmail.com>
#          Zane Dufour <zane.dufour@gmail.com>
# License: BSD 3 clause

import warnings
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

# cython implementation of steps in PAM algorithm.
from _k_medoids_helper import _compute_optimal_swap, _build
#from GA_Operations import K_Median


def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
    sample distances to closest cluster centers.
    Parameters
    ----------
    distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
        Distances to cluster centers.
    Returns
    -------
    Sum of sample distances to closest cluster centers.
    """

    # Define inertia as the sum of the sample-distances
    # to closest cluster centers
    inertia = np.sum(np.min(distances, axis=1))

    return inertia


class K_Medoids_Custom(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.
    Read more in the :ref:`User Guide <k_medoids>`.
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.
    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.
    init : {'random', 'heuristic', 'build}, optional, default: 'heuristic'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).
    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.
    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)
    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X
    labels_ : array, shape = (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np
    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1., 2.],
           [4., 2.]])
    >>> kmedoids.inertia_
    8.0
    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.
    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.
    See also
    --------
    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.
    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).
    """

    def __init__(
        self,
        n_clusters=8,
        method="pam",
        init="heuristic",
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def _check_nonnegative_int(self, value, desc, strict=True):
        """Validates if value is a valid integer > 0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments."""

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter", False)

        # Check init
        init_methods = ["random", "heuristic", "build"]
        if self.init not in init_methods:
            raise ValueError(
                "init needs to be one of "
                + "the following: "
                + "%s" % init_methods
            )
        if self.method != "pam":
            raise ValueError(
                "this method is currently not supported"
            )

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.
        Parameters
        ----------
        X : (n_public, private) if metric == 'precomputed'
            Dataset to cluster.
        y : Ignored
        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)

        self._check_init_args()
        D = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )
        if self.n_clusters > X.shape[1]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of public samples %d."
                % (self.n_clusters, X.shape[1])
            )

        medoid_idxs = self._initialize_medoids(
            D, self.n_clusters, random_state_
        )
        labels = None

        if self.method == "pam":
            # Compute the distance to the first and second closest points
            # among medoids.

            if self.n_clusters == 1 and self.max_iter > 0:
                # PAM SWAP step can only be used for n_clusters > 1
                warnings.warn(
                    "n_clusters should be larger than 2 if max_iter != 0 "
                    "setting max_iter to 0."
                )
                self.max_iter = 0
            elif self.max_iter > 0:
                sorted_table = np.sort(D[:,medoid_idxs], axis=1)
                Djs = sorted_table[:,0]
                Ejs = sorted_table[:,1]

        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded

        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[:,medoid_idxs], axis=1)

            if self.method == "alternate":
                # Update medoids with the new cluster indices
                self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            elif self.method == "pam":
                not_medoid_idxs = np.delete(np.arange(len(D[0])), medoid_idxs)
                optimal_swap = _compute_optimal_swap( #TODO fix this
                    D,
                    medoid_idxs.astype(np.intc),
                    not_medoid_idxs.astype(np.intc),
                    Djs,
                    Ejs,
                    self.n_clusters,
                )
                if optimal_swap is not None:
                    i, j, _ = optimal_swap
                    medoid_idxs[medoid_idxs == i] = j

                    # update Djs and Ejs with new medoids
                    sorted_table = np.sort(D[:,medoid_idxs], axis=1)
                    Djs = sorted_table[:,0]
                    Ejs = sorted_table[:,1]

            else:
                raise ValueError(
                    f"method={self.method} is not supported. Supported methods "
                    f"are 'pam' and 'alternate'."
                )

            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

        # Set the resulting instance variables.
        self.cluster_centers_ = None

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = np.argmin(D[:, medoid_idxs], axis=1)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X))

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs): #Currently not implemented 
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        """Compute the cose for a given configuration of the medoids"""
        return _compute_inertia(D[:, medoid_idxs])

    def transform(self, X):
        """Transforms X to cluster-distance space.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.
        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )


        check_is_fitted(self, "medoid_indices_")
        return X[:, self.medoid_indices_]

       

    def predict(self, X):
        """Predict the closest cluster for each sample in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.
        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        check_is_fitted(self, "medoid_indices_")
        return np.argmin(X[:, self.medoid_indices_], axis=1)

    def _initialize_medoids(self, D, n_clusters, random_state_):
        """Select initial mediods when beginning clustering."""
        if self.init == "random":  # Random initialization
            # Pick random k medoids as the initial ones.
            medoids = random_state_.choice(len(D[0]), n_clusters)
        elif self.init == "heuristic":  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            medoids = np.argpartition(np.sum(D, axis=0), n_clusters - 1)[:n_clusters]
        elif self.init == "build":  # Build initialization
            medoids = _build(D, n_clusters).astype(np.int64)
        else:
            raise ValueError(f"init value '{self.init}' not recognized")

        return medoids

#if __name__ == "__main__":
    # utils = {}
    # for dataset in ['Spam', 'Mushrooms', 'Adult', 'Credit']:
    #     data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
    #     public_data = data_dict['clustering_sets']['public']
    #     private_data = data_dict['clustering_sets']['private']
    #     ga_object = K_Median(public_data, private_data, 5, None, None)

    #     dist_mat = ga_object.distance_matrix
    #     print(len(dist_mat), len(dist_mat[0]))
    #     #----------------SK_Learn Comparison----------------------------------------
    #     kmediods = K_Medoids_Custom(init='build', n_clusters=5).fit(dist_mat)
    #     result = kmediods.medoid_indices_
    #     utils[dataset] = ga_object.get_utility(result)

    #     #---------------------------------------------------------------------------
    # print(utils)