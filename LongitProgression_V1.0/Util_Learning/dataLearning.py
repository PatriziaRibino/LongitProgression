from tslearn.clustering import TimeSeriesKMeans
from abc import ABC, abstractmethod



class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, multivariate_ts_datasets):
        pass

class ClusteringContext:
    def __init__(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def cluster(self, multivariate_ts_datasets):
        return self._strategy.cluster(multivariate_ts_datasets)



class TimeSeriesKMeansClustering(ClusteringStrategy):
    def __init__(self, n_clusters, metric_used, max_iter, max_iter_barycenter, init):
        self.n_clusters = n_clusters
        self.metric = metric_used
        self.max_iter = max_iter
        self.max_iter_barycenter= max_iter_barycenter
        # If an ndarray is passed, it should be of shape (n_clusters, ts_size, d) and gives the initial centers.
        self.init = init

    def cluster(self, multivariate_ts_datasets):
        km = TimeSeriesKMeans(n_clusters=self.n_clusters, metric=self.metric, max_iter=self.max_iter,
                              max_iter_barycenter = self.max_iter_barycenter,
                              random_state=100, init=self.init)
        cluster_label = km.fit_predict(multivariate_ts_datasets)
        centroids = km.cluster_centers_
        return centroids, cluster_label

