

from sklearn.metrics.cluster import calinski_harabasz_score as sklearn_calinski_harabasz
from sklearn.metrics.cluster import davies_bouldin_score as sklearn_bouldin_score

from scipy.spatial.distance import cdist

from tslearn.metrics import cdist_dtw, cdist_soft_dtw_normalized

from tslearn.utils import to_time_series_dataset, to_time_series


def calinski_score(X, labels, metric=None, sample_size=None,
                     metric_params=None, n_jobs=None, verbose=0,
                     random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples (cf.  [1]_ and
    [2]_).

    Read more in the `scikit-learn documentation
    <http://scikit-learn.org/stable/modules/clustering.html\
    #silhouette-coefficient>`_.

    Parameters
    ----------
    X : array [n_ts, n_ts] if metric == "precomputed", or, \
             [n_ts, sz, d] otherwise
        Array of pairwise distances between time series, or a time series
        dataset.
    labels : array, shape = [n_ts]
         Predicted labels for each time series.
    metric : string, callable or None (default: None)
        The metric to use when calculating distance between time series.
        Should be one of {'dtw', 'softdtw', 'euclidean'} or a callable distance
        function or None.
        If 'softdtw' is passed, a normalized version of Soft-DTW is used that
        is defined as `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`.
        If X is the distance array itself, use ``metric="precomputed"``.
        If None, dtw is used.
    sample_size : int or None (default: None)
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function,
        just as for the `metric_params` parameter.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------

    >>> from tslearn.generators import random_walks
    >>> from tslearn.metrics import cdist_dtw
    >>> numpy.random.seed(0)
    >>> X = random_walks(n_ts=20, sz=16, d=1)
    >>> labels = numpy.random.randint(2, size=20)
    >>> silhouette_score(X, labels, metric="dtw")  # doctest: +ELLIPSIS
    0.13383800...
    >>> silhouette_score(X, labels, metric="euclidean")  # doctest: +ELLIPSIS
    0.09126917...
    >>> silhouette_score(X, labels, metric="softdtw")  # doctest: +ELLIPSIS
    0.17953934...
    >>> silhouette_score(X, labels, metric="softdtw",
    ...                  metric_params={"gamma": 2.}) \
    # doctest: +ELLIPSIS
    0.17591060...
    >>> silhouette_score(cdist_dtw(X), labels,
    ...                  metric="precomputed")  # doctest: +ELLIPSIS
    0.13383800...
    """
    sklearn_metric = None
    if metric_params is None:
        metric_params_ = {}
    else:
        metric_params_ = metric_params.copy()
    for k in kwds.keys():
        metric_params_[k] = kwds[k]
    if "n_jobs" in metric_params_.keys():
        del metric_params_["n_jobs"]
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw" or metric is None:
        sklearn_X = cdist_dtw(X, n_jobs=n_jobs, verbose=verbose,
                              **metric_params_)
    elif metric == "softdtw":
        sklearn_X = cdist_soft_dtw_normalized(X, **metric_params_)
    elif metric == "euclidean":
        X_ = to_time_series_dataset(X)
        X_ = X_.reshape((X.shape[0], -1))
        sklearn_X = cdist(X_, X_, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))

        def sklearn_metric(x, y):
            return metric(to_time_series(x.reshape((sz, d)),
                                         remove_nans=True),
                          to_time_series(y.reshape((sz, d)),
                                         remove_nans=True))
    metric = "precomputed" if sklearn_metric is None else sklearn_metric
    return sklearn_calinski_harabasz(X=sklearn_X,
                              labels=labels)
    
    
def davies_bouldin_score(X, labels, metric=None, sample_size=None,
                     metric_params=None, n_jobs=None, verbose=0,
                     random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples (cf.  [1]_ and
    [2]_).

    Read more in the `scikit-learn documentation
    <http://scikit-learn.org/stable/modules/clustering.html\
    #silhouette-coefficient>`_.

    Parameters
    ----------
    X : array [n_ts, n_ts] if metric == "precomputed", or, \
             [n_ts, sz, d] otherwise
        Array of pairwise distances between time series, or a time series
        dataset.
    labels : array, shape = [n_ts]
         Predicted labels for each time series.
    metric : string, callable or None (default: None)
        The metric to use when calculating distance between time series.
        Should be one of {'dtw', 'softdtw', 'euclidean'} or a callable distance
        function or None.
        If 'softdtw' is passed, a normalized version of Soft-DTW is used that
        is defined as `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`.
        If X is the distance array itself, use ``metric="precomputed"``.
        If None, dtw is used.
    sample_size : int or None (default: None)
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function,
        just as for the `metric_params` parameter.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> from tslearn.metrics import cdist_dtw
    >>> numpy.random.seed(0)
    >>> X = random_walks(n_ts=20, sz=16, d=1)
    >>> labels = numpy.random.randint(2, size=20)
    >>> silhouette_score(X, labels, metric="dtw")  # doctest: +ELLIPSIS
    0.13383800...
    >>> silhouette_score(X, labels, metric="euclidean")  # doctest: +ELLIPSIS
    0.09126917...
    >>> silhouette_score(X, labels, metric="softdtw")  # doctest: +ELLIPSIS
    0.17953934...
    >>> silhouette_score(X, labels, metric="softdtw",
    ...                  metric_params={"gamma": 2.}) \
    # doctest: +ELLIPSIS
    0.17591060...
    >>> silhouette_score(cdist_dtw(X), labels,
    ...                  metric="precomputed")  # doctest: +ELLIPSIS
    0.13383800...
    """
    sklearn_metric = None
    if metric_params is None:
        metric_params_ = {}
    else:
        metric_params_ = metric_params.copy()
    for k in kwds.keys():
        metric_params_[k] = kwds[k]
    if "n_jobs" in metric_params_.keys():
        del metric_params_["n_jobs"]
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw" or metric is None:
        sklearn_X = cdist_dtw(X, n_jobs=n_jobs, verbose=verbose,
                              **metric_params_)
    elif metric == "softdtw":
        sklearn_X = cdist_soft_dtw_normalized(X, **metric_params_)
    elif metric == "euclidean":
        X_ = to_time_series_dataset(X)
        X_ = X_.reshape((X.shape[0], -1))
        sklearn_X = cdist(X_, X_, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))

        def sklearn_metric(x, y):
            return metric(to_time_series(x.reshape((sz, d)),
                                         remove_nans=True),
                          to_time_series(y.reshape((sz, d)),
                                         remove_nans=True))
    metric = "precomputed" if sklearn_metric is None else sklearn_metric
    return sklearn_bouldin_score(X=sklearn_X,
                              labels=labels)
    
 
def save_metrics(path, txt):
    file=path+"/Metrics_Result.txt"
    f = open(file, "a")
    f.write(txt)
    f.close()