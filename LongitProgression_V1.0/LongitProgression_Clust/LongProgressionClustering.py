
from tslearn.clustering import silhouette_score
from Util_Metrics  import metrics
from config.conf_pars import Configurator
from Util_DataProcessing.dataProcessor import *
from Util_Learning.dataLearning import ClusteringContext, TimeSeriesKMeansClustering
from Util_Metrics.metrics import save_metrics
import warnings        

     

warnings.simplefilter('ignore')

def main(ConfigPath):

    conf = Configurator(ConfigPath)
    """
    parametri da settare all'inizio
    """
    time_points = conf.parameters['longitudinal_setting']['time_points']
    features = conf.parameters['longitudinal_setting']['features']
    num_clusters = conf.parameters['clustering_setting']['num_clusters']
    metric_used = conf.parameters['clustering_setting']['metric_used']
    max_iter= conf.parameters['clustering_setting']['max_iter']
    max_iter_barycenter=conf.parameters['clustering_setting']['max_iter_barycenter']
    init=conf.parameters['clustering_setting']['init']
    normalization_method = conf.parameters['clustering_setting']['normalization_method']
    dataset_path=conf.parameters['path_data']
    sep_data=conf.parameters['sep_data']
    output= conf.parameters['path_output']
    
    feature_dataset_list, subject_list, RID, Param1, Param2 = get_longitudinal_tensor(dataset_path,sep_data, conf, features,normalization_method )
    """
    Matrix creation in the form [ts,time_points,n_features] 
    where ts is the number of time series samples
    """
    multivariate_ts_datasets = get_multivariate_ts(feature_dataset_list, subject_list, time_points)
    print(multivariate_ts_datasets)
    """
    Multivariate longitudinal KMeans execution
    """
    context = ClusteringContext(TimeSeriesKMeansClustering(n_clusters=num_clusters, metric_used=metric_used, max_iter=max_iter, max_iter_barycenter=max_iter_barycenter, init=init))
    centroids, cluster_label = context.cluster(multivariate_ts_datasets)
    """
    Save Clusters
    """
    save_clusters_(RID, num_clusters, cluster_label, conf, dataset_path,sep_data,to_print=True, to_save = True)
 
    """
    Evaluate Clustering metrics
    """
    silh= silhouette_score(multivariate_ts_datasets, cluster_label, metric=metric_used)  
    CHI=metrics.calinski_score(multivariate_ts_datasets, cluster_label,metric=metric_used)    
    DBI=metrics.davies_bouldin_score(multivariate_ts_datasets, cluster_label,metric=metric_used)
   

    """
    Plotting Figures and save metrics
    """
    multivariant_plot(multivariate_ts_datasets, features, num_clusters, time_points, centroids, Param1, Param2, conf, toShow=True, toSave= True)
    res_metrics="For K = "+ str(num_clusters)+ " clusters: \n The Silhouette Score is "+str(silh)+"\n"+"The Calinski_harabasz Score for "+ str(num_clusters)+ " is "+str(CHI)+"\n"+"Davies_Bouldin_Score for "+ str(num_clusters)+ " is "+str(DBI)+"\n"
    save_metrics(output,res_metrics)
   

 
 
