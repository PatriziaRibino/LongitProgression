
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_longitudinal_tensor(file_path,sep, config,features,normalization_method):
    df_cluster = pd.read_csv(file_path, sep=sep)
    subjects = df_cluster[config.parameters['data_setting']['ID']]
    subject_list = list(set(subjects))

    RID = df_cluster.loc[(df_cluster[config.parameters['data_setting']['time_point']] == 1), [config.parameters['data_setting']['ID']]]
    feature_dataset_list = []
    df_whole = RID
    
    Par1 = []
    Par2 = []
    time_points = config.parameters['longitudinal_setting']['time_points']
   
    # Iterate through features
    for f in features:
        # Initialize a list to store the DataFrames for each visit
        feature_list = []

        # Collect DataFrames for all visits
       
        for time in range(1, time_points+1):
            feature_list.append(df_cluster.loc[df_cluster[config.parameters['data_setting']['time_point']] == time, [config.parameters['data_setting']['ID'], f]])
           
        # Merge all visit DataFrames on 'Subject'
        
        df_Feature = feature_list[0]
        
        count=1
        
        for feature_df in feature_list[1:]:
            
            df_Feature = pd.merge(df_Feature, feature_df, on=config.parameters['data_setting']['ID'], suffixes=("_l"+str(count), "_r"+str(count)))
            count=count+1
           

        # Merge with the whole DataFrame and drop the 'Subject' column
        df_whole = pd.merge(df_whole, df_Feature, on=config.parameters['data_setting']['ID'])
       
        df_Feature = df_Feature.drop(config.parameters['data_setting']['ID'], axis=1)
        

        if normalization_method=="MinMaxScaler":
            norm_Feature=((df_Feature-df_Feature.values.min())/(df_Feature.values.max()-df_Feature.values.min()))
            Param1=df_Feature.values.min()
            Param2=df_Feature.values.max()
           
        if normalization_method=="StandardScaler":
            norm_Feature=((df_Feature-df_Feature.values.mean())/df_Feature.values.std())
            Param1=df_Feature.values.mean()
            Param2=df_Feature.values.std()
       
        #Salvo questi valori per la denormalizzazione nei grafici
        Par1.append(Param1)
        Par2.append(Param2)
        feature_dataset_list.append(norm_Feature.values)
    return feature_dataset_list, subject_list,RID,Par1, Par2


def get_multivariate_ts(feature_dataset_list, subject_list, nvisits):
    time_series = []
   
    for i in range(len(subject_list) - 1):
        time_seriestemp = []
        for j in range(nvisits):
            vect = []
            for h in range(len(feature_dataset_list)):
                vect.append(feature_dataset_list[h][i][j])
            time_seriestemp.append(vect)
        time_series.append(time_seriestemp)
    print(time_series)
    multivariate_ts_datasets = np.array(time_series)
    return multivariate_ts_datasets


def multivariant_plot(multivariate_ts_datasets,features, num_clusters, nvisits, centroids, Param1, Param2, conf, toShow=True, toSave=True ):
    nfeatures = multivariate_ts_datasets.shape[2]
    
    normalization_method = conf.parameters['clustering_setting']['normalization_method']
    output= conf.parameters['path_output']
    
    
    color = ["bo-", "go-", "ro-", "co-"]
    plt.figure(figsize=(11, 6))

    for h in range(nfeatures):

        for cluster in range(num_clusters):

            newcentroids = []

            plt.rcParams.update({
                # "font.weight": "bold",
                "xtick.major.size": conf.parameters["plotting_setting"]["major_size"],
                "xtick.major.pad": conf.parameters["plotting_setting"]["major_pad"],
                "xtick.labelsize": conf.parameters["plotting_setting"]["xtick_labelsize"],
                "ytick.labelsize": conf.parameters["plotting_setting"]["ytick_labelsize"],
                "grid.color": conf.parameters["plotting_setting"]["grid_color"],
                "grid.linestyle": conf.parameters["plotting_setting"]["grid_linestyle"],
                # "grid.linewidth": conf.parameters["plotting_setting"]["major_size"],
                "lines.linewidth": conf.parameters["plotting_setting"]["lines_linewidth"],
                "lines.color": conf.parameters["plotting_setting"]["lines_color"],
                "axes.facecolor": conf.parameters["plotting_setting"]["axes_facecolor"]
            })

            centroid = []
            for k in range(nvisits):
                if normalization_method=="MinMaxScaler":
                    denormalised = centroids[cluster][k][h] * (Param2[h] - Param1[h]) + Param1[h]
                if normalization_method=="StandardScaler":
                    denormalised = centroids[cluster][k][h] * Param2[h] + Param1[h]
                centroid.append(denormalised)
            newcentroids.append(centroid)

            plt.subplot(conf.parameters['plotting_setting']['nrows_subplot'], conf.parameters['plotting_setting']['ncolumns_subplot'], h + 1)

            plt.grid(True)
            label_leg = "Cluster " + str(cluster + 1)
            plt.plot(centroid, color[cluster], label=label_leg)
            plt.xlim(0, nvisits - 1)
            plt.xticks(np.arange(0, nvisits, step=1))
          
            plt.text(0.8, 1.02, features[h], transform=plt.gca().transAxes)

            if (h == 1):
                plt.legend(loc='upper center', bbox_to_anchor=(-0.8, 1.4), ncol=4)

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    # plt.legend()
    if toSave is True:
        filename=output+"/ClustersTrend.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    
    if toShow==True:
        plt.show()
   
    

def save_clusters_(RID, num_clusters, cluster_label, conf , file_path,sep=";",to_print=True, to_save = True):
   
    ID_ = conf.parameters['data_setting']['ID']
    path=conf.parameters['path_output']
    
    subjList = RID.values
    
    for clust in range(num_clusters):
        Cluster = []
        Cluster_Index = np.where(cluster_label == clust)[0]
        for elem in Cluster_Index:
            subj = subjList[elem]
            Cluster.append(subj[0])
        df = pd.read_csv(file_path, sep=sep)
        clustertosave = df[ID_].isin(Cluster)
        cluster_data = df[clustertosave]
            
        if to_print:
            print(str(len(Cluster)), " Patients ID in Cluster ", clust, " ", Cluster)

        # Salvo i Cluster
        if to_save:
            cluster_data.to_csv(path+"/Cluster"+str(clust+1)+".csv",index=False)
            

