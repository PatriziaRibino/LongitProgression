
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase, Colorbar
from seaborn import heatmap
from matplotlib import colors

from config.conf_pars import Configurator

def save_Stats(path, txt):
    file=path+"/Statistic_Result.txt"
    f = open(file, "a")
    f.write(txt)
    f.close()
   
def anova(ConfigPath,num_clusters):
    conf = Configurator(ConfigPath)
    output_dir =  conf.parameters['path_output']
    time_points=conf.parameters['longitudinal_setting']['time_points']
    ID_feature = conf.parameters['data_setting']['ID']
    time_point_feature=conf.parameters['data_setting']['time_point']
            
    for point in range(1,time_points+1):
        df_cluster=[]
       
        for c in range(1, num_clusters+1):
             
            clusterName=output_dir+"/Cluster"+str(c)+".csv"
            temp_csv = pd.read_csv(clusterName, sep = ',')  
            temp_csv_single_point=temp_csv[(temp_csv[conf.parameters['data_setting']['time_point']] == point)] 
            temp_csv_single_point=temp_csv_single_point.drop([ID_feature,time_point_feature], axis=1) 
            Feature_List=temp_csv_single_point.columns
            df_cluster.append(temp_csv_single_point)
       
        
        for f in range(len(Feature_List)):
                Feature=Feature_List[f]   
                    
                check=False
                Res_matrix_toPrint=pd.DataFrame(columns=np.arange(1,num_clusters+1),index=np.arange(1,num_clusters+1))
                
                for i in range(num_clusters):
                    Cl=df_cluster[i][Feature]
                    Cl=list(Cl)
                               
                    res, p_value_A =stats.shapiro(Cl)
                        
                         
                    if p_value_A<0.05:
                        #not normal
                        check=True
                        print(i, " ", p_value_A, " ",Feature," ", "Not Normal")
                    else:
                        print(i, " ", p_value_A, " ",Feature," ", "Normal")
                
                if(check==True):
                    Test="Kruskal"
                   
                    if (num_clusters==2):
                        t_statistic_A, p_value_A= stats.kruskal(df_cluster[0][Feature],df_cluster[1][Feature])
                        res=sp.posthoc_mannwhitney([df_cluster[0][Feature],df_cluster[1][Feature]], p_adjust='fdr_tsbh',sort=True)    
                    if (num_clusters==3):
                        t_statistic_A, p_value_A= stats.kruskal(df_cluster[0][Feature],df_cluster[1][Feature],df_cluster[2][Feature])   
                        res=sp.posthoc_mannwhitney([df_cluster[0][Feature],df_cluster[1][Feature],df_cluster[2][Feature]],p_adjust='fdr_tsbh',sort=True) 
		          
                
                else: 
                    Test="Anova" 
                    if (num_clusters==2):                
                        t_statistic_A, p_value_A=f_oneway(df_cluster[0][Feature],df_cluster[1][Feature]) 
                        res=sp.posthoc_ttest(df_cluster[0][Feature],df_cluster[1][Feature],p_adjust='fdr_tsbh',sort=True)
                    if (num_clusters==3):                
                        t_statistic_A, p_value_A=f_oneway(df_cluster[0][Feature],df_cluster[1][Feature],df_cluster[2][Feature]) 
                        res=sp.posthoc_ttest([df_cluster[0][Feature],df_cluster[1][Feature],df_cluster[2][Feature]],p_adjust='fdr_tsbh',sort=True)
                        
                for icol in range(1,num_clusters+1):
                
                    for jcol in range(1,num_clusters+1):
                        if ((res[icol][jcol] < 0.001) and (res[icol][jcol]>= 0)):
                           Res_matrix_toPrint.loc[icol,jcol]=1
                        if ((res[icol][jcol] < 0.01) and (res[icol][jcol]>= 0.001)):
                           Res_matrix_toPrint.loc[icol,jcol]=2
                        if ((res[icol][jcol] < 0.05) and (res[icol][jcol]>= 0.01)):
                           Res_matrix_toPrint.loc[icol,jcol]=3
                        if (res[icol][jcol]>= 0.05):
                           Res_matrix_toPrint.loc[icol,jcol]=0  
                             
                if p_value_A<0.05:
                    subString="Result: Statistical Independent Clusters: "+str(p_value_A)
                    txt_ToSave="Time"+str(point)+"\nFeature "+Feature+"\nType of Test executed: "+Test+"\n"+subString+"\n"+"Result PostHoc Test:\n"+str(res)+"\n \n"
                else: 
                    subString="Result: NON Statistical Independent Clusters: "+str(p_value_A)
                    txt_ToSave="Time"+str(point)+"\nFeature "+Feature+"\n"+ "Type of Test executed: "+Test+"\n"+subString+"\n \n"
                
                save_Stats(output_dir,txt_ToSave)
                
                
                   
                fig,ax=plt.subplots(figsize=(10,6))
                ax.set_title("Significance plot for "+Feature+"_Time"+str(point))
               
                cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
            
                np.fill_diagonal(Res_matrix_toPrint.values, -1)
                heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.8, 0.35, 0.04, 0.3]}
          
            
                hax = heatmap(Res_matrix_toPrint.astype(np.float64), vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1, cbar=False,ax=ax,square= True,linecolor= '0.5',linewidths= 1)
            
                cbar_ax = hax.figure.add_axes( [0.8, 0.35, 0.04, 0.3])
                cbar = ColorbarBase(cbar_ax, cmap=(ListedColormap(cmap[2:] + [cmap[1]])), norm=colors.NoNorm(),
                                            boundaries=[0, 1, 2, 3, 4])
                cbar.set_ticks(list(np.linspace(0, 3, 4)), labels=[
                                       'p < 0.001', 'p < 0.01', 'p < 0.05', 'NS'])
                cbar.outline.set_linewidth(1)
                cbar.outline.set_edgecolor('0.5')
                cbar.ax.tick_params(size=0)
                cbar.outline.set_linewidth(1)
                cbar.outline.set_edgecolor('0.5')
                cbar.ax.tick_params(size=0)
                    
                plt.savefig(output_dir+"/"+Feature+"_Time"+str(point)+".pdf", format="pdf", bbox_inches="tight")
                   
      