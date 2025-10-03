# LongitProgression - Usage Example
This README file describes the steps for performing a whole pipeline on the provided example dataset.
 
 # 1) In the terminal, type the following command: 
  
  python MainGui


 # 2) Startup GUI and input parameters:
 Import the dataset and enter all the information related to the study in the Dataset Description tab, and establish the output folder.
 For executing the longitudinal clustering on the provided dataset, insert data according to the structure reported in the following figure. 
 
 ![3](https://github.com/user-attachments/assets/05a01834-d966-4d39-b54b-e14f0059281d)
 
# 3) Configure Cluster parameters:

Choose the normalization method, distance metric, and initialization parameters for k-means. 
For this example, insert data as reported in the figure. 

![4](https://github.com/user-attachments/assets/e286c868-47f6-470c-9596-9fd81211657b)

# 4) Start Clustering
Click on the Start Clustering button to start the clustering process. 

# 5) Statistical Analysis
Click on Compute Statics button for performing statistical analysis on the clustering results.

# 6) Outputs
At the end of the process, the tool generates the following set of outputs that you find in the output folder:
- CSV Cluster files: One CSV file (three in this example) is produced per
cluster, each containing the records of individuals belonging to that cluster.

- Trend of cluster trajectories: A PDF file is also generated, containing a visu-
alization of the joint trajectories identified for each cluster. In our example,
this is shown in Figure 5, which illustrates how the clinical markers (i.e., Creatinine and Potassium)
evolve over time within each cluster, outlining three
distinct subject profiles that reflect different patterns of kidney disease pro-
gression. In particular, in this example, Cluster 1 includes individuals without
signs of kidney dysfunction, who maintain low creatinine and normal potassium levels over time. Cluster 2 groups individuals with mild kidney impairment,
characterized by creatinine levels around two and fluctuating potassium
levels; however, their condition appears stable. Finally, Cluster 3 includes individuals
with initially severe kidney issues, whose condition shows moderate
improvement, likely due to treatment administered during hospitalization.

- Metrics Result.txt: This file contains the evaluation metrics values corresponding to the specific instance of the longitudinal clustering performed.
  In our example, the results indicate that the identified clusters are well-defined, although they may not represent the optimal clustering solution.
  Multiple clustering runs can be performed to identify the configuration that yields the best overall performance.

- Statistic result.txt: This file contains the results of a statistical analysis per-
formed on the identified clusters at each time point. An excerpt from this file,
shown below for our example, demonstrates statistically significant differences
among the three clusters. Furthermore, the post-hoc analysis highlights significant differences between specific pairs of clusters.

-  Significance matrices: These are a set of PDF files displaying the statistical
significance results for each performed post-hoc test. To illustrate, we present
the matrices related to Creatinine and Potassium at time points two and eight. From these matrices, for example, researchers can observe that at
time point two, there is a statistically significant difference among individuals
in all three clusters regarding Creatinine. Conversely, for Potassium, a significant difference is observed only between Cluster 1 and Cluster 2. Similar
conclusions can be drawn by examining the results at time point eight.
