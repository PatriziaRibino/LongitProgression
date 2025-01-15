import tkinter as tk                     
from tkinter import ttk 
from tkinter import *
import json
import os
from urllib.parse import urlsplit
from tkinter import filedialog

from LongitProgression_Clust import LongProgressionClustering as clust
from Statistics.Anova_test import anova

def get_filename_from_url(url=None):
    if url is None:
        return None
    urlpath = urlsplit(url).path
    return os.path.basename(urlpath)

ListofFeatures=[]
newFeature=""
check=0

def new():
   global lastrow
   global check
   global newFeature
   global prec
   f=Entry(tab1)
   lastrow=lastrow+1
   f.grid(row=lastrow, column=1)     
   newFeature=prec.get()
   ListofFeatures.append(newFeature)
   check=1
   prec=f

def stop():   
   newFeature=prec.get()
   ListofFeatures.append(newFeature)
   print(ListofFeatures)

def browse(): 
    global dbpath  
    dbpath = filedialog.askopenfilename(initialdir=os.getcwd()) 
    db=get_filename_from_url(dbpath)
    path_data = ttk.Label(tab1, text=db)
    path_data.grid(row=1, column=3)

def browse_output(): 
    global output_path  
    output_path = filedialog.askdirectory(initialdir=os.getcwd())
    out=get_filename_from_url(output_path)
    path_data = ttk.Label(tab1, text=out)
    path_data.grid(row=2, column=3)
   
def check():
  
    data = {
      "path_data": dbpath,
      "sep_data":Sep.get(),
      "path_output": output_path,
    
      "longitudinal_setting": {
        "time_points": int(Time_points.get()),
        "features": ListofFeatures
        },
    
      "clustering_setting": {
        "normalization_method":normalization_method,
        "metric_used": selected_metric,
        "num_clusters": int(Num_clusters.get()),
         "max_iter":int(Max_iter.get()),
         "max_iter_barycenter":50,
         "init":Init_method
        },
    
    
      "data_setting": {
      "ID": ID.get(),
    "time_point": time_point.get()
  },

    
      "plotting_setting": {
                    "nrows_subplot": 3,
                    "ncolumns_subplot": 3,
                    "major_size": 5,
                    "major_pad": 5,
                    "xtick_labelsize": 5,
                    "ytick_labelsize": 5,
                    "grid_color": "white",
                    "grid_linestyle": "-",
                    "grid_linewidth": 5,
                    "lines_linewidth": 2,
                    "lines_color": "g",
                    "axes_facecolor": "lavender"
      }
    
    
    }
        
    print(data)
    files = [('JSON File', '*.json')]
    fileName='conf.json'
    filepos = filedialog.asksaveasfile(initialdir="../Config",filetypes = files,defaultextension = json,initialfile='conf')
    print(filepos.name)
    global configPath
    configPath =filepos.name
    writeToJSONFile(filepos, fileName, data)

def start():
   
    clust.main(configPath)
    
def statistic():
   
     anova(configPath,int(Num_clusters.get()))
    
def selected_item():
    global normalization_method
    for i in listbox.curselection():
        normalization_method=listbox.get(i)
        print(listbox.get(i))


def select_metric():
    global selected_metric
    for i in listbox2.curselection():
        selected_metric=listbox2.get(i)
        print(listbox2.get(i))
        
def select_Init_method():
    global Init_method 
    for i in listbox3.curselection():
        Init_method=listbox3.get(i)
        print(listbox3.get(i))
 

def writeToJSONFile(path, fileName, data):
        json.dump(data, path)
        
window = tk.Tk()
window.geometry('650x450')
window.title('LongitProgression')
tabControl = ttk.Notebook(window) 


tab1 = ttk.Frame(tabControl) 
tab2 = ttk.Frame(tabControl) 
  
tabControl.add(tab1, text ='Dataset Description') 
tabControl.add(tab2, text ='Clustering Settings') 
tabControl.pack(expand = 1, fill ="both") 

label1=ttk.Label(tab1, text="  ")
label1.grid(row=0, column=1)

path_lab = ttk.Label(tab1, text="Dataset Path:")
path_lab.grid(row=1, column=0)

path=ttk.Button(tab1,text='Browse',command = browse).grid(row=1, column=1)


path_output = ttk.Label(tab1, text="Path for Results:")
path_output.grid(row=2, column=0)
pathoutput=ttk.Button(tab1,text='Browse',command = browse_output).grid(row=2, column=1)


sep_data = ttk.Label(tab1, text="Dataset entry separator")
sep_data.grid(row=3, column=0)
Sep = ttk.Entry(tab1)
Sep.grid(row=3, column=1)




ID_lab=ttk.Label(tab1, text="Name of the ID Feature")
ID_lab.grid(row=4, column=0)
ID=ttk.Entry(tab1)
ID.grid(row=4, column=1)


time_point_lab=ttk.Label(tab1, text="Name of the time_point Feature")
time_point_lab.grid(row=5, column=0)
time_point=ttk.Entry(tab1)
time_point.grid(row=5, column=1)

time_points_lab=ttk.Label(tab1, text="Number of Time points ")
time_points_lab.grid(row=6, column=0)
Time_points=ttk.Entry(tab1)
Time_points.grid(row=6, column=1)

features=ttk.Label(tab1, text="Features Names")
features.grid(row=7, column=0)
lastrow=7

Features = ttk.Entry(tab1)
Features.grid(row=7, column=1)
prec=Features
new = ttk.Button(tab1,text='add',command = new).grid(row=7, column=2)
finish = ttk.Button(tab1,text='save',command = stop).grid(row=8, column=2)


label2=ttk.Label(tab2, text=" ")
label2.grid(row=0, column=5)


normalization_method = ttk.Label(tab2, text="Normalization Method:")
normalization_method.grid(row=1, column=4)

listbox = Listbox(tab2,  height=2,selectmode=SINGLE,exportselection=False)
listbox.insert(1, "MinMaxScaler")
listbox.insert(2, "StandardScaler")


listbox.grid(row=1, column=5)
btn = ttk.Button(tab2, text='Select', command=selected_item).grid(row=2, column=5)


metric_used = ttk.Label(tab2, text="Metric")
metric_used.grid(row=4, column=4)
listbox2 = Listbox(tab2,  height=2,selectmode=SINGLE,exportselection=False)
listbox2.insert(1, "softdtw")
listbox2.insert(2, "dtw")

listbox2.grid(row=4, column=5)
btn = ttk.Button(tab2, text='Select', command=select_metric).grid(row=5, column=5)

init = ttk.Label(tab2, text="Init")
init.grid(row=7, column=4)
listbox3 = Listbox(tab2,  height=2,selectmode=SINGLE,exportselection=False)
listbox3.insert(1, "k-means++")
listbox3.insert(2, "random")

listbox3.grid(row=7, column=5)
btn = ttk.Button(tab2, text='Select', command=select_Init_method).grid(row=8, column=5)

num_clusters = ttk.Label(tab2, text="Num_clusters")
num_clusters.grid(row=10, column=4)
Num_clusters = Entry(tab2)
Num_clusters.grid(row=10, column=5)

max_iter = ttk.Label(tab2, text="Max_iter")
max_iter.grid(row=11, column=4)
Max_iter = ttk.Entry(tab2)
Max_iter.grid(row=11, column=5)



submit = ttk.Button(tab2,text='Save All',command = check)
submit.grid(row=12, column=5)

startClust = ttk.Button(tab2,text='Start Clustering',command = start)
startClust.grid(row=13, column=5)

startStat = ttk.Button(tab2,text='Compute Statistics',command = statistic)
startStat.grid(row=15, column=6)



window.mainloop()   