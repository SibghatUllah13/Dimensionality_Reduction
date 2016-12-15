import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import sys
lib=__import__("lib1772576")


#Extracting Data Into Pandas

args = sys.argv[1:]
values = args[1:]
for i in range(len(values)):
    values[i] = int(values[i])
file_name = args[0]
cwd=os.getcwd()
path=cwd+"/"+file_name
Old=pd.read_csv(path, encoding='utf-8', sep=",")

#Pivot the Data and Fill NA's
New=Old.pivot(index=Old.columns[0], columns=Old.columns[1], values=Old.columns[2])
New.fillna(New.mean(axis=0, skipna = True), axis = 0, inplace=True)


#Reducing the Datasets 
Reduced=[]
Sampled_Reduced_Data=[]
for number in values:
    Reduced.append(pd.DataFrame(lib.reduce(New,number)))

#Sampling the Reduced Datasets

rand = np.random.choice(New.shape[0], size=min(250, New.shape[0]), replace=False)
New_Sample_Data=New.iloc[rand, :]
for data_set in Reduced:
    Sampled_Reduced_Data.append(data_set.iloc[rand, :])

#Calculating the Distances and Distortions
Old_Distances=lib.alldist(New_Sample_Data)
Reduced_Distances=[]
for i in range(len(Sampled_Reduced_Data)):
    Reduced_Distances.append(lib.alldist(Sampled_Reduced_Data[i]))
Distortions=[]
for f in Reduced_Distances:
    Distortions.append(lib.distortion(Old_Distances, f))

#Printing the Results and Plot
for i in range(len(Distortions)):
    print (New_Sample_Data.memory_usage(index=False).sum()/
           pd.DataFrame(Sampled_Reduced_Data[i]).memory_usage(index=False).sum(),format(np.min(Distortions[i].flatten()), '.2f'),format(np.mean(Distortions[i].flatten()), '.2f'), format(np.max(Distortions[i].flatten()), '.2f'))
fig = plt.figure()
for i in range(len(Distortions)):
    plt.hist(Distortions[i].flatten(), bins = 60, histtype='step', normed = True, label='d = {}'.format(values[i]))
plt.legend()
plt.xlabel('distortion')
plt.ylabel('frequency')
plt.show(block = True)
