#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
import sys
import argparse
from sklearn.cluster import KMeans
from sklearn import metrics
from joblib import dump, load
np.set_printoptions(threshold=sys.maxsize)

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'final'
data = pd.read_csv(file_name + '.csv', header=0)
#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining all the data X
X = data[['Ring', 'HBD','logP','RB','HBA']]

#Define number of clusters
clusters = 100

#Define which KMeans algorithm to use and fit it
#Y_Kmeans = KMeans(n_clusters = clusters,n_jobs=-2,random_state=2)
Y_Kmeans = KMeans(n_clusters = clusters,random_state=3)
Y_Kmeans.fit(X)
#dump(Y_Kmeans, 'model.joblib')

#Directly load from the saved model
#Y_Kmeans=load('model.joblib')

Y_Kmeans_labels = Y_Kmeans.labels_
Y_Kmeans_silhouette = metrics.silhouette_score(X, Y_Kmeans_labels, metric='sqeuclidean')
print("Silhouette for Kmeans: {0}".format(Y_Kmeans_silhouette))
print("Results for Kmeans: {0}".format(Y_Kmeans_labels))

'''
print(Y_Kmeans_labels)
clust = {}
n = 0
for item in Y_Kmeans_labels:
    print(item)
    f = open('final.csv')
    lines = f.readlines()
    if item in clust:
        clust[item].append(lines[n+1])
    else:
        clust[item] = [lines[n+1]]
    n +=1
    f.close()


for item in clust:
    print "Cluster ", item
    for i in clust[item]:
        print i
'''
score_file=open('all_ids_scores.txt')

#csv file to numpy array form
df=data.values

# Selecting 10 representatives from each cluster which are
# closer to the centroid of the clusters.

temp_file="test.txt"
with open(temp_file,"w") as output:
    for j in range(clusters):
        output.write('cluster='+str(j)+'\n')
        d = Y_Kmeans.transform(X)[:, j]
        ind = np.argsort(d)[0:10]
        for i in ind:
            for lines in score_file.readlines():
                if df[i][0] in lines:
                    output.write(lines)
                score_file.seek(0)

count  =  0
cutoff = -8.0
with open(temp_file) as handle:
    for line in handle:
        if line.startswith('c'):
            a=line
        if line.startswith('Z') and float(line.split()[1]) <= cutoff:
            print('Cluster_id containing highest score representative is [' + a.strip()[8:] + ']')
            print("Highest score representative from the above cluster is "+ line.split()[0] + " with a score of " + line.split()[1] + " kcal/mol")

parser = argparse.ArgumentParser()
parser.add_argument("-Rings", help = "Rings")
parser.add_argument("-HBD", help = "HBD")
parser.add_argument("-logP", help = "logP")
parser.add_argument("-RB", help = "RB")
parser.add_argument("-HBA", help = "HBA")
args=parser.parse_args()
Ring = args.Rings
HBD  = args.HBD
logP = args.logP
RB   = args.RB 
HBA  = args.HBA

new_point=[[Ring,HBD,logP,RB,HBA]]
pred=Y_Kmeans.predict(new_point)

print("Input molecule will be clustered in cluster_id = ["+' '.join(map(str, pred))+']'+'\n')

if a.strip()[8:] == ' '.join(map(str, pred)):
    print("Prediction : Input molecule can be a potent inhibitor against RPN11")
else:
    print("Prediction : Input molecule doesn't belong to any cluster of potent inhibitors of RPN11")
