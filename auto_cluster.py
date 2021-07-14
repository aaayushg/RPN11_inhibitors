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

#Directly load from the saved model
Y_Kmeans=load('kmeans_saved_model.joblib')

# Selecting 10 representatives from each cluster which are
# closer to the centroid of the clusters.

temp_file="cluster_rep_scores.txt"

count  =  0
cutoff = -7.9
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

