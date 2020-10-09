import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
data=pd.read_csv("Mall_Customers.csv")

X=data.iloc[:,[3,4]].values


#Constructing a dendrogram
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Checking Optimal Number of Clusters")
plt.xlabel("Data points")
plt.ylabel("Euclidean Distances")
plt.show()

#From Dendrogram we get optimal number of clusters=5

from sklearn.cluster import AgglomerativeClustering
Hc=AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc=Hc.fit_predict(X)

#Visualising the results
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c="red",label="Careful")
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c="blue",label="Standard")
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c="green",label="Target")
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c="cyan",label="Careless")
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c="magenta",label="Sensible")


plt.title("Cluster of Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()