#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
# sklearn package for machine learning in python:
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

df= pd.read_csv('country_data.csv')
df.head(50)

#drop the rows with NaN values
df = df.dropna(axis = 0)
print(df.head(), '\n')



# In[70]:


#KMeans
# Select the columns
X = df.iloc[:, [1, 3, 5]].values

# Construct the model using KMeans
model = KMeans(n_clusters=3, n_init='auto', random_state=0, max_iter=2000)
model.fit(X)

cluster_centers = model.cluster_centers_

# Print the center positions of the clusters
centers = model.cluster_centers_
print('Child mortality vs Health vs Income Centroids:', centers, '\n')

# Visualize the result in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Store the normalization of the color encodings based on the number of clusters
nm = Normalize(vmin=0, vmax=len(centers)-1)

# Plot the clustered data in 3D
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                     c=model.predict(X), s=50, cmap='plasma', norm=nm)

# Plot the centroids using a for loop
for i in range(centers.shape[0]):
    ax.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
            bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

# Labeling the graph with preferred columns
ax.set_title('KMeansclustster for Child_mort,Health,Income')
ax.set_xlabel(df.columns[1])  # Child_mort
ax.set_ylabel(df.columns[3])  #  Health
ax.set_zlabel(df.columns[5])  # Income
plt.show()

# Produce a legend with the distinct colors from the scatter
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
fig.savefig('cluster_childmortality__Healh_Income_plot.png')


# In[71]:


#Mean Shift
# Select the columns
X = df.iloc[:, [1,3,5]].values

# Construct the model using MeanShift
model = MeanShift()
model.fit(X)

# Extract cluster centers and labels
cluster_centers = model.cluster_centers_
labels = model.labels_

# Print the center positions of the clusters
print('Cluster Centers:', cluster_centers, '\n')

# Visualize the result in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Store the normalization of the color encodings based on the number of clusters
nm = Normalize(vmin=0, vmax=len(cluster_centers)-1)

# Plot the clustered data in 3D
scatter = ax.scatter(X[:, 0], X[:, 1],X[:, 2],
        c=labels, s=50, cmap='plasma', norm=nm)

# Plot the centroids
for i in range(len(cluster_centers)):
    ax.text(cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2], str(i), c='black',
            bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

# Labeling the graph with preferred columns
ax.set_title('Meanshiftcluster  for Child_mort,Health,Income')
ax.set_xlabel(df.columns[1])  # Child mortality
ax.set_ylabel(df.columns[3])  # Health
ax.set_zlabel(df.columns[5])  # Income

# Produce a legend with the distinct colors from the scatter
legend = ax.legend(*scatter.legend_elements(),loc="upper right", title="Clusters")
ax.add_artist(legend)
fig.savefig('Meanshift_cluster_childmortality_Health_Income_plot.png')


# In[69]:


#relationship between child mortality and Income

#select the columns
X = df.iloc[:,[1,5]].values

#Construct the model using Kmeans
model = KMeans(n_clusters = 4, n_init='auto', random_state=0, max_iter=2000)
model.fit(X)

cluster_centers = model.cluster_centers_

#print the center positions of the clusters
centers =model.cluster_centers_
print('Child mortality vs Income Centriods:',centers ,'\n')

#Visualise the result
fig2, axes = plt.subplots()


# store the normalisation of the color encodings
# based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)

# plot the clustered data
scatter2 = axes.scatter(X[:, 0], X[:, 1],
c = model.predict(X), s = 50, cmap = 'plasma', norm = nm)

# plot the centroids using a for loop
for i in range(centers.shape[0]):
  axes.text(centers[i, 0], centers[i, 1], str(i), c = 'black',
  bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))
  
#labelling the graph with prefered column   
axes.set_xlabel(df.columns[1])
axes.set_ylabel(df.columns[5])

#clusters 
cluster= model.fit_predict(X)

df['cluster']=cluster

print(df.head)

#Displaying some countries clusters
countries = df.groupby('cluster')['country']. apply(lambda X: X.head(20))
print(countries)

# produce a legend with the distint colors from the scatter
legend2 = axes.legend(*scatter2.legend_elements(),loc="upper right", title="Clusters")
axes.add_artist(legend2)
fig2.savefig('cluster_childmortality_income_plot.png')



# In[ ]:





# In[ ]:




