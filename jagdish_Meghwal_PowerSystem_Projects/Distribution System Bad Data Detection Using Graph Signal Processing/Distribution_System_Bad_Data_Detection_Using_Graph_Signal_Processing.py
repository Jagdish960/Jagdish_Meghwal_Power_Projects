#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Perform DBSCAN clustering with adjusted parameters
dbscan = DBSCAN(eps=3, min_samples=5)  # Increase eps value
clusters = dbscan.fit_predict(tsne_result)

# Add cluster labels to the DataFrame
voltage_df['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering of t-SNE-Embedded Voltage Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Cluster')


# Annotate outliers (points with cluster label -1)
outliers = voltage_df[voltage_df['cluster'] == -1]
plt.scatter(outliers['v1'], outliers['v2'], color='red', label='Outliers')

plt.legend()
plt.show()

# Analyze the clusters
cluster_counts = voltage_df['cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)

# Investigate cluster properties
for cluster_label, count in cluster_counts.items():
    cluster_data = voltage_df[voltage_df['cluster'] == cluster_label]
    if cluster_label == -1:
        print("\nOutliers:")
        print(f"Number of outliers: {count}")
        continue
    print(f"\nCluster {cluster_label}:")
    print(f"Number of points: {count}")
    print("Mean values:")
    print(cluster_data[['v1', 'v2', 'v3']].mean())
    print("Standard deviation:")
    print(cluster_data[['v1', 'v2', 'v3']].std())



# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import mplcursors

# Read the CSV file
df = pd.read_csv(r"C:\Users\jagdi\OneDrive\Documents\Ankush Sharma\Term Paper Smart Grid Sem2\Testing indian data.csv", delimiter='\t')

# Split the "Time,v1,v2,v3" column into separate columns
columns = df["Time,v1,v2,v3"].str.split(",", expand=True)

# Rename columns
columns.columns = ["Time", "v1","v2","v3"]

# Extract v1, v2, and v3 values
v1_values = columns["v1"].astype(float)
v2_values = columns["v2"].astype(float)
v3_values = columns["v3"].astype(float)

# Perform Fourier Transform on v1, v2, and v3
fft_v1 = np.fft.fft(v1_values)
fft_v2 = np.fft.fft(v2_values)
fft_v3 = np.fft.fft(v3_values)

# Frequencies corresponding to the Fourier Transform
frequencies = np.fft.fftfreq(len(df), 1)  # Assuming a constant time interval of 1

# Plot the Fourier Transform
plt.figure(figsize=(24, 12))

plt.subplot(3, 1, 1)
plt.plot(frequencies, np.abs(fft_v1))
plt.title('Fourier Transform of v1')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(frequencies, np.abs(fft_v2))
plt.title('Fourier Transform of v2')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(frequencies, np.abs(fft_v3))
plt.title('Fourier Transform of v3')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


# In[10]:


# Define threshold for error detection
threshold = 7000

# Classify points as error or non-error based on threshold
error_points = np.abs(tsne_transformed[:, 0] - threshold) > 1000  # Reduced threshold deviation

# Plot the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], c='blue', alpha=0.5, label='Normal Data')
plt.scatter(tsne_transformed[error_points, 0], tsne_transformed[error_points, 1], c='red', label='Error Data')
plt.title('Error Detection using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()






# In[6]:


from sklearn.cluster import DBSCAN
# Perform DBSCAN clustering with adjusted parameters
dbscan = DBSCAN(eps=3, min_samples=5)  # Increase eps value
clusters = dbscan.fit_predict(X_tsne)

# Add cluster labels to the DataFrame
voltage_df['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering of t-SNE-Embedded Voltage Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Cluster')

# Annotate outliers (points with cluster label -1)
outliers = voltage_df[voltage_df['cluster'] == -1]
plt.scatter(outliers['v1'], outliers['v2'], color='red', label='Outliers')

plt.legend()
plt.show()

# Analyze the clusters
cluster_counts = voltage_df['cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)

# Investigate cluster properties
for cluster_label, count in cluster_counts.items():
    cluster_data = voltage_df[voltage_df['cluster'] == cluster_label]
    if cluster_label == -1:
        print("\nOutliers:")
        print(f"Number of outliers: {count}")
        continue
    print(f"\nCluster {cluster_label}:")
    print(f"Number of points: {count}")
    print("Mean values:")
    print(cluster_data[['v1', 'v2', 'v3']].mean())
    print("Standard deviation:")
    print(cluster_data[['v1', 'v2', 'v3']].std())



# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Read the CSV file
df = pd.read_csv(r"C:\Users\jagdi\OneDrive\Documents\Ankush Sharma\Term Paper Smart Grid Sem2\Voltages with noise.csv", delimiter='\t')

# Split the "Time,v1,v2,v3" column into separate columns
columns = df["Time,v1,v2,v3"].str.split(",", expand=True)

# Rename columns
columns.columns = ["Time", "v1","v2","v3"]

# Extract v1, v2, and v3 values
v1_values = columns["v1"].astype(float)
v2_values = columns["v2"].astype(float)
v3_values = columns["v3"].astype(float)

# Perform Fourier Transform on v1, v2, and v3
fft_v1 = np.fft.fft(v1_values)
fft_v2 = np.fft.fft(v2_values)
fft_v3 = np.fft.fft(v3_values)

# Frequencies corresponding to the Fourier Transform
frequencies = np.fft.fftfreq(len(df), 1)  # Assuming a constant time interval of 1

# Plot the Fourier Transform
plt.figure(figsize=(24, 12))

plt.subplot(3, 1, 1)
plt.plot(frequencies, np.abs(fft_v1))
plt.title('Fourier Transform of v1')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(frequencies, np.abs(fft_v2))
plt.title('Fourier Transform of v2')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(frequencies, np.abs(fft_v3))
plt.title('Fourier Transform of v3')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Stack Fourier Transformed values for t-SNE
X = np.column_stack((np.abs(fft_v1), np.abs(fft_v2), np.abs(fft_v3)))

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', s=30)
plt.title('t-SNE Visualization of Fourier Transformed Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()

# Perform DBSCAN clustering with adjusted parameters
dbscan = DBSCAN(eps=3, min_samples=5)
clusters = dbscan.fit_predict(X_tsne)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering of t-SNE-Embedded Voltage Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Cluster')


# Annotate outliers (points with cluster label -1)
outliers = df[df['cluster'] == -1]
plt.scatter(outliers[X_tsne[:, 0]], outliers[X_tsne[:, 1]], color='red', label='Outliers')

plt.legend()
plt.show()

# Analyze the clusters
cluster_counts = df['cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)

# Investigate cluster properties
for cluster_label, count in cluster_counts.items():
    cluster_data = df[df['cluster'] == cluster_label]
    if cluster_label == -1:
        print("\nOutliers:")
        print(f"Number of outliers: {count}")
        continue
    print(f"\nCluster {cluster_label}:")
    print(f"Number of points: {count}")
    print("Mean values:")
    print(cluster_data[['v1', 'v2', 'v3']].mean())
    print("Standard deviation:")
    print(cluster_data[['v1', 'v2', 'v3']].std())


# In[17]:


# Perform DBSCAN clustering with adjusted parameters
dbscan = DBSCAN(eps=3, min_samples=5)  # Increase eps value
clusters = dbscan.fit_predict(X_tsne)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering of t-SNE-Embedded Voltage Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Cluster')

# Annotate outliers (points with cluster label -1)
outliers = df[df['cluster'] == -1]
plt.scatter(outliers['v1'], outliers['v2'], color='red', label='Outliers')

plt.legend()
plt.show()

# Analyze the clusters
cluster_counts = df['cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)

# Investigate cluster properties
for cluster_label, count in cluster_counts.items():
    cluster_data = df[df['cluster'] == cluster_label]
    if cluster_label == -1:
        print("\nOutliers:")
        print(f"Number of outliers: {count}")
        continue
    print(f"\nCluster {cluster_label}:")
    print(f"Number of points: {count}")
    print("Mean values:")
    print(cluster_data[['v1', 'v2', 'v3']].mean())
    print("Standard deviation:")
    print(cluster_data[['v1', 'v2', 'v3']].std())



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




