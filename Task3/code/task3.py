#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Import dataset and make basic preprocessing and feature engineering
df = pd.read_excel('../data/data.xlsx')
df['Area'] = df['Area'].str.replace(r'^State - ', '', regex=True) 
df['Area'] = df['Area'].str.replace(r'\s\(\d+\)$', '', regex=True)
df['Males_working'] = df['Males.1'] + df['Males.2'] + df['Males.3']
df['Females_working'] = df['Females.1'] + df['Females.2'] + df['Females.3']
df['Population_working'] = df['Males'] + df['Females']

# Create and save a stacked area chart
fig, ax = plt.subplots(figsize=(10, 6))
male_percentage = (df['Males.1'] / (df['Males.1'] + df['Females.1'])) * 100
female_percentage = (df['Females.1'] / (df['Males.1'] + df['Females.1'])) * 100
ax.stackplot(range(len(df)), male_percentage, female_percentage, labels=['Male Workforce %', 'Female Workforce %'], colors=['blue', 'orange'], alpha=0.8)
ax.set_xlabel('States')
ax.set_ylabel('Percentage')
ax.set_title('Stacked Area Chart for Workforce Participation')
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Area'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
fig.savefig('workforce_participation_chart.png')

# Create and save a scatter plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.scatter(df['Males.1'] / df['Males'], df['Females.1'] / df['Females'], color='blue', alpha=0.7, s=100)

# Plot labeling
plt.xlabel('Male Workforce')
plt.ylabel('Female Workforce')
plt.title('Scatter Plot of Male vs Female Workforce')
plt.grid(True)
plt.tight_layout()
plt.savefig('male_vs_female_workforce_scatter.png')

# Perform K-Means clustering and save the cluster visualization
df['Males_ratio_main'] = df['Males.1'] / df['Males']
df['Females_ratio_main'] = df['Females.1'] / df['Females']
X = np.array(list(zip(df['Males_ratio_main'], df['Females_ratio_main'])))

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'yellow', 'black']

for cluster in range(4):
    clustered_data = df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['Males_ratio_main'], clustered_data['Females_ratio_main'], 
                color=colors[cluster], label=f'Cluster {cluster}', s=100, alpha=0.7)

plt.xlabel('Male Workforce Proportion')
plt.ylabel('Female Workforce Proportion')
plt.title('K-Means Clustering of Male vs Female Workforce Proportions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('kmeans_clustering_workforce.png')

# More feature engineering
df['Female_Literate_Percentage'] = (df['female_litereates']/df['Females'])*100
df['Male_Literate_Percentage'] = (df['male_literates']/df['Females'])*100
df['Literate_Percentage'] = (df['literate']/df['Population'])*100

# Create and save a violin plot with a custom palette
custom_palette = {
    '0': 'red',
    '1': 'blue',
    '2': 'yellow',
    '3': 'black',
}

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Cluster', y='Literate_Percentage', palette=custom_palette)
plt.xlabel('Clusters')
plt.ylabel('Literates (%)')
plt.title('Distribution of Literate Percentage by Clusters')
plt.tight_layout()
plt.savefig('literate_percentage_by_clusters.png')

# Loop through each cluster to create and save individual plots
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.figure(figsize=(6, 4))
    sns.histplot(cluster_data['Literate_Percentage'], kde=True, color='blue')
    plt.xlabel('Literates (%)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Literacy Percentage - Cluster {cluster}')
    plt.xlim(0, 100)
    plt.savefig(f'literacy_{cluster}.png', dpi=300, bbox_inches='tight')
    plt.close()

# More Feature engineering
df['Weighted_Female_Workforce_Participation'] = df['Females_ratio_main'] * df["Female_Literate_Percentage"]
df['urbanization'] = (df['urban_pop']/df['Population'])*100

# Scatter plot for Urbanization vs Female Literacy and save the image
plt.figure(figsize=(8, 6))
plt.scatter(df['urbanization'], df['Female_Literate_Percentage'], color='blue', alpha=0.7, s=100)

plt.xlabel('Urbanization Rate (%)')
plt.ylabel('Female Literacy Rate (%)')
plt.title('Urbanization vs Female Literacy Rate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('urbanization_vs_female_literacy.png')


# Scatter plot for Urbanization vs Female Workforce Ratio and save the image
plt.figure(figsize=(8, 6))
plt.scatter(df['urbanization'], df['Females_ratio_main'], color='blue', alpha=0.7, s=100)

plt.xlabel('Urbanization Rate (%)')
plt.ylabel('Female Workforce Ratio (%)')
plt.title('Urbanization vs Female Workforce Ratio')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('urbanization_vs_female_workforce_ratio.png')


# Prepare the feature matrix for clustering
X = df[['urbanization', 'Females_ratio_main']].values

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=62)
df['Cluster'] = kmeans.fit_predict(X)

# Define cluster colors
cluster_colors = ['red', 'blue', 'yellow']

# Scatter plot with clustering
plt.figure(figsize=(8, 6))
for cluster in range(3):
    clustered_data = df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['urbanization'], clustered_data['Females_ratio_main'], 
                color=cluster_colors[cluster], label=f'Cluster {cluster}', s=100, alpha=0.7)

plt.xlabel('Urbanization Rate (%)')
plt.ylabel('Female Workforce Ratio (%)')
plt.title('Urbanization vs Female Workforce Ratio with Clustering')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('urbanization_vs_female_workforce_clustering.png')


# Create a DataFrame for plotting
plot_df = df[['Females_ratio_main', 'Males_ratio_main', 'urbanization', 'Cluster']]

# Normalize the data for better readability
plot_df[['Females_ratio_main', 'Males_ratio_main', 'urbanization']] = (
    plot_df[['Females_ratio_main', 'Males_ratio_main', 'urbanization']] - plot_df[['Females_ratio_main', 'Males_ratio_main', 'urbanization']].min()
) / (
    plot_df[['Females_ratio_main', 'Males_ratio_main', 'urbanization']].max() - plot_df[['Females_ratio_main', 'Males_ratio_main', 'urbanization']].min()
)

# Parallel coordinates plot
sns.set(style="whitegrid")
pd.plotting.parallel_coordinates(plot_df, 'Cluster', color=['red', 'yellow', 'blue'], alpha=0.7)

plt.title('Parallel Coordinates Plot: Workforce Participation with Urbanization Clusters')
plt.ylabel('Normalized Values')
plt.xlabel('Features')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('parallel_coordinates_workforce_clusters.png')

# Compute correlation matrix
corr = df[['Females_ratio_main', 'Males_ratio_main', 'urbanization', 'Cluster']].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Heatmap: Correlation Between Clusters and Features')
plt.tight_layout()
plt.savefig('heatmap_correlation_clusters_features.png')

from sklearn.preprocessing import MinMaxScaler

# Normalize the data to ensure all features are on a similar scale
scaler = MinMaxScaler()
df[['Males_ratio_main', 'Females_ratio_main', 'Weighted_Female_Workforce_Participation']] = scaler.fit_transform(
    df[['Males_ratio_main', 'Females_ratio_main', 'Weighted_Female_Workforce_Participation']]
)

# Prepare the feature matrix for clustering (including Weighted_Female_Workforce_Participation)
X = np.array(list(zip(df['Males_ratio_main'], 
                      df['Females_ratio_main'], 
                      df['Weighted_Female_Workforce_Participation'])
                      ))

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters (only Male vs Female proportions on the axes)
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'yellow', 'black']

for cluster in range(4):
    clustered_data = df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['Males_ratio_main'], clustered_data['Females_ratio_main'], 
                color=colors[cluster], label=f'Cluster {cluster}', s=100, alpha=0.7)

# Plot details
plt.xlabel('Male Workforce Proportion', fontsize=10)
plt.ylabel('Female Workforce Proportion', fontsize=10)
plt.title('K-Means Clustering with Weighted Female Workforce Participation Considered', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig('kmeans_clustering_with_weighted_female_workforce.png')

import matplotlib.pyplot as plt

# Bubble plot: Male vs Female workforce participation with bubble sizes based on weighted female participation
plt.figure(figsize=(10, 6))
bubble_sizes = df['Weighted_Female_Workforce_Participation'] * 100  # Scale for visibility

plt.scatter(df['Males_ratio_main'], df['Females_ratio_main'], 
            s=bubble_sizes, alpha=0.6, color='orange', edgecolors='black')

# Labels and title
plt.xlabel('Male Workforce Participation', fontsize=12)
plt.ylabel('Female Workforce Participation', fontsize=12)
plt.title('Bubble Plot: Workforce Participation and Weighting', fontsize=14)
plt.grid(alpha=0.4)

# Annotate states
for i, state in enumerate(df['Area']):
    plt.text(df['Males_ratio_main'].iloc[i], df['Females_ratio_main'].iloc[i], state, fontsize=8)

# Save the plot
plt.tight_layout()
plt.legend()
plt.savefig('workforce_bubble_plot.png', dpi=300)
plt.show()
