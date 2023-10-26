import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import networkx as nx
import numpy as np

# Load the hero data (replace 'heroes7.csv' with your data file)
data_ori = pd.read_csv('./heroes7.csv', encoding='utf-8')

# Define the features to be used
features = ['MaxHP', 'HPGrowth', 'InitialHP', 'MaxMana', 'ManaGrowth', 'InitialMana',
            'MaxPhysicalAttack', 'PhysicalAttackGrowth', 'InitialPhysicalAttack',
            'MaxPhysicalDefense', 'PhysicalDefenseGrowth', 'InitialPhysicalDefense',
            'MaxHPRegen', 'HPRegenGrowth', 'InitialHPRegen', 'MaxManaRegen',
            'ManaRegenGrowth', 'InitialManaRegen', 'MaxAttackSpeed', 'AttackRange']

# Select the relevant features
data = data_ori[features]

# Data Visualization: Correlation Heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Feature selection: Remove highly correlated features
features_remain = ['MaxHP', 'InitialHP', 'MaxMana', 'MaxPhysicalAttack', 'InitialPhysicalAttack',
                   'MaxPhysicalDefense', 'InitialPhysicalDefense', 'MaxHPRegen', 'MaxManaRegen', 'InitialManaRegen',
                   'MaxAttackSpeed', 'AttackRange']
data['MaxAttackSpeed'] = data['MaxAttackSpeed'].apply(lambda x: float(x.strip('%')) / 100)
data['AttackRange'] = data['AttackRange'].map({'Ranged': 1, 'Melee': 0})

# Data Normalization: Z-Score
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Dimensionality Reduction: Principal Component Analysis (PCA)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Dimensionality Reduction: t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(data)

# Visualization: 2D Scatter Plot (PCA and t-SNE)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data_ori['Hero'], cmap='tab20')
plt.title('PCA')
plt.subplot(1, 2, 2)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data_ori['Hero'], cmap='tab20')
plt.title('t-SNE')
plt.show()

# Clustering: Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
data['Cluster'] = gmm.fit_predict(data)

# Visualize the clusters in 2D
fig = px.scatter(data, x='MaxHP', y='MaxMana', color='Cluster', title='Clustering Results (GMM)')
fig.show()

# Network Analysis: Create a graph
G = nx.Graph()
for hero in data_ori['Hero']:
    G.add_node(hero)

# Calculate cluster statistics
cluster_stats = data['Cluster'].value_counts().reset_index()
cluster_stats.columns = ['Cluster', 'Count']

# Display the cluster statistics
fig = px.bar(cluster_stats, x='Cluster', y='Count', labels={'Count': 'Number of Heroes in Cluster'})
fig.update_layout(title='Cluster Statistics')
fig.show()

# Calculate and display the Calinski-Harabasz score
from sklearn.metrics import calinski_harabaz_score
ch_score = calinski_harabaz_score(data, data['Cluster'])
print(f'Calinski-Harabaz Score: {ch_score}')
