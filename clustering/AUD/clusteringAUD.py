import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_10x_data(matrix_path, barcodes_path, features_path):
    matrix = scipy.io.mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, names=['barcode'])
    
    features = pd.read_csv(features_path, sep='\t', header=None, 
                          names=['gene_id', 'gene_name', 'feature_type'])
    
    return matrix, barcodes, features

matrix, barcodes, features = load_10x_data('matrix.mtx', 'barcodes.tsv', 'features.tsv')

print(f"Matrix shape: {matrix.shape} (cells x genes)")
print(f"Number of barcodes: {len(barcodes)}")
print(f"Number of features: {len(features)}")

min_counts = 1000
cell_counts = np.array(matrix.sum(axis=1)).flatten()
filtered_cells = cell_counts >= min_counts
matrix_filtered = matrix[filtered_cells, :]
barcodes_filtered = barcodes[filtered_cells]

print(f"Removed {sum(~filtered_cells)} cells with less than {min_counts} counts")

min_cells = 10
gene_counts = np.array((matrix_filtered > 0).sum(axis=0)).flatten()
filtered_genes = gene_counts >= min_cells
matrix_filtered = matrix_filtered[:, filtered_genes]
features_filtered = features[filtered_genes]

print(f"Removed {sum(~filtered_genes)} genes expressed in less than {min_cells} cells")

counts_per_cell = np.array(matrix_filtered.sum(axis=1)).flatten()
matrix_norm = matrix_filtered / counts_per_cell[:, np.newaxis] * 1e6

matrix_log = np.log1p(matrix_norm)

scaler = StandardScaler(with_mean=False)
matrix_scaled = scaler.fit_transform(matrix_log)

pca = PCA(n_components=50)
matrix_pca = pca.fit_transform(matrix_scaled.toarray())

def kmeans_analysis(data, max_clusters=10):
    silhouette_scores = []
    cluster_range = range(2, max_clusters+1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(data)
    
    return final_labels, optimal_clusters

cluster_labels, n_clusters = kmeans_analysis(matrix_pca)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    x=matrix_pca[:, 0], 
    y=matrix_pca[:, 1], 
    c=cluster_labels, 
    cmap='viridis',
    alpha=0.6
)
plt.colorbar(scatter, label='Cluster')
plt.title('Cell Clusters in PCA Space', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.grid(True)
plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    matrix_pca, cluster_labels, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred))

if svm.kernel == 'linear':
    feature_importance = np.abs(svm.coef_[0])
    important_features_idx = np.argsort(feature_importance)[::-1][:20]
    important_features = features_filtered.iloc[important_features_idx]
    print("\nTop 20 important features:")
    print(important_features)

cluster_probs = svm.predict_proba(matrix_pca)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(matrix_pca)
distances = kmeans.transform(matrix_pca)

derived_features = pd.DataFrame({
    'barcode': barcodes_filtered['barcode'].values,
    'cluster': cluster_labels
})

for i in range(n_clusters):
    derived_features[f'cluster_{i}_prob'] = cluster_probs[:, i]

for i in range(n_clusters):
    derived_features[f'distance_to_center_{i}'] = distances[:, i]

top_genes = 5
highly_variable_genes = np.argsort(np.var(matrix_scaled.toarray(), axis=0))[::-1][:top_genes]

for i, gene_idx in enumerate(highly_variable_genes):
    gene_name = features_filtered.iloc[gene_idx]['gene_name']
    derived_features[f'top_gene_{i}_{gene_name}'] = matrix_scaled[:, gene_idx].toarray().flatten()

derived_features.to_csv('derived_features.csv', index=False)
print("\nDerived features saved to 'derived_features.csv'")

for i, gene_idx in enumerate(highly_variable_genes[:3]):
    gene_name = features_filtered.iloc[gene_idx]['gene_name']
    plt.figure(figsize=(10, 6))
    
    data = []
    for cluster in np.unique(cluster_labels):
        data.append(matrix_scaled[cluster_labels == cluster, gene_idx].toarray().flatten())
    
    plt.boxplot(data, labels=np.unique(cluster_labels))
    plt.title(f'Expression of {gene_name} across clusters', fontsize=14)
    plt.ylabel('Scaled expression', fontsize=12)
    plt.xlabel('Cluster', fontsize=12)
    plt.grid(True)
    plt.savefig(f'gene_expression_{gene_name}.png', dpi=300, bbox_inches='tight')
    plt.show()