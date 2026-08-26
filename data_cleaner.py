import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def select_features(df, features=None):
    """
    Selects the relevant features from the DataFrame.
    Args:
        df (pd.DataFrame): The input dataset.
        features (list): List of column names to select. Defaults to ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    Returns:
        pd.DataFrame: Selected features dataframe.
    """
    if features is None:
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    return df[features]

def scale_features(X):
    """
    Standardizes the features using StandardScaler.
    Args:
        X (pd.DataFrame or np.array): Input features.
    Returns:
        np.array: Standardized features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def compute_kmeans_metrics(X_scaled, k_range=range(2,11), random_state=42):
    """
    Computes inertia and silhouette scores for multiple K values.
    Args:
        X_scaled (np.array): Standardized features.
        k_range (iterable): Range of K values to test.
    Returns:
        tuple: (inertias list, silhouettes list)
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return inertias, silhouettes

def reduce_dimensions_pca(X_scaled, n_components=2):
    """
    Reduces dimensions using PCA.
    Args:
        X_scaled (np.array): Standardized features.
        n_components (int): Number of PCA components.
    Returns:
        np.array: Transformed features.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def perform_kmeans(X_scaled, n_clusters, random_state=42):
    """
    Fits KMeans and returns the cluster labels.
    Args:
        X_scaled (np.array): Standardized features.
        n_clusters (int): Number of clusters.
    Returns:
        np.array: Cluster labels
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X_scaled)
    return labels

def best_k_by_silhouette(silhouettes, k_range=range(2,11)):
    """
    Returns the K with the highest silhouette score.
    """
    index = silhouettes.index(max(silhouettes))
    return list(k_range)[index]

def main():
    # Load dataset
    df = pd.read_csv("Mall_Customers.csv")
    print("Dataset shape:", df.shape)
    
    # Feature selection and scaling
    X = select_features(df)
    X_scaled = scale_features(X)
    
    # Compute KMeans metrics
    k_range = range(2, 11)
    inertias, silhouettes = compute_kmeans_metrics(X_scaled, k_range)
    
    # Determine best k
    best_k = best_k_by_silhouette(silhouettes, k_range)
    print("Best k by silhouette score:", best_k)
    
    # Perform KMeans with best k
    labels = perform_kmeans(X_scaled, best_k)
    
    # PCA
    X_pca = reduce_dimensions_pca(X_scaled)

# Run the script
if __name__ == "__main__":
    main()
