# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# 1. CREATE SYNTHETIC DATASET
# ============================================

def create_agricultural_dataset(n_regions=200):
    """
    Create a synthetic dataset of agricultural regions with 20 features
    """
    
    # Generate region names
    regions = [f"Region_{i+1}" for i in range(n_regions)]
    
    # Generate features with realistic correlations
    # Average rainfall (mm) - normal distribution with some regions having extreme values
    avg_rainfall = np.random.normal(800, 200, n_regions)
    avg_rainfall = np.clip(avg_rainfall, 200, 1800)
    
    # Average temperature (°C)
    avg_temp = np.random.normal(22, 5, n_regions)
    avg_temp = np.clip(avg_temp, 5, 35)
    
    # Soil moisture level (%) - correlated with rainfall
    soil_moisture = 30 + (avg_rainfall - 800) / 40 + np.random.normal(0, 5, n_regions)
    soil_moisture = np.clip(soil_moisture, 15, 80)
    
    # Soil pH
    soil_ph = np.random.normal(6.5, 0.8, n_regions)
    soil_ph = np.clip(soil_ph, 4.5, 8.5)
    
    # Nitrogen content (mg/kg)
    nitrogen = np.random.normal(150, 50, n_regions)
    nitrogen = np.clip(nitrogen, 30, 300)
    
    # Phosphorus content (mg/kg)
    phosphorus = np.random.normal(50, 20, n_regions)
    phosphorus = np.clip(phosphorus, 10, 150)
    
    # Potassium content (mg/kg)
    potassium = np.random.normal(200, 60, n_regions)
    potassium = np.clip(potassium, 50, 400)
    
    # Irrigation coverage (%)
    irrigation = np.random.uniform(20, 90, n_regions)
    
    # Crop area (hectares)
    crop_area = np.random.normal(5000, 2000, n_regions)
    crop_area = np.clip(crop_area, 1000, 15000)
    
    # Fertilizer usage (kg/hectare)
    fertilizer = np.random.normal(150, 50, n_regions)
    fertilizer = np.clip(fertilizer, 30, 350)
    
    # Pesticide usage (kg/hectare)
    pesticide = np.random.normal(10, 4, n_regions)
    pesticide = np.clip(pesticide, 1, 25)
    
    # Sunlight hours
    sunlight_hours = np.random.normal(8, 2, n_regions)
    sunlight_hours = np.clip(sunlight_hours, 4, 12)
    
    # Humidity (%)
    humidity = 40 + (avg_rainfall - 800) / 30 + np.random.normal(0, 8, n_regions)
    humidity = np.clip(humidity, 30, 90)
    
    # Altitude (meters)
    altitude = np.random.normal(500, 300, n_regions)
    altitude = np.clip(altitude, 0, 2500)
    
    # Wind speed (km/h)
    wind_speed = np.random.normal(15, 5, n_regions)
    wind_speed = np.clip(wind_speed, 5, 35)
    
    # Groundwater level (meters)
    groundwater = np.random.normal(15, 8, n_regions)
    groundwater = np.clip(groundwater, 1, 50)
    
    # Crop diversity index (0-1 scale)
    crop_diversity = np.random.beta(2, 2, n_regions)
    
    # Farming mechanization score (0-100)
    mechanization = np.random.uniform(30, 95, n_regions)
    
    # Historical yield variance
    yield_variance = np.random.exponential(2, n_regions)
    yield_variance = np.clip(yield_variance, 0.1, 10)
    
    # Average crop yield (tons/hectare) - target variable influenced by multiple factors
    avg_yield = (2 + 
                (avg_rainfall - 800) / 200 + 
                (avg_temp - 22) / 10 +
                soil_moisture / 40 +
                (soil_ph - 6.5) * 0.5 +
                nitrogen / 200 +
                phosphorus / 100 +
                potassium / 300 +
                irrigation / 50 +
                fertilizer / 200 +
                sunlight_hours / 5 +
                (100 - pesticide) / 100 +
                np.random.normal(0, 1, n_regions))
    
    avg_yield = np.clip(avg_yield, 1.5, 12)
    
    # Create DataFrame
    dataset = pd.DataFrame({
        'Region': regions,
        'Avg_Rainfall_mm': np.round(avg_rainfall, 2),
        'Avg_Temperature_C': np.round(avg_temp, 2),
        'Soil_Moisture_%': np.round(soil_moisture, 2),
        'Soil_pH': np.round(soil_ph, 2),
        'Nitrogen_Content': np.round(nitrogen, 2),
        'Phosphorus_Content': np.round(phosphorus, 2),
        'Potassium_Content': np.round(potassium, 2),
        'Irrigation_Coverage_%': np.round(irrigation, 2),
        'Crop_Area_ha': np.round(crop_area, 2),
        'Fertilizer_Usage': np.round(fertilizer, 2),
        'Pesticide_Usage': np.round(pesticide, 2),
        'Sunlight_Hours': np.round(sunlight_hours, 2),
        'Humidity_%': np.round(humidity, 2),
        'Altitude_m': np.round(altitude, 2),
        'Wind_Speed_kmh': np.round(wind_speed, 2),
        'Groundwater_Level_m': np.round(groundwater, 2),
        'Crop_Diversity_Index': np.round(crop_diversity, 3),
        'Farming_Mechanization': np.round(mechanization, 2),
        'Historical_Yield_Variance': np.round(yield_variance, 3),
        'Avg_Crop_Yield_tons_ha': np.round(avg_yield, 2)
    })
    
    return dataset

# Create the dataset
print("Creating agricultural dataset...")
df = pd.read_csv('agricultural_regions.csv') if False else create_agricultural_dataset(200)
df.to_csv('agricultural_regions.csv', index=False)
print(f"Dataset created with {df.shape[0]} regions and {df.shape[1]} features")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

def perform_eda(df):
    """
    Perform comprehensive EDA on the dataset
    """
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Basic info
    print("\n1. Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    
    # Statistical summary
    print("\n2. Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\n3. Missing Values:")
    print(df.isnull().sum())
    
    # Data types
    print("\n4. Data Types:")
    print(df.dtypes)
    
    # Visualizations
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()
    
    feature_cols = [col for col in df.columns if col != 'Region']
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx].set_title(f'Distribution of {col}', fontsize=10)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("\n5. Correlation Analysis:")
    numeric_cols = [col for col in feature_cols if col != 'Region']
    correlation_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('eda_correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Top correlations with crop yield
    yield_corr = correlation_matrix['Avg_Crop_Yield_tons_ha'].sort_values(ascending=False)
    print("\nTop 10 features correlated with Average Crop Yield:")
    print(yield_corr.head(11))
    
    return correlation_matrix

# Perform EDA
correlation_matrix = perform_eda(df)

# ============================================
# 3. DATA PREPROCESSING
# ============================================

def preprocess_data(df):
    """
    Handle missing values and normalize the data
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Separate features and target
    X = df.drop(['Region', 'Avg_Crop_Yield_tons_ha'], axis=1)
    y = df['Avg_Crop_Yield_tons_ha']
    
    # Check for missing values
    print("\nMissing values before handling:")
    print(X.isnull().sum())
    
    # Handle missing values (if any - in synthetic data there are none)
    if X.isnull().any().any():
        print("\nHandling missing values...")
        # Fill numeric missing values with median
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
    
    # Normalization using StandardScaler
    print("\nApplying StandardScaler normalization...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Alternative: MinMaxScaler for comparison
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    X_minmax = pd.DataFrame(X_minmax, columns=X.columns)
    
    print("\nNormalization complete!")
    print(f"Original feature ranges:")
    for col in X.columns[:5]:
        print(f"{col}: [{X[col].min():.2f}, {X[col].max():.2f}]")
    print(f"\nNormalized feature ranges (StandardScaler):")
    for col in X_scaled.columns[:5]:
        print(f"{col}: [{X_scaled[col].min():.2f}, {X_scaled[col].max():.2f}]")
    
    return X_scaled, y, scaler, X_minmax

# Preprocess the data
X_scaled, y, scaler, X_minmax = preprocess_data(df)

# ============================================
# 4. FEATURE ELIMINATION
# ============================================

def feature_elimination(X, y, method='forward', n_features=10):
    """
    Perform forward or backward feature elimination
    """
    print("\n" + "="*80)
    print(f"FEATURE ELIMINATION - {method.upper()} SELECTION")
    print("="*80)
    
    if method == 'forward':
        print("\nPerforming Forward Feature Selection...")
        # Simple forward selection using mutual information
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        selected_features = feature_scores.head(n_features)['Feature'].tolist()
        
    elif method == 'backward':
        print("\nPerforming Backward Feature Elimination using correlation...")
        # Use correlation with target for backward elimination
        correlations = pd.DataFrame({
            'Feature': X.columns,
            'Correlation': [abs(X[col].corr(y)) for col in X.columns]
        }).sort_values('Correlation', ascending=False)
        
        selected_features = correlations.head(n_features)['Feature'].tolist()
        
    else:
        print("\nUsing feature importance from variance...")
        # Fallback: select features with highest variance
        variances = X.var()
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Variance': variances
        }).sort_values('Variance', ascending=False)
        selected_features = feature_scores.head(n_features)['Feature'].tolist()
    
    print(f"\nSelected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Show feature importance scores
    if method == 'forward':
        print("\nFeature Importance Scores (Mutual Information):")
        print(feature_scores.head(15))
    
    return selected_features

# Perform forward feature selection
selected_features = feature_elimination(X_scaled, y, method='forward', n_features=10)

# Reduce dataset to selected features
X_selected = X_scaled[selected_features]
print(f"\nReduced dataset shape: {X_selected.shape}")

# ============================================
# 5. VISUALIZE FEATURE SPACE BEFORE CLUSTERING
# ============================================

def visualize_feature_space(X_selected):
    """
    Visualize the feature space using PCA
    """
    print("\n" + "="*80)
    print("VISUALIZING FEATURE SPACE (PCA)")
    print("="*80)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    
    print(f"\nPCA Explained Variance Ratio:")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Plot the PCA visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Average Crop Yield (tons/ha)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Feature Space Visualization (PCA) - Colored by Crop Yield')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_space_pca.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # 3D PCA visualization
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_selected)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                        c=y, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Average Crop Yield (tons/ha)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Feature Space Visualization')
    plt.savefig('feature_space_pca_3d.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return X_pca

# Visualize feature space
X_pca = visualize_feature_space(X_selected)

# ============================================
# 6. DETERMINE OPTIMAL K USING ELBOW METHOD
# ============================================

def find_optimal_k(X_selected, max_k=15):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score
    """
    print("\n" + "="*80)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS (K)")
    print("="*80)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_selected)
        inertias.append(kmeans.inertia_)
        
        if k >= 2:
            score = silhouette_score(X_selected, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Plot Elbow Method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal K', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Highlight potential optimal K
    ax1.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Suggested K=4')
    ax1.legend()
    
    # Silhouette scores
    ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score for Different K', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Highlight best silhouette score
    best_k = K_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, 
                label=f'Best K={best_k} (Score={max(silhouette_scores):.3f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimal_k_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("\nInertia values for each K:")
    for k, inertia in zip(K_range, inertias):
        print(f"K={k}: Inertia={inertia:.2f}")
    
    print("\nSilhouette Scores for each K:")
    for k, score in zip(K_range, silhouette_scores):
        print(f"K={k}: Silhouette Score={score:.3f}")
    
    # Determine optimal K (using elbow + silhouette)
    optimal_k = best_k
    print(f"\nRecommended optimal K: {optimal_k}")
    print(f"Best Silhouette Score at K={optimal_k}: {max(silhouette_scores):.3f}")
    
    return optimal_k, inertias, silhouette_scores

# Find optimal K
optimal_k, inertias, silhouette_scores = find_optimal_k(X_selected, max_k=12)

# ============================================
# 7. APPLY K-MEANS CLUSTERING
# ============================================

def apply_kmeans_clustering(X_selected, n_clusters):
    """
    Apply K-Means clustering with the optimal number of clusters
    """
    print("\n" + "="*80)
    print(f"APPLYING K-MEANS CLUSTERING (K={n_clusters})")
    print("="*80)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_selected)
    
    print(f"\nClustering completed!")
    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} regions ({count/len(cluster_labels)*100:.1f}%)")
    
    # Add cluster labels to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    return kmeans, cluster_labels, df_with_clusters

# Apply K-Means
kmeans_model, cluster_labels, df_clustered = apply_kmeans_clustering(X_selected, optimal_k)

# ============================================
# 8. POST-CLUSTERING VISUALIZATION
# ============================================

def visualize_clusters(X_selected, cluster_labels, df_clustered):
    """
    Create comprehensive visualizations of clustering results
    """
    print("\n" + "="*80)
    print("POST-CLUSTERING VISUALIZATION")
    print("="*80)
    
    # 2D PCA visualization of clusters
    pca = PCA(n_components=2)
    X_pca_clusters = pca.fit_transform(X_selected)
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: PCA with clusters
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_pca_clusters[:, 0], X_pca_clusters[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, 
                         s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('K-Means Clustering Results (2D PCA)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cluster centers visualization
    plt.subplot(2, 2, 2)
    cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, 
                                   columns=selected_features)
    
    # Normalize for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler_viz = MinMaxScaler()
    centers_normalized = scaler_viz.fit_transform(cluster_centers.T)
    
    im = plt.imshow(centers_normalized, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, label='Normalized Value')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.title('Cluster Centers Heatmap', fontsize=14)
    plt.xticks(range(len(cluster_centers)), [f'C{i}' for i in range(len(cluster_centers))])
    plt.yticks(range(len(selected_features)), selected_features, rotation=0, fontsize=8)
    
    # Plot 3: Feature distributions by cluster
    plt.subplot(2, 2, 3)
    # Select top 3 most important features
    feature_importance = X_selected.var().sort_values(ascending=False).head(3)
    top_features = feature_importance.index.tolist()
    
    for i, feature in enumerate(top_features):
        for cluster in np.unique(cluster_labels):
            cluster_data = X_selected[cluster_labels == cluster][feature]
            plt.hist(cluster_data, alpha=0.5, bins=20, 
                    label=f'Cluster {cluster}', density=True)
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.title(f'Distribution of {feature} by Cluster', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        break  # Show only one for clarity
    
    # Plot 4: Cluster sizes
    plt.subplot(2, 2, 4)
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    bars = plt.bar(cluster_sizes.index, cluster_sizes.values, 
                   color=plt.cm.tab10(range(len(cluster_sizes))))
    plt.xlabel('Cluster')
    plt.ylabel('Number of Regions')
    plt.title('Cluster Size Distribution', fontsize=14)
    plt.xticks(cluster_sizes.index)
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(size), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('clustering_visualizations.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Cluster characteristics summary
    print("\nCluster Characteristics Summary:")
    cluster_summary = df_clustered.groupby('Cluster')[selected_features + ['Avg_Crop_Yield_tons_ha']].mean()
    print(cluster_summary.round(3))
    
    return cluster_summary

# Visualize clusters
cluster_summary = visualize_clusters(X_selected, cluster_labels, df_clustered)

# ============================================
# 9. EVALUATION USING SILHOUETTE SCORE
# ============================================

def evaluate_clustering(X_selected, cluster_labels):
    """
    Evaluate clustering performance using Silhouette Score
    """
    print("\n" + "="*80)
    print("CLUSTERING EVALUATION")
    print("="*80)
    
    # Calculate overall silhouette score
    silhouette_avg = silhouette_score(X_selected, cluster_labels)
    print(f"\nOverall Silhouette Score: {silhouette_avg:.4f}")
    
    # Calculate silhouette score for each sample
    sample_silhouette_values = silhouette_samples(X_selected, cluster_labels)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Silhouette plot
    y_lower = 10
    n_clusters = len(np.unique(cluster_labels))
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 5
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average Silhouette Score: {silhouette_avg:.3f}')
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Label")
    ax1.set_title("Silhouette Plot for Clusters", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of silhouette scores by cluster
    cluster_silhouette_scores = []
    for i in range(n_clusters):
        cluster_score = sample_silhouette_values[cluster_labels == i].mean()
        cluster_silhouette_scores.append(cluster_score)
    
    bars = ax2.bar(range(n_clusters), cluster_silhouette_scores, 
                   color=plt.cm.tab10(range(n_clusters)))
    ax2.axhline(y=silhouette_avg, color='red', linestyle='--', 
                label=f'Overall: {silhouette_avg:.3f}')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette Score by Cluster', fontsize=14)
    ax2.set_xticks(range(n_clusters))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, cluster_silhouette_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('silhouette_evaluation.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Interpretation
    print("\nInterpretation:")
    if silhouette_avg > 0.7:
        print("Excellent clustering structure (Silhouette Score > 0.7)")
    elif silhouette_avg > 0.5:
        print("Good clustering structure (Silhouette Score > 0.5)")
    elif silhouette_avg > 0.3:
        print("Reasonable clustering structure (Silhouette Score > 0.3)")
    else:
        print("Weak clustering structure (Silhouette Score < 0.3)")
    
    return silhouette_avg

# Evaluate clustering
silhouette_avg = evaluate_clustering(X_selected, cluster_labels)

# ============================================
# 10. PREDICT FOR NEW/UNSEEN DATA
# ============================================

def predict_new_regions(kmeans_model, scaler, selected_features):
    """
    Demonstrate prediction for new/unseen agricultural regions
    """
    print("\n" + "="*80)
    print("PREDICTING CLUSTERS FOR NEW REGIONS")
    print("="*80)
    
    # Create sample new region data
    print("\nCreating 5 new/unseen agricultural regions...")
    
    new_regions = pd.DataFrame({
        'Region': ['New_Region_A', 'New_Region_B', 'New_Region_C', 'New_Region_D', 'New_Region_E'],
        'Avg_Rainfall_mm': [1200, 450, 900, 1500, 600],
        'Avg_Temperature_C': [25, 18, 22, 28, 20],
        'Soil_Moisture_%': [65, 35, 55, 75, 45],
        'Soil_pH': [7.2, 6.0, 6.8, 7.5, 6.3],
        'Nitrogen_Content': [180, 80, 150, 220, 120],
        'Phosphorus_Content': [70, 30, 55, 90, 45],
        'Potassium_Content': [280, 120, 210, 320, 180],
        'Irrigation_Coverage_%': [75, 40, 65, 85, 55],
        'Crop_Area_ha': [8000, 2500, 5500, 12000, 4000],
        'Fertilizer_Usage': [200, 80, 150, 250, 120],
        'Pesticide_Usage': [12, 6, 10, 15, 8],
        'Sunlight_Hours': [9, 6, 8, 10, 7],
        'Humidity_%': [75, 45, 65, 80, 55],
        'Altitude_m': [200, 800, 500, 100, 600],
        'Wind_Speed_kmh': [12, 20, 15, 10, 18],
        'Groundwater_Level_m': [8, 25, 15, 5, 20],
        'Crop_Diversity_Index': [0.7, 0.4, 0.6, 0.8, 0.5],
        'Farming_Mechanization': [85, 45, 70, 90, 60],
        'Historical_Yield_Variance': [2.5, 5.0, 3.5, 2.0, 4.0],
        'Avg_Crop_Yield_tons_ha': [8.5, 3.2, 6.8, 9.5, 5.0]
    })
    
    print("\nNew Regions Data:")
    print(new_regions[['Region', 'Avg_Rainfall_mm', 'Avg_Temperature_C', 
                       'Avg_Crop_Yield_tons_ha']])
    
    # Preprocess new data
    X_new = new_regions.drop(['Region', 'Avg_Crop_Yield_tons_ha'], axis=1)
    
    # Apply same preprocessing (using the fitted scaler)
    X_new_scaled = scaler.transform(X_new)
    X_new_selected = pd.DataFrame(X_new_scaled, columns=X_new.columns)[selected_features]
    
    # Predict clusters
    predicted_clusters = kmeans_model.predict(X_new_selected)
    
    # Add predictions to new regions data
    new_regions['Predicted_Cluster'] = predicted_clusters
    
    print("\nPrediction Results:")
    print(new_regions[['Region', 'Avg_Crop_Yield_tons_ha', 'Predicted_Cluster']])
    
    # Visualize predictions in context
    plt.figure(figsize=(12, 6))
    
    # Combine original and new data for visualization
    pca_full = PCA(n_components=2)
    X_combined = np.vstack([X_selected, X_new_selected])
    X_combined_pca = pca_full.fit_transform(X_combined)
    
    # Split back
    X_original_pca = X_combined_pca[:len(X_selected)]
    X_new_pca = X_combined_pca[len(X_selected):]
    
    # Plot original clusters
    scatter1 = plt.scatter(X_original_pca[:, 0], X_original_pca[:, 1], 
                          c=cluster_labels, cmap='tab10', alpha=0.5, 
                          s=80, label='Original Regions')
    
    # Plot new regions with predictions
    scatter2 = plt.scatter(X_new_pca[:, 0], X_new_pca[:, 1], 
                          c=predicted_clusters, cmap='tab10', 
                          s=200, marker='*', edgecolors='black', 
                          linewidth=2, label='New Regions')
    
    # Add labels for new regions
    for i, region in enumerate(new_regions['Region']):
        plt.annotate(region, (X_new_pca[i, 0], X_new_pca[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Prediction for New Agricultural Regions', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter1, label='Cluster')
    plt.tight_layout()
    plt.savefig('new_region_predictions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*40)
    print("DETAILED PREDICTION ANALYSIS")
    print("="*40)
    for idx, row in new_regions.iterrows():
        print(f"\n{row['Region']}:")
        print(f"  - Expected Yield: {row['Avg_Crop_Yield_tons_ha']} tons/ha")
        print(f"  - Assigned to Cluster: {row['Predicted_Cluster']}")
        
        # Find similar original regions in same cluster
        same_cluster_regions = df_clustered[df_clustered['Cluster'] == row['Predicted_Cluster']]['Region'].head(3).tolist()
        print(f"  - Similar to: {', '.join(same_cluster_regions)}")
    
    return new_regions, predicted_clusters

# Predict for new regions
new_predictions, predicted_clusters = predict_new_regions(kmeans_model, scaler, selected_features)

# ============================================
# 11. FINAL SUMMARY AND SAVE MODEL
# ============================================

def save_model_and_summarize(kmeans_model, scaler, selected_features, silhouette_avg):
    """
    Save the trained model and provide final summary
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY AND MODEL SAVING")
    print("="*80)
    
    # Save model components using pickle
    import pickle
    
    model_components = {
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'silhouette_score': silhouette_avg,
        'optimal_k': optimal_k
    }
    
    with open('agricultural_clustering_model.pkl', 'wb') as file:
        pickle.dump(model_components, file)
    
    print("\nModel saved as 'agricultural_clustering_model.pkl'")
    
    # Save cluster assignments
    df_clustered.to_csv('agricultural_regions_with_clusters.csv', index=False)
    print("Cluster assignments saved as 'agricultural_regions_with_clusters.csv'")
    
    # Final summary
    print("\n" + "="*80)
    print("PROJECT COMPLETION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Dataset: {df.shape[0]} agricultural regions with 20 features")
    print(f"✅ Features after selection: {len(selected_features)}")
    print(f"✅ Optimal number of clusters (K): {optimal_k}")
    print(f"✅ Silhouette Score: {silhouette_avg:.4f}")
    print(f"✅ Model predictions made for 5 new regions")
    
    print("\nCluster Characteristics:")
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        avg_yield = cluster_data['Avg_Crop_Yield_tons_ha'].mean()
        print(f"  Cluster {cluster}: {len(cluster_data)} regions, Avg Yield: {avg_yield:.2f} tons/ha")
    
    print("\n📊 Generated Files:")
    print("  1. agricultural_regions.csv - Original dataset")
    print("  2. agricultural_regions_with_clusters.csv - Data with cluster labels")
    print("  3. agricultural_clustering_model.pkl - Trained model")
    print("  4. eda_distributions.png - Feature distribution plots")
    print("  5. eda_correlation_matrix.png - Correlation heatmap")
    print("  6. optimal_k_analysis.png - Elbow method and silhouette scores")
    print("  7. clustering_visualizations.png - Cluster visualizations")
    print("  8. silhouette_evaluation.png - Silhouette analysis")
    print("  9. new_region_predictions.png - Predictions visualization")
    
    print("\n🎯 Project Successfully Completed!")
    print("   The system can now predict clusters for any new agricultural region.")

# Save model and generate final summary
save_model_and_summarize(kmeans_model, scaler, selected_features, silhouette_avg)

# ============================================
# OPTIONAL: FUNCTION TO LOAD AND USE THE MODEL
# ============================================

def load_and_predict(new_data_path=None):
    """
    Load the saved model and predict for new data
    """
    import pickle
    
    # Load model
    with open('agricultural_clustering_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    print("Model loaded successfully!")
    print(f"Optimal K: {model['optimal_k']}")
    print(f"Silhouette Score: {model['silhouette_score']:.4f}")
    
    if new_data_path:
        # Load new data from CSV
        new_data = pd.read_csv(new_data_path)
        print(f"\nLoaded {len(new_data)} new regions")
        
        # Preprocess
        X_new = new_data.drop(['Region', 'Avg_Crop_Yield_tons_ha'], axis=1)
        X_new_scaled = model['scaler'].transform(X_new)
        X_new_selected = X_new_scaled[:, [list(X_new.columns).index(f) for f in model['selected_features']]]
        
        # Predict
        predictions = model['kmeans_model'].predict(X_new_selected)
        new_data['Predicted_Cluster'] = predictions
        
        return new_data
    else:
        return None

print("\n" + "="*80)
print("To load and use the model later, use:")
print("  model = load_and_predict()")
print("  # or")
print("  new_data_with_clusters = load_and_predict('your_new_data.csv')")
print("="*80)
