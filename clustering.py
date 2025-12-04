#!/usr/bin/env python3
"""
clustering.py
Step 1: Cluster restaurant foods based on nutritional composition.
Output: Dataset with cluster labels for use in classification.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# === CONFIGURATION ===
INPUT_FILE = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/output/restaurant_foods_nutrition.parquet"
OUTPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output"

# Features for clustering (nutritional composition)
NUTRIENT_FEATURES = [
    'calories_final', 'protein', 'fat', 'carbohydrate',
    'fiber', 'sugar', 'sodium', 'cholesterol',
    'saturated_fat', 'calcium', 'iron', 'potassium'
]

# Clustering parameters
N_CLUSTERS_DEFAULT = 8
K_RANGE = range(2, 15)  # For elbow method
RANDOM_STATE = 42

# === END CONFIGURATION ===


def load_data(filepath):
    """Load preprocessed restaurant food data."""
    print(f"Loading data from: {filepath}")
    
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df):,} foods")
    return df


def prepare_data(df, features):
    """Clean and prepare data for clustering."""
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Check available features
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    print(f"\nUsing {len(available_features)} features:")
    print(", ".join(available_features))
    
    # Keep required columns
    required_cols = available_features + ['fdc_id', 'description', 'cooking_method', 
                                          'data_type', 'category_name']
    required_cols = [col for col in required_cols if col in df.columns]
    
    df_clean = df[required_cols].copy()
    
    # Remove rows with missing nutrient values
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=available_features)
    after = len(df_clean)
    
    print(f"\nRemoved {before - after:,} rows with missing nutrient data")
    print(f"Dataset size: {after:,} foods")
    
    # Remove outliers (beyond 3 standard deviations)
    print("\nRemoving outliers...")
    for feature in available_features:
        mean = df_clean[feature].mean()
        std = df_clean[feature].std()
        df_clean = df_clean[
            (df_clean[feature] >= mean - 3*std) & 
            (df_clean[feature] <= mean + 3*std)
        ]
    
    print(f"After outlier removal: {len(df_clean):,} foods")
    
    return df_clean, available_features


def find_optimal_clusters(X_scaled, k_range, output_dir):
    """Find optimal number of clusters using multiple metrics."""
    print("\n" + "="*80)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*80)
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    print("\nTesting cluster numbers:", list(k_range))
    
    for k in k_range:
        print(f"  k={k}...", end=" ")
        
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
        
        print(f"Silhouette: {silhouette_scores[-1]:.3f}")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'k': list(k_range),
        'inertia': inertias,
        'silhouette_score': silhouette_scores,
        'calinski_harabasz_score': calinski_scores,
        'davies_bouldin_score': davies_bouldin_scores
    })
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'clustering_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Recommend optimal k
    optimal_silhouette = metrics_df.loc[metrics_df['silhouette_score'].idxmax(), 'k']
    optimal_calinski = metrics_df.loc[metrics_df['calinski_harabasz_score'].idxmax(), 'k']
    optimal_davies = metrics_df.loc[metrics_df['davies_bouldin_score'].idxmin(), 'k']
    
    print(f"\nRecommended k values:")
    print(f"  By Silhouette Score: k={optimal_silhouette}")
    print(f"  By Calinski-Harabasz: k={optimal_calinski}")
    print(f"  By Davies-Bouldin: k={optimal_davies}")
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Elbow plot
    axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette score
    axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Good threshold')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (higher is better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz score
    axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2)
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (higher is better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Davies-Bouldin score
    axes[1, 1].plot(k_range, davies_bouldin_scores, 'mo-', linewidth=2)
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('Davies-Bouldin Score (lower is better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_clusters_analysis.png'), dpi=300)
    plt.close()
    
    print(f"Saved visualization to: optimal_clusters_analysis.png")
    
    return int(optimal_silhouette)


def perform_clustering(df, X_scaled, features, n_clusters, output_dir):
    """Perform K-Means clustering."""
    print("\n" + "="*80)
    print(f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
    print("="*80)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate quality metrics
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    calinski = calinski_harabasz_score(X_scaled, clusters)
    
    print(f"\nClustering Quality Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f} (range: -1 to 1, higher is better)")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski:.2f} (higher is better)")
    
    # Add clusters to dataframe
    df['nutritional_cluster'] = clusters
    
    # Analyze cluster characteristics
    print(f"\n{'='*80}")
    print("CLUSTER CHARACTERISTICS")
    print("="*80)
    
    cluster_stats = df.groupby('nutritional_cluster')[features].mean().round(2)
    cluster_counts = df['nutritional_cluster'].value_counts().sort_index()
    
    cluster_summary = cluster_stats.copy()
    cluster_summary['count'] = cluster_counts
    cluster_summary['percentage'] = (cluster_counts / len(df) * 100).round(1)
    
    print("\nCluster sizes:")
    for i in range(n_clusters):
        count = cluster_counts[i]
        pct = cluster_counts[i] / len(df) * 100
        print(f"  Cluster {i}: {count:,} foods ({pct:.1f}%)")
    
    # Save cluster characteristics
    cluster_path = os.path.join(output_dir, 'cluster_characteristics.csv')
    cluster_summary.to_csv(cluster_path)
    print(f"\nSaved cluster characteristics to: {cluster_path}")
    
    # Name clusters based on nutritional profile
    cluster_names = name_clusters(cluster_stats, features)
    df['cluster_name'] = df['nutritional_cluster'].map(cluster_names)
    
    print("\nCluster names based on nutritional profile:")
    for cluster_id, name in cluster_names.items():
        print(f"  Cluster {cluster_id}: {name}")
    
    # Save cluster names
    names_df = pd.DataFrame({
        'cluster_id': list(cluster_names.keys()),
        'cluster_name': list(cluster_names.values())
    })
    names_path = os.path.join(output_dir, 'cluster_names.csv')
    names_df.to_csv(names_path, index=False)
    
    return df, kmeans, cluster_names


def name_clusters(cluster_stats, features):
    """Assign meaningful names to clusters based on nutritional characteristics."""
    cluster_names = {}
    
    # Calculate medians for comparison
    medians = cluster_stats.median()
    
    for cluster_id in cluster_stats.index:
        cluster_data = cluster_stats.loc[cluster_id]
        
        characteristics = []
        
        # Check calories
        if 'calories_final' in cluster_data.index:
            if cluster_data['calories_final'] > medians['calories_final'] * 1.4:
                characteristics.append("Very High Calorie")
            elif cluster_data['calories_final'] > medians['calories_final'] * 1.15:
                characteristics.append("High Calorie")
            elif cluster_data['calories_final'] < medians['calories_final'] * 0.6:
                characteristics.append("Very Low Calorie")
            elif cluster_data['calories_final'] < medians['calories_final'] * 0.85:
                characteristics.append("Low Calorie")
        
        # Check macros
        if 'protein' in cluster_data.index and cluster_data['protein'] > medians['protein'] * 1.3:
            characteristics.append("High Protein")
        
        if 'fat' in cluster_data.index and cluster_data['fat'] > medians['fat'] * 1.3:
            characteristics.append("High Fat")
        
        if 'carbohydrate' in cluster_data.index and cluster_data['carbohydrate'] > medians['carbohydrate'] * 1.3:
            characteristics.append("High Carb")
        
        # Check micronutrients
        if 'fiber' in cluster_data.index and cluster_data['fiber'] > medians['fiber'] * 1.4:
            characteristics.append("High Fiber")
        
        if 'sugar' in cluster_data.index and cluster_data['sugar'] > medians['sugar'] * 1.4:
            characteristics.append("High Sugar")
        
        if 'sodium' in cluster_data.index and cluster_data['sodium'] > medians['sodium'] * 1.4:
            characteristics.append("High Sodium")
        
        # Assign name
        if characteristics:
            cluster_names[cluster_id] = " / ".join(characteristics)
        else:
            cluster_names[cluster_id] = "Balanced / Moderate"
    
    return cluster_names


def visualize_clusters(df, X_scaled, features, output_dir):
    """Create visualizations of clusters."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
    
    # Plot 1: Cluster visualization
    plt.figure(figsize=(14, 10))
    
    n_clusters = df['nutritional_cluster'].nunique()
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_data = df[df['nutritional_cluster'] == i]
        cluster_name = cluster_data['cluster_name'].iloc[0]
        
        plt.scatter(
            cluster_data['pca_1'],
            cluster_data['pca_2'],
            c=[colors[i]],
            label=f"C{i}: {cluster_name} (n={len(cluster_data):,})",
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Food Clusters Based on Nutritional Composition\n(PCA Visualization)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_pca_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: clusters_pca_visualization.png")
    
    # Plot 2: Cluster composition by cooking method
    if 'cooking_method' in df.columns:
        cluster_cooking = pd.crosstab(
            df['nutritional_cluster'], 
            df['cooking_method'], 
            normalize='index'
        ) * 100
        
        plt.figure(figsize=(14, 8))
        cluster_cooking.plot(kind='bar', stacked=True, ax=plt.gca(), 
                            colormap='tab20', width=0.8)
        plt.xlabel('Nutritional Cluster')
        plt.ylabel('Percentage (%)')
        plt.title('Cooking Methods Distribution within Each Nutritional Cluster')
        plt.legend(title='Cooking Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_cooking_method_distribution.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: cluster_cooking_method_distribution.png")
    
    # Plot 3: Radar chart of cluster characteristics
    plot_radar_chart(df, features, output_dir)
    
    return df


def plot_radar_chart(df, features, output_dir):
    """Create radar chart showing average nutritional profile of each cluster."""
    from math import pi
    
    # Get cluster means and normalize to 0-1 scale
    cluster_stats = df.groupby('nutritional_cluster')[features].mean()
    
    # Normalize each feature to 0-1 scale
    normalized_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
    
    n_clusters = len(normalized_stats)
    n_features = len(features)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]
    
    for idx, (cluster_id, row) in enumerate(normalized_stats.iterrows()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        values = row.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f[:8] for f in features], size=8)
        
        cluster_name = df[df['nutritional_cluster'] == cluster_id]['cluster_name'].iloc[0]
        ax.set_title(f'Cluster {cluster_id}: {cluster_name}', size=10, pad=20)
        ax.set_ylim(0, 1)
        ax.grid(True)
    
    # Hide extra subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_radar_charts.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_radar_charts.png")


def save_cluster_samples(df, output_dir):
    """Save sample foods from each cluster."""
    print("\n" + "="*80)
    print("SAVING CLUSTER SAMPLES")
    print("="*80)
    
    samples_per_cluster = 10
    sample_list = []
    
    for cluster_id in sorted(df['nutritional_cluster'].unique()):
        cluster_data = df[df['nutritional_cluster'] == cluster_id]
        
        # Get diverse samples (sorted by different criteria)
        samples = cluster_data.sample(n=min(samples_per_cluster, len(cluster_data)), 
                                     random_state=RANDOM_STATE)
        sample_list.append(samples)
    
    samples_df = pd.concat(sample_list, ignore_index=True)
    
    # Select columns to save
    sample_cols = ['fdc_id', 'description', 'nutritional_cluster', 'cluster_name',
                   'cooking_method', 'calories_final', 'protein', 'fat', 
                   'carbohydrate', 'fiber', 'sodium']
    sample_cols = [col for col in sample_cols if col in samples_df.columns]
    
    samples_path = os.path.join(output_dir, 'cluster_sample_foods.csv')
    samples_df[sample_cols].to_csv(samples_path, index=False)
    print(f"Saved {len(samples_df)} sample foods to: {samples_path}")


def save_outputs(df, scaler, kmeans, features, output_dir):
    """Save all outputs."""
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save full dataset with cluster labels (for classification step)
    output_cols = ['fdc_id', 'description', 'cooking_method', 'data_type',
                   'category_name', 'nutritional_cluster', 'cluster_name'] + features
    output_cols = [col for col in output_cols if col in df.columns]
    
    output_path = os.path.join(output_dir, 'foods_with_nutritional_clusters.csv')
    df[output_cols].to_csv(output_path, index=False)
    print(f"✓ Saved full dataset: foods_with_nutritional_clusters.csv")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(output_cols)}")
    
    # Also save as parquet for faster loading
    parquet_path = os.path.join(output_dir, 'foods_with_nutritional_clusters.parquet')
    df[output_cols].to_parquet(parquet_path, index=False)
    print(f"✓ Saved parquet: foods_with_nutritional_clusters.parquet")
    
    # Save scaler and model for future use
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: scaler.pkl")
    
    kmeans_path = os.path.join(output_dir, 'kmeans_model.pkl')
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"✓ Saved K-Means model: kmeans_model.pkl")
    
    # Save feature list
    features_path = os.path.join(output_dir, 'feature_list.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    print(f"✓ Saved feature list: feature_list.txt")


def main():
    print("="*80)
    print("STEP 1: NUTRITIONAL CLUSTERING")
    print("Group foods by nutritional composition")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    df = load_data(INPUT_FILE)
    df_clean, features = prepare_data(df, NUTRIENT_FEATURES)
    
    # Scale features
    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    print("Features scaled using StandardScaler")
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled, K_RANGE, OUTPUT_DIR)
    
    # User can override
    print(f"\nUsing k={N_CLUSTERS_DEFAULT} clusters (can be changed in configuration)")
    n_clusters = N_CLUSTERS_DEFAULT
    
    # Perform clustering
    df_clustered, kmeans, cluster_names = perform_clustering(
        df_clean, X_scaled, features, n_clusters, OUTPUT_DIR
    )
    
    # Visualize clusters
    df_clustered = visualize_clusters(df_clustered, X_scaled, features, OUTPUT_DIR)
    
    # Save sample foods
    save_cluster_samples(df_clustered, OUTPUT_DIR)
    
    # Save all outputs
    save_outputs(df_clustered, scaler, kmeans, features, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("CLUSTERING COMPLETE")
    print("="*80)
    print(f"\nTotal foods clustered: {len(df_clustered):,}")
    print(f"Number of clusters: {n_clusters}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. foods_with_nutritional_clusters.csv/parquet - Main output")
    print("  2. cluster_characteristics.csv - Cluster statistics")
    print("  3. cluster_names.csv - Cluster naming")
    print("  4. cluster_sample_foods.csv - Example foods")
    print("  5. clustering_metrics.csv - Quality metrics")
    print("  6. optimal_clusters_analysis.png - Elbow curve")
    print("  7. clusters_pca_visualization.png - 2D cluster plot")
    print("  8. cluster_cooking_method_distribution.png")
    print("  9. cluster_radar_charts.png - Nutritional profiles")
    print("  10. scaler.pkl, kmeans_model.pkl - Saved models")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nNext step: Run classification.py to predict clusters from nutrition")


if __name__ == "__main__":
    main()