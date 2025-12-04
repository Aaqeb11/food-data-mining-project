#!/usr/bin/env python3
"""
validation.py
Step 3: Comprehensive validation and analysis of classification model.
Includes error analysis, prediction confidence, and real-world testing.
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

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import learning_curve, validation_curve

# === CONFIGURATION ===
DATA_FILE = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output/foods_with_nutritional_clusters.parquet"
MODEL_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output/classification"
OUTPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output/validation"

RANDOM_STATE = 42

# === END CONFIGURATION ===


def load_model_and_data():
    """Load trained model, scaler, and data."""
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'best_classifier.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded model: {type(model).__name__}")
    
    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, 'classifier_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler")
    
    # Load feature list
    features_path = os.path.join(MODEL_DIR, 'feature_list.txt')
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(features)} features")
    
    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    if DATA_FILE.endswith('.parquet'):
        df = pd.read_parquet(DATA_FILE)
    else:
        df = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(df):,} foods")
    
    return model, scaler, features, df


def prepare_validation_data(df, features):
    """Prepare validation dataset."""
    print("\n" + "="*80)
    print("PREPARING VALIDATION DATA")
    print("="*80)
    
    # Select features and remove missing values
    df_clean = df[features + ['nutritional_cluster', 'cluster_name', 
                              'description', 'fdc_id']].dropna(subset=features)
    
    X = df_clean[features].values
    y = df_clean['nutritional_cluster'].values
    
    print(f"Validation dataset: {len(df_clean):,} samples")
    print(f"Number of clusters: {len(np.unique(y))}")
    
    return X, y, df_clean


def comprehensive_metrics(model, scaler, X, y, output_dir):
    """Calculate comprehensive performance metrics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE METRICS")
    print("="*80)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Additional metrics
    kappa = cohen_kappa_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    
    print(f"\nAdditional Metrics:")
    print(f"  Cohen's Kappa: {kappa:.4f} (agreement beyond chance)")
    print(f"  Matthews Correlation: {mcc:.4f} (-1 to 1, higher is better)")
    
    # Per-class metrics
    print("\nPer-Class Performance:")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    # Display per-class
    for cluster_id in sorted(np.unique(y)):
        if str(cluster_id) in report:
            metrics = report[str(cluster_id)]
            print(f"  Cluster {cluster_id}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1-score']:.3f}")
            print(f"    Support: {int(metrics['support'])}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'validation_detailed_report.csv')
    report_df.to_csv(report_path)
    print(f"\n✓ Saved detailed report: validation_detailed_report.csv")
    
    # Save summary metrics
    summary = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', "Cohen's Kappa", 
                  'Matthews Correlation'],
        'Value': [accuracy, precision, recall, f1, kappa, mcc]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'validation_metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    return y_pred, y_pred_proba


def confusion_matrix_analysis(y, y_pred, output_dir):
    """Detailed confusion matrix analysis."""
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    cm = confusion_matrix(y, y_pred)
    n_clusters = len(np.unique(y))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Normalize by true labels (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=range(n_clusters),
                yticklabels=range(n_clusters),
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Confusion Matrix\n(Values show counts, colors show row-normalized proportions)')
    plt.ylabel('True Cluster')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_detailed.png'), dpi=300)
    plt.close()
    print("✓ Saved: confusion_matrix_detailed.png")
    
    # Analyze misclassifications
    print("\nMisclassification Analysis:")
    total_errors = np.sum(cm) - np.trace(cm)
    print(f"  Total misclassifications: {total_errors:,} ({total_errors/np.sum(cm)*100:.2f}%)")
    
    # Find most confused pairs
    confused_pairs = []
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'True_Cluster': i,
                    'Predicted_Cluster': j,
                    'Count': cm[i, j],
                    'Percentage': cm[i, j] / cm[i].sum() * 100
                })
    
    confused_df = pd.DataFrame(confused_pairs).sort_values('Count', ascending=False)
    
    print("\nTop 10 Most Common Misclassifications:")
    for idx, row in confused_df.head(10).iterrows():
        print(f"  True={int(row['True_Cluster'])} → Predicted={int(row['Predicted_Cluster'])}: "
              f"{int(row['Count'])} cases ({row['Percentage']:.1f}%)")
    
    # Save misclassification analysis
    confused_path = os.path.join(output_dir, 'misclassification_analysis.csv')
    confused_df.to_csv(confused_path, index=False)
    print(f"\n✓ Saved: misclassification_analysis.csv")


def prediction_confidence_analysis(y, y_pred, y_pred_proba, df_clean, output_dir):
    """Analyze prediction confidence."""
    print("\n" + "="*80)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*80)
    
    if y_pred_proba is None:
        print("Model does not support probability predictions. Skipping.")
        return
    
    # Get confidence scores (max probability)
    confidence_scores = np.max(y_pred_proba, axis=1)
    
    # Add predictions and confidence to dataframe
    df_clean['predicted_cluster'] = y_pred
    df_clean['confidence'] = confidence_scores
    df_clean['correct_prediction'] = (y == y_pred)
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {confidence_scores.mean():.4f}")
    print(f"  Median confidence: {np.median(confidence_scores):.4f}")
    print(f"  Min confidence: {confidence_scores.min():.4f}")
    print(f"  Max confidence: {confidence_scores.max():.4f}")
    
    # Confidence by correctness
    correct_conf = confidence_scores[y == y_pred]
    incorrect_conf = confidence_scores[y != y_pred]
    
    print(f"\nConfidence by Prediction Correctness:")
    print(f"  Correct predictions: {correct_conf.mean():.4f} (mean confidence)")
    print(f"  Incorrect predictions: {incorrect_conf.mean():.4f} (mean confidence)")
    
    # Plot confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall confidence distribution
    axes[0].hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(confidence_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidence_scores.mean():.3f}')
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Confidence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence by correctness
    axes[1].hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
    axes[1].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution by Correctness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300)
    plt.close()
    print("✓ Saved: confidence_analysis.png")
    
    # Low confidence predictions
    low_conf_threshold = 0.5
    low_conf = df_clean[df_clean['confidence'] < low_conf_threshold]
    
    print(f"\nLow Confidence Predictions (< {low_conf_threshold}):")
    print(f"  Count: {len(low_conf):,} ({len(low_conf)/len(df_clean)*100:.2f}%)")
    
    if len(low_conf) > 0:
        low_conf_path = os.path.join(output_dir, 'low_confidence_predictions.csv')
        low_conf_cols = ['fdc_id', 'description', 'nutritional_cluster', 
                        'predicted_cluster', 'confidence']
        low_conf[low_conf_cols].to_csv(low_conf_path, index=False)
        print(f"  Saved examples: low_confidence_predictions.csv")


def error_analysis(y, y_pred, df_clean, output_dir):
    """Analyze prediction errors in detail."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Identify errors
    errors = df_clean[y != y_pred].copy()
    correct = df_clean[y == y_pred].copy()
    
    print(f"\nError Statistics:")
    print(f"  Total predictions: {len(df_clean):,}")
    print(f"  Correct: {len(correct):,} ({len(correct)/len(df_clean)*100:.2f}%)")
    print(f"  Incorrect: {len(errors):,} ({len(errors)/len(df_clean)*100:.2f}%)")
    
    if len(errors) > 0:
        # Save error examples
        error_cols = ['fdc_id', 'description', 'nutritional_cluster', 
                     'cluster_name', 'predicted_cluster']
        
        if 'confidence' in errors.columns:
            error_cols.append('confidence')
        
        error_cols = [col for col in error_cols if col in errors.columns]
        
        # Sample errors
        error_sample = errors[error_cols].sample(n=min(100, len(errors)), random_state=RANDOM_STATE)
        error_path = os.path.join(output_dir, 'prediction_errors_sample.csv')
        error_sample.to_csv(error_path, index=False)
        print(f"\n✓ Saved error sample: prediction_errors_sample.csv")
        
        # Most common error patterns
        error_patterns = errors.groupby(['nutritional_cluster', 'predicted_cluster']).size()
        error_patterns = error_patterns.sort_values(ascending=False).head(10)
        
        print("\nMost Common Error Patterns:")
        for (true_cluster, pred_cluster), count in error_patterns.items():
            pct = count / len(errors) * 100
            print(f"  True={true_cluster} → Predicted={pred_cluster}: "
                  f"{count} errors ({pct:.1f}% of all errors)")


def cluster_purity_analysis(df_clean, output_dir):
    """Analyze cluster purity and composition."""
    print("\n" + "="*80)
    print("CLUSTER PURITY ANALYSIS")
    print("="*80)
    
    if 'cooking_method' not in df_clean.columns:
        print("Cooking method not available. Skipping purity analysis.")
        return
    
    # Calculate cluster purity based on cooking method
    cluster_cooking = pd.crosstab(
        df_clean['nutritional_cluster'],
        df_clean['cooking_method'],
        normalize='index'
    ) * 100
    
    print("\nCooking Method Distribution by Cluster (%):")
    print(cluster_cooking.round(1).to_string())
    
    # Save cluster composition
    comp_path = os.path.join(output_dir, 'cluster_composition.csv')
    cluster_cooking.to_csv(comp_path)
    print(f"\n✓ Saved: cluster_composition.csv")
    
    # Visualize
    cluster_cooking.plot(kind='bar', stacked=True, figsize=(14, 8), 
                        colormap='tab20')
    plt.xlabel('Nutritional Cluster')
    plt.ylabel('Percentage (%)')
    plt.title('Cooking Method Distribution within Nutritional Clusters')
    plt.legend(title='Cooking Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_purity_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: cluster_purity_visualization.png")


def generate_validation_report(output_dir):
    """Generate comprehensive validation report."""
    print("\n" + "="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MODEL VALIDATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Read metrics summary
    metrics_path = os.path.join(output_dir, 'validation_metrics_summary.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        report_lines.append("PERFORMANCE METRICS:")
        report_lines.append("-"*80)
        for _, row in metrics_df.iterrows():
            report_lines.append(f"  {row['Metric']}: {row['Value']:.4f}")
        report_lines.append("")
    
    # Summary
    report_lines.append("FILES GENERATED:")
    report_lines.append("-"*80)
    report_lines.append("  1. validation_metrics_summary.csv - Overall performance")
    report_lines.append("  2. validation_detailed_report.csv - Per-class metrics")
    report_lines.append("  3. confusion_matrix_detailed.png - Visual confusion matrix")
    report_lines.append("  4. misclassification_analysis.csv - Error patterns")
    report_lines.append("  5. confidence_analysis.png - Prediction confidence")
    report_lines.append("  6. low_confidence_predictions.csv - Uncertain predictions")
    report_lines.append("  7. prediction_errors_sample.csv - Example errors")
    report_lines.append("  8. cluster_composition.csv - Cluster characteristics")
    report_lines.append("  9. cluster_purity_visualization.png")
    report_lines.append("")
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, 'VALIDATION_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Saved: VALIDATION_REPORT.txt")


def main():
    print("="*80)
    print("STEP 3: MODEL VALIDATION & PERFORMANCE ANALYSIS")
    print("Comprehensive evaluation of trained classification model")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and data
    model, scaler, features, df = load_model_and_data()
    
    # Prepare validation data
    X, y, df_clean = prepare_validation_data(df, features)
    
    # Comprehensive metrics
    y_pred, y_pred_proba = comprehensive_metrics(model, scaler, X, y, OUTPUT_DIR)
    
    # Confusion matrix analysis
    confusion_matrix_analysis(y, y_pred, OUTPUT_DIR)
    
    # Prediction confidence
    prediction_confidence_analysis(y, y_pred, y_pred_proba, df_clean, OUTPUT_DIR)
    
    # Error analysis
    error_analysis(y, y_pred, df_clean, OUTPUT_DIR)
    
    # Cluster purity
    cluster_purity_analysis(df_clean, OUTPUT_DIR)
    
    # Generate final report
    generate_validation_report(OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\n✅ All analysis complete! Check the output directories for results.")


if __name__ == "__main__":
    main()