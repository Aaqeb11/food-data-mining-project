#!/usr/bin/env python3
"""
classification.py
Step 2: Train classifiers to predict nutritional cluster from nutrient values.
Uses clustering output as ground truth labels.
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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

# === CONFIGURATION ===
INPUT_FILE = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output/foods_with_nutritional_clusters.parquet"
OUTPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/ml_output/classification"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Hold out for final validation
CV_FOLDS = 5

# Whether to perform hyperparameter tuning (can be slow)
PERFORM_GRID_SEARCH = False

# === END CONFIGURATION ===


def load_data(filepath):
    """Load clustered data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Loading data from: {filepath}\n")
    
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("File must be .parquet or .csv")
    
    print(f"✓ Loaded {len(df):,} foods")
    print(f"✓ Number of nutritional clusters: {df['nutritional_cluster'].nunique()}")
    
    # Show cluster distribution
    print("\nCluster distribution:")
    cluster_counts = df['nutritional_cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        if 'cluster_name' in df.columns:
            cluster_name = df[df['nutritional_cluster'] == cluster_id]['cluster_name'].iloc[0]
            print(f"  Cluster {cluster_id} ({cluster_name}): {count:,}")
        else:
            print(f"  Cluster {cluster_id}: {count:,}")
    
    return df


def prepare_classification_data(df):
    """Prepare features and labels for classification."""
    print("\n" + "="*80)
    print("DATA PREPARATION FOR CLASSIFICATION")
    print("="*80)
    
    # Get nutrient features (exclude metadata and cluster labels)
    exclude_cols = ['fdc_id', 'description', 'cooking_method', 'data_type', 
                   'category_name', 'nutritional_cluster', 'cluster_name',
                   'pca_1', 'pca_2', 'brand_owner', 'brand_name', 
                   'branded_food_category', 'publication_date']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeatures for classification ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    # Remove any remaining missing values
    df_clean = df[feature_cols + ['nutritional_cluster', 'cluster_name', 
                                  'description', 'fdc_id']].copy()
    
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=feature_cols)
    after = len(df_clean)
    
    if before - after > 0:
        print(f"\nRemoved {before - after:,} rows with missing values")
    
    print(f"\nFinal dataset size: {after:,} foods")
    
    # Class distribution after cleaning
    print("\nClass balance after cleaning:")
    cluster_dist = df_clean['nutritional_cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_dist.items():
        pct = count / len(df_clean) * 100
        print(f"  Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
    
    # Check for class imbalance
    min_class = cluster_dist.min()
    max_class = cluster_dist.max()
    imbalance_ratio = max_class / min_class
    
    if imbalance_ratio > 3:
        print(f"\n⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        print("   Consider using stratified sampling or class weights")
    else:
        print(f"\n✓ Classes are relatively balanced (ratio: {imbalance_ratio:.1f}:1)")
    
    return df_clean, feature_cols


def split_and_scale_data(df, feature_cols):
    """Split data into train/validation/test and scale features."""
    print("\n" + "="*80)
    print("TRAIN-VALIDATION-TEST SPLIT & SCALING")
    print("="*80)
    
    X = df[feature_cols].values
    y = df['nutritional_cluster'].values
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: separate validation set from training
    val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Total:          {len(X):,} samples")
    
    # Scale features
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled")
    
    # Show feature statistics
    print("\nFeature statistics (after scaling):")
    print(f"  Mean: {X_train_scaled.mean():.6f} (should be ~0)")
    print(f"  Std:  {X_train_scaled.std():.6f} (should be ~1)")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)


def define_classifiers():
    """Define classification models to train."""
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
            verbose=0
        ),
        'Support Vector Machine': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=RANDOM_STATE,
            probability=True,
            verbose=False
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            random_state=RANDOM_STATE
        )
    }
    
    return classifiers


def train_classifiers(X_train, X_val, y_train, y_val, output_dir):
    """Train multiple classification models."""
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*80)
    
    classifiers = define_classifiers()
    results = {}
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\n{'-'*80}")
        print(f"Training: {name}")
        print(f"{'-'*80}")
        print(f"Model: {type(clf).__name__}")
        
        # Train model
        print("Fitting model...", end=" ")
        clf.fit(X_train, y_train)
        print("✓")
        
        # Predictions
        print("Making predictions...", end=" ")
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        print("✓")
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        # Cross-validation score
        print(f"Running {CV_FOLDS}-fold cross-validation...", end=" ")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, 
                                    n_jobs=-1, verbose=0)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print("✓")
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred_val, average='weighted', zero_division=0
        )
        
        print(f"\n📊 Results:")
        print(f"  Training Accuracy:   {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  CV Accuracy:         {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted):    {recall:.4f}")
        print(f"  F1-Score (weighted):  {f1:.4f}")
        
        # Check for overfitting
        overfit = train_acc - val_acc
        if overfit > 0.1:
            print(f"  ⚠️  Overfitting detected: {overfit:.4f}")
        elif overfit < 0.05:
            print(f"  ✓ Good generalization: {overfit:.4f}")
        
        # Store results
        results[name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'overfitting': overfit,
            'y_pred_val': y_pred_val,
            'cv_scores': cv_scores
        }
        
        trained_models[name] = clf
        
        # Classification report
        report = classification_report(
            y_val, y_pred_val,
            output_dict=True,
            zero_division=0
        )
        
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, 
                                   f"report_{name.replace(' ', '_').lower()}.csv")
        report_df.to_csv(report_path)
        print(f"  ✓ Saved detailed report")
        
        # Feature importance (for applicable models)
        if hasattr(clf, 'feature_importances_'):
            print(f"  ✓ Model has feature importances")
    
    return results, trained_models, y_val


def analyze_feature_importance(trained_models, feature_cols, output_dir):
    """Analyze and visualize feature importance."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance_models = {name: model for name, model in trained_models.items() 
                        if hasattr(model, 'feature_importances_')}
    
    if not importance_models:
        print("No models with feature importance available")
        return
    
    for name, model in importance_models.items():
        print(f"\n{name}:")
        
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        for idx, row in importances.head(10).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.6f}")
        
        # Save importances
        imp_path = os.path.join(output_dir, 
                               f"importance_{name.replace(' ', '_').lower()}.csv")
        importances.to_csv(imp_path, index=False)
        print(f"\n✓ Saved feature importances")
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importances.head(15)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top 15 Feature Importances - {name}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 
                                f"importance_{name.replace(' ', '_').lower()}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization")


def create_confusion_matrices(results, trained_models, y_val, output_dir):
    """Create confusion matrices for all models."""
    print("\n" + "="*80)
    print("CREATING CONFUSION MATRICES")
    print("="*80)
    
    n_models = len(results)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        cm = confusion_matrix(y_val, result['y_pred_val'])
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, annot=cm, fmt='d', cmap='Blues',
            ax=axes[idx], cbar=True, square=True, 
            linewidths=0.5, cbar_kws={'label': 'Proportion'}
        )
        
        axes[idx].set_title(
            f'{name}\n'
            f'Val Acc: {result["val_accuracy"]:.3f} | '
            f'CV: {result["cv_mean"]:.3f}±{result["cv_std"]:.3f}'
        )
        axes[idx].set_ylabel('True Cluster')
        axes[idx].set_xlabel('Predicted Cluster')
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_all_models.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_matrices_all_models.png")


def compare_models(results, output_dir):
    """Create comparison visualizations and summary."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Create summary dataframe
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            'Model': name,
            'Train_Accuracy': metrics['train_accuracy'],
            'Val_Accuracy': metrics['val_accuracy'],
            'CV_Mean': metrics['cv_mean'],
            'CV_Std': metrics['cv_std'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'Overfitting': metrics['overfitting']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Val_Accuracy', ascending=False)
    
    print("\n📊 Model Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'model_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved: model_comparison.csv")
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = range(len(summary_df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], summary_df['Train_Accuracy'], 
            width, label='Train', alpha=0.8, color='skyblue')
    ax1.bar([i + width/2 for i in x], summary_df['Val_Accuracy'], 
            width, label='Validation', alpha=0.8, color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Train vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Cross-validation scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, summary_df['CV_Mean'], yerr=summary_df['CV_Std'],
            capsize=5, alpha=0.8, color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax2.set_ylabel('CV Accuracy')
    ax2.set_title(f'{CV_FOLDS}-Fold Cross-Validation Scores')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Precision, Recall, F1
    ax3 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(summary_df))
    width = 0.25
    
    ax3.bar(x_pos - width, summary_df['Precision'], width, 
            label='Precision', alpha=0.8)
    ax3.bar(x_pos, summary_df['Recall'], width, 
            label='Recall', alpha=0.8)
    ax3.bar(x_pos + width, summary_df['F1_Score'], width, 
            label='F1-Score', alpha=0.8)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision, Recall, and F1-Score Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Overfitting analysis
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ['red' if x > 0.1 else 'green' if x < 0.05 else 'orange' 
              for x in summary_df['Overfitting']]
    ax4.barh(range(len(summary_df)), summary_df['Overfitting'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(summary_df)))
    ax4.set_yticklabels(summary_df['Model'])
    ax4.set_xlabel('Overfitting (Train - Val Accuracy)')
    ax4.set_title('Overfitting Analysis\n(Green: Good | Orange: OK | Red: High)')
    ax4.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax4.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    plt.savefig(os.path.join(output_dir, 'model_comparison_detailed.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison_detailed.png")
    
    # Identify best model
    best_model_name = summary_df.iloc[0]['Model']
    best_accuracy = summary_df.iloc[0]['Val_Accuracy']
    best_f1 = summary_df.iloc[0]['F1_Score']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Validation Accuracy: {best_accuracy:.4f}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    return best_model_name, summary_df


def test_best_model(best_model_name, trained_models, X_test, y_test, output_dir):
    """Test the best model on held-out test set."""
    print("\n" + "="*80)
    print("TESTING BEST MODEL ON TEST SET")
    print("="*80)
    
    best_model = trained_models[best_model_name]
    
    print(f"\nTesting: {best_model_name}")
    print("Making predictions on test set...", end=" ")
    y_pred_test = best_model.predict(X_test)
    print("✓")
    
    # Calculate test metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=0
    )
    
    print(f"\n📊 Test Set Performance:")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    
    # Detailed classification report
    report = classification_report(y_test, y_pred_test, zero_division=0)
    print(f"\nDetailed Classification Report:")
    print(report)
    
    # Save test results
    test_results = {
        'Model': best_model_name,
        'Test_Accuracy': test_acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }
    
    test_df = pd.DataFrame([test_results])
    test_path = os.path.join(output_dir, 'test_set_results.csv')
    test_df.to_csv(test_path, index=False)
    print(f"\n✓ Saved test results: test_set_results.csv")
    
    return test_acc


def save_best_model(best_model_name, trained_models, scaler, feature_cols, output_dir):
    """Save the best performing model and associated artifacts."""
    print("\n" + "="*80)
    print("SAVING BEST MODEL & ARTIFACTS")
    print("="*80)
    
    best_model = trained_models[best_model_name]
    
    # Save model
    model_path = os.path.join(output_dir, 'best_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Saved model: best_classifier.pkl")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'classifier_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: classifier_scaler.pkl")
    
    # Save feature list
    features_path = os.path.join(output_dir, 'feature_list.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"✓ Saved feature list: feature_list.txt ({len(feature_cols)} features)")
    
    # Save model info
    info = {
        'model_name': best_model_name,
        'model_type': type(best_model).__name__,
        'n_features': len(feature_cols),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_path = os.path.join(output_dir, 'model_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nFeatures:\n")
        for feat in feature_cols:
            f.write(f"  - {feat}\n")
    print(f"✓ Saved model info: model_info.txt")


def main():
    print("="*80)
    print("STEP 2: CLASSIFICATION - PREDICT NUTRITIONAL CLUSTERS")
    print("Train models to predict cluster membership from nutrient values")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load clustered data
    df = load_data(INPUT_FILE)
    
    # Prepare data
    df_clean, feature_cols = prepare_classification_data(df)
    
    # Split and scale
    (X_train, X_val, X_test, y_train, y_val, y_test, scaler) = split_and_scale_data(
        df_clean, feature_cols
    )
    
    # Train classifiers
    results, trained_models, y_val = train_classifiers(
        X_train, X_val, y_train, y_val, OUTPUT_DIR
    )
    
    # Analyze feature importance
    analyze_feature_importance(trained_models, feature_cols, OUTPUT_DIR)
    
    # Create confusion matrices
    create_confusion_matrices(results, trained_models, y_val, OUTPUT_DIR)
    
    # Compare models
    best_model_name, summary_df = compare_models(results, OUTPUT_DIR)
    
    # Test best model on held-out test set
    test_acc = test_best_model(best_model_name, trained_models, X_test, y_test, OUTPUT_DIR)
    
    # Save best model
    save_best_model(best_model_name, trained_models, scaler, feature_cols, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\n📁 Generated files:")
    print("  1. model_comparison.csv - Performance metrics for all models")
    print("  2. model_comparison_detailed.png - Visual comparison")
    print("  3. report_*.csv - Detailed classification reports")
    print("  4. importance_*.csv/png - Feature importances")
    print("  5. confusion_matrices_all_models.png - All confusion matrices")
    print("  6. test_set_results.csv - Final test performance")
    print("  7. best_classifier.pkl - Trained best model")
    print("  8. classifier_scaler.pkl - Feature scaler")
    print("  9. feature_list.txt - Feature names")
    print("  10. model_info.txt - Model metadata")
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    print(f"\n✅ Classification training complete!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nNext step: Run validation.py for comprehensive model validation")


if __name__ == "__main__":
    main()