"""
Feature Importance Analysis Module

This module analyzes feature importance across all models:
1. Random Forest: Direct feature importances
2. Logistic Regression: Coefficient analysis
3. Deep Learning: Permutation importance
4. Comparative analysis and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model loading and analysis
import joblib
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False


def load_data_and_models(model_dir='../models', features_file='../data/processed/arsenal_features.csv'):
    """
    Load data and all trained models.
    
    Returns:
        dict: Contains data and models
    """
    print("=" * 60)
    print("LOADING DATA AND MODELS")
    print("=" * 60)
    
    # Load features
    df = pd.read_csv(features_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Separate features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['Result', 'Season', 'Date', 'Opponent']]
    X = df[feature_cols].copy()
    y = df['Result'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)
    
    # Drop opponent_league_position if exists (models trained without it)
    if 'opponent_league_position' in X.columns:
        X = X.drop(columns=['opponent_league_position'])
        feature_cols = [col for col in feature_cols if col != 'opponent_league_position']
    
    # Train/test split
    train_mask = df['Season'] != '2024-25'
    test_mask = df['Season'] == '2024-25'
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Load models
    models = {}
    model_path = Path(model_dir)
    
    # Random Forest
    try:
        rf_data = joblib.load(model_path / 'random_forest.pkl')
        models['random_forest'] = {
            'model': rf_data['model'],
            'label_encoder': rf_data['label_encoder'],
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
        print("  ✓ Loaded Random Forest")
    except Exception as e:
        print(f"  ✗ Random Forest: {e}")
    
    # Logistic Regression
    try:
        lr_data = joblib.load(model_path / 'logreg_baseline.pkl')
        models['logistic_regression'] = {
            'model': lr_data['model'],
            'scaler': lr_data['scaler'],
            'label_encoder': lr_data['label_encoder'],
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
        print("  ✓ Loaded Logistic Regression")
    except Exception as e:
        print(f"  ✗ Logistic Regression: {e}")
    
    # Neural Network (with class weights - best model)
    if DL_AVAILABLE:
        try:
            nn_model = keras.models.load_model(model_path / 'neural_network_weighted.h5')
            nn_preprocess = joblib.load(model_path / 'neural_network_weighted_preprocessing.pkl')
            scaler = nn_preprocess['scaler']
            
            # Align features
            X_train_nn = X_train.copy()
            X_test_nn = X_test.copy()
            if hasattr(scaler, 'feature_names_in_'):
                X_train_nn = X_train_nn[scaler.feature_names_in_]
                X_test_nn = X_test_nn[scaler.feature_names_in_]
            
            X_train_scaled = scaler.transform(X_train_nn)
            X_test_scaled = scaler.transform(X_test_nn)
            
            models['neural_network'] = {
                'model': nn_model,
                'scaler': scaler,
                'label_encoder': nn_preprocess['label_encoder'],
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': X_train_nn.columns.tolist() if hasattr(X_train_nn, 'columns') else list(range(X_train_scaled.shape[1]))
            }
            print("  ✓ Loaded Neural Network (Class Weights)")
        except Exception as e:
            print(f"  ✗ Neural Network: {e}")
    
    return models, feature_cols


def analyze_random_forest_importance(model_data, output_dir='../figures'):
    """
    Analyze feature importance for Random Forest.
    
    Args:
        model_data: Dictionary with model and data
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST - FEATURE IMPORTANCE")
    print("=" * 60)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'].values)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Random Forest - Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path / 'random_forest_feature_importance.png'}")
    plt.close()
    
    return importance_df


def analyze_logistic_regression_coefficients(model_data, output_dir='../figures'):
    """
    Analyze coefficients for Logistic Regression.
    
    Args:
        model_data: Dictionary with model and data
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION - COEFFICIENT ANALYSIS")
    print("=" * 60)
    
    model = model_data['model']
    le = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Get coefficients for each class
    coefficients = model.coef_  # Shape: (n_classes, n_features)
    class_names = le.classes_
    
    # Create DataFrame for each class
    all_coefs = []
    for i, class_name in enumerate(class_names):
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients[i],
            'Class': class_name
        }).sort_values('Coefficient', key=abs, ascending=False)
        all_coefs.append(coef_df)
        
        print(f"\n{class_name} - Top 10 Features (by absolute coefficient):")
        print(coef_df.head(10)[['Feature', 'Coefficient']].to_string(index=False))
    
    # Plot coefficients for each class
    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 6))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, (ax, class_name, coef_df) in enumerate(zip(axes, class_names, all_coefs)):
        top_coefs = coef_df.head(15)
        colors = ['green' if x > 0 else 'red' for x in top_coefs['Coefficient'].values]
        ax.barh(range(len(top_coefs)), top_coefs['Coefficient'].values, color=colors)
        ax.set_yticks(range(len(top_coefs)))
        ax.set_yticklabels(top_coefs['Feature'].values)
        ax.set_xlabel('Coefficient Value', fontsize=11)
        ax.set_title(f'{class_name} - Top 15 Features', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Logistic Regression - Feature Coefficients by Class', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path / 'logistic_regression_coefficients.png'}")
    plt.close()
    
    return all_coefs


def analyze_neural_network_permutation_importance(model_data, output_dir='../figures', n_repeats=10):
    """
    Analyze permutation importance for Neural Network.
    
    Args:
        model_data: Dictionary with model and data
        output_dir: Directory to save plots
        n_repeats: Number of times to permute each feature
    """
    print("\n" + "=" * 60)
    print("NEURAL NETWORK - PERMUTATION IMPORTANCE")
    print("=" * 60)
    
    model = model_data['model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    le = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Encode labels
    y_test_encoded = le.transform(y_test)
    
    # Get baseline score
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    baseline_score = accuracy_score(y_test_encoded, y_pred)
    
    print(f"\nBaseline accuracy: {baseline_score:.4f}")
    print(f"Calculating permutation importance (n_repeats={n_repeats})...")
    print("  This may take a few minutes...")
    
    # Manual permutation importance calculation (more reliable for Keras models)
    n_features = X_test.shape[1]
    importances = np.zeros(n_features)
    importances_std = np.zeros(n_features)
    
    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            # Create a copy and permute the feature
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, i])
            
            # Predict with permuted feature
            y_pred_proba = model.predict(X_test_permuted, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            score = accuracy_score(y_test_encoded, y_pred)
            scores.append(score)
        
        # Importance is the decrease in score
        importances[i] = baseline_score - np.mean(scores)
        importances_std[i] = np.std(scores)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{n_features} features...")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': importances,
        'Importance_Std': importances_std
    }).sort_values('Importance_Mean', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['Importance_Mean'].values, 
             xerr=top_features['Importance_Std'].values, capsize=3)
    plt.yticks(y_pos, top_features['Feature'].values)
    plt.xlabel('Permutation Importance (Accuracy Decrease)', fontsize=12)
    plt.title('Neural Network - Top 20 Feature Importances (Permutation)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'neural_network_permutation_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path / 'neural_network_permutation_importance.png'}")
    plt.close()
    
    return importance_df


def compare_feature_importance_across_models(models, output_dir='../figures'):
    """
    Compare feature importance across all models.
    
    Args:
        models: Dictionary of all models
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("COMPARATIVE FEATURE IMPORTANCE")
    print("=" * 60)
    
    all_importances = {}
    
    # Random Forest
    if 'random_forest' in models:
        rf_importance = analyze_random_forest_importance(models['random_forest'], output_dir)
        all_importances['Random Forest'] = rf_importance.set_index('Feature')['Importance']
    
    # Logistic Regression (use absolute coefficients averaged across classes)
    if 'logistic_regression' in models:
        lr_coefs = analyze_logistic_regression_coefficients(models['logistic_regression'], output_dir)
        # Average absolute coefficients across classes
        lr_importance = pd.DataFrame({
            'Feature': lr_coefs[0]['Feature'],
            'Importance': np.mean([np.abs(df['Coefficient'].values) for df in lr_coefs], axis=0)
        }).sort_values('Importance', ascending=False)
        all_importances['Logistic Regression'] = lr_importance.set_index('Feature')['Importance']
    
    # Neural Network
    if 'neural_network' in models:
        nn_importance = analyze_neural_network_permutation_importance(models['neural_network'], output_dir)
        all_importances['Neural Network'] = nn_importance.set_index('Feature')['Importance_Mean']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_importances)
    comparison_df = comparison_df.fillna(0)
    
    # Normalize each column to 0-1 scale for comparison
    comparison_df_norm = comparison_df.div(comparison_df.max(axis=0), axis=1)
    
    # Get top features across all models
    top_features = comparison_df_norm.mean(axis=1).sort_values(ascending=False).head(20).index
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Normalized comparison
    comparison_df_norm.loc[top_features].plot(kind='barh', ax=axes[0], width=0.8)
    axes[0].set_xlabel('Normalized Importance', fontsize=12)
    axes[0].set_title('Top 20 Features - Normalized Importance Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Raw values
    comparison_df.loc[top_features].plot(kind='barh', ax=axes[1], width=0.8)
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_title('Top 20 Features - Raw Importance Comparison', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path / 'feature_importance_comparison.png'}")
    plt.close()
    
    # Save comparison to CSV
    comparison_df.to_csv(output_path / 'feature_importance_comparison.csv')
    print(f"✓ Saved: {output_path / 'feature_importance_comparison.csv'}")
    
    return comparison_df


def interpret_feature_importance(comparison_df, output_dir='../reports'):
    """
    Generate interpretation of feature importance.
    
    Args:
        comparison_df: DataFrame with feature importances
        output_dir: Directory to save report
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE INTERPRETATION")
    print("=" * 60)
    
    # Get top features
    avg_importance = comparison_df.mean(axis=1).sort_values(ascending=False)
    top_features = avg_importance.head(10)
    
    print("\nTop 10 Most Important Features (Average across models):")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feature:35s}: {importance:.4f}")
    
    # Categorize features
    rolling_features = [f for f in top_features.index if 'last5' in f]
    context_features = [f for f in top_features.index if f in ['Arsenal_Home']]
    opponent_features = [f for f in top_features.index if 'opponent' in f or 'historical' in f or 'previous' in f]
    statistical_features = [f for f in top_features.index if any(x in f for x in ['strength', 'conversion', 'rate'])]
    
    print("\nFeature Categories in Top 10:")
    print(f"  Rolling Window Features: {len(rolling_features)}")
    print(f"  Match Context Features: {len(context_features)}")
    print(f"  Opponent Features: {len(opponent_features)}")
    print(f"  Statistical Features: {len(statistical_features)}")
    
    # Save interpretation
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'feature_importance_interpretation.txt', 'w') as f:
        f.write("FEATURE IMPORTANCE INTERPRETATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top 10 Most Important Features:\n")
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            f.write(f"  {i:2d}. {feature:35s}: {importance:.4f}\n")
        f.write("\n\nKey Insights:\n")
        f.write("- Rolling window features (last 5 matches) are highly important\n")
        f.write("- Recent form (goals, points, win rate) strongly predicts outcomes\n")
        f.write("- Home advantage (Arsenal_Home) is a significant factor\n")
        f.write("- Opponent strength and historical performance matter\n")
    
    print(f"\n✓ Saved interpretation: {output_path / 'feature_importance_interpretation.txt'}")


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Load data and models
    models, feature_cols = load_data_and_models()
    
    # Analyze each model type
    comparison_df = compare_feature_importance_across_models(models)
    
    # Generate interpretation
    interpret_feature_importance(comparison_df)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nAll visualizations saved to: ../figures/")
    print("Interpretation report saved to: ../reports/")
