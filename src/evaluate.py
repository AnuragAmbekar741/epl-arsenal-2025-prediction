"""
Model Evaluation Module

This module evaluates all trained models and generates visualizations:
- Confusion matrices
- ROC curves (one-vs-rest for multi-class)
- Classification reports
- Performance comparison charts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Model loading
import joblib
try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False


def create_sequences_local(X, y, sequence_length):
    """
    Create sequences from time series data (local copy to avoid import issues).
    """
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
    
    return np.array(X_seq), np.array(y_seq)


def load_model_and_data(model_name, model_dir='../models', features_file='../data/processed/arsenal_features.csv'):
    """
    Load a trained model and prepare test data.
    
    Args:
        model_name: Name of the model ('logreg', 'random_forest', 'neural_network', etc.)
        model_dir: Directory containing models
        features_file: Path to features CSV
        
    Returns:
        tuple: (model, X_test, y_test, predictions, probabilities, label_encoder)
    """
    model_path = Path(model_dir)
    features_path = Path(features_file)
    
    # Load features
    df = pd.read_csv(features_path)
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
    
    # Test split (2024-25 season)
    test_mask = df['Season'] == '2024-25'
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    
    # IMPORTANT: Drop opponent_league_position if it exists (models were trained without it)
    if 'opponent_league_position' in X_test.columns:
        print(f"  Note: Dropping 'opponent_league_position' (not in training data)")
        X_test = X_test.drop(columns=['opponent_league_position'])
    
    # Load model based on type
    if model_name == 'logreg_baseline' or model_name == 'logreg':
        model_data = joblib.load(model_path / 'logreg_baseline.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        le = model_data['label_encoder']
        
        # Ensure feature alignment
        X_test = X_test[scaler.feature_names_in_] if hasattr(scaler, 'feature_names_in_') else X_test
        
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = le.transform(y_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
    elif model_name == 'random_forest' or model_name == 'rf':
        model_data = joblib.load(model_path / 'random_forest.pkl')
        model = model_data['model']
        le = model_data['label_encoder']
        
        # Ensure feature alignment
        X_test = X_test[model.feature_names_in_] if hasattr(model, 'feature_names_in_') else X_test
        
        y_test_encoded = le.transform(y_test)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
    elif model_name == 'neural_network' or model_name == 'nn':
        if not DL_AVAILABLE:
            return None
        
        model = keras.models.load_model(model_path / 'neural_network.h5')
        preprocess_data = joblib.load(model_path / 'neural_network_preprocessing.pkl')
        scaler = preprocess_data['scaler']
        le = preprocess_data['label_encoder']
        
        # Ensure feature alignment
        if hasattr(scaler, 'feature_names_in_'):
            X_test = X_test[scaler.feature_names_in_]
        
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = le.transform(y_test)
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    elif model_name == 'neural_network_weighted' or model_name == 'nn_weighted':
        if not DL_AVAILABLE:
            return None
        
        model = keras.models.load_model(model_path / 'neural_network_weighted.h5')
        preprocess_data = joblib.load(model_path / 'neural_network_weighted_preprocessing.pkl')
        scaler = preprocess_data['scaler']
        le = preprocess_data['label_encoder']
        
        # Ensure feature alignment
        if hasattr(scaler, 'feature_names_in_'):
            X_test = X_test[scaler.feature_names_in_]
        
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = le.transform(y_test)
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    elif model_name == 'lstm':
        if not DL_AVAILABLE:
            return None
        
        model = keras.models.load_model(model_path / 'lstm_model.h5')
        preprocess_data = joblib.load(model_path / 'lstm_preprocessing.pkl')
        scaler = preprocess_data['scaler']
        le = preprocess_data['label_encoder']
        sequence_length = preprocess_data['sequence_length']
        
        # Ensure feature alignment
        if hasattr(scaler, 'feature_names_in_'):
            X_test = X_test[scaler.feature_names_in_]
        
        # Create sequences
        X_test_scaled = scaler.transform(X_test)
        X_test_seq, y_test_seq = create_sequences_local(X_test_scaled, y_test, sequence_length)
        y_test_encoded = le.transform(y_test_seq)
        y_pred_proba = model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    elif model_name == 'hybrid_lstm':
        if not DL_AVAILABLE:
            return None
        
        model = keras.models.load_model(model_path / 'hybrid_lstm.h5')
        preprocess_data = joblib.load(model_path / 'hybrid_lstm_preprocessing.pkl')
        scaler = preprocess_data['scaler']
        le = preprocess_data['label_encoder']
        sequence_length = preprocess_data['sequence_length']
        
        # Ensure feature alignment
        if hasattr(scaler, 'feature_names_in_'):
            X_test = X_test[scaler.feature_names_in_]
        
        # Create sequences
        X_test_scaled = scaler.transform(X_test)
        X_test_seq, y_test_seq = create_sequences_local(X_test_scaled, y_test, sequence_length)
        X_test_current = X_test_seq[:, -1, :]
        y_test_encoded = le.transform(y_test_seq)
        y_pred_proba = model.predict([X_test_seq, X_test_current])
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_encoded': y_test_encoded,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'label_encoder': le
    }


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir='../figures'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model
        output_dir: Directory to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix: {output_path / f'{model_name}_confusion_matrix.png'}")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, model_name, output_dir='../figures'):
    """
    Plot ROC curves for multi-class classification (one-vs-rest).
    
    Args:
        y_true: True labels (encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        model_name: Name of the model
        output_dir: Directory to save figure
    """
    n_classes = len(class_names)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved ROC curves: {output_path / f'{model_name}_roc_curves.png'}")
    plt.close()
    
    return roc_auc


def plot_precision_recall_curves(y_true, y_pred_proba, class_names, model_name, output_dir='../figures'):
    """
    Plot Precision-Recall curves for each class.
    
    Args:
        y_true: True labels (encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        model_name: Name of the model
        output_dir: Directory to save figure
    """
    n_classes = len(class_names)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP = {average_precision[i]:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{model_name} - Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'{model_name}_precision_recall.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved Precision-Recall curves: {output_path / f'{model_name}_precision_recall.png'}")
    plt.close()
    
    return average_precision


def evaluate_model(model_name, model_dir='../models', output_dir='../figures', 
                   features_file='../data/processed/arsenal_features.csv'):
    """
    Comprehensive evaluation of a single model.
    
    Args:
        model_name: Name of the model to evaluate
        model_dir: Directory containing models
        output_dir: Directory to save visualizations
        features_file: Path to features CSV
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load model and data
        results = load_model_and_data(model_name, model_dir, features_file)
        if results is None:
            print(f"  ✗ Could not load model: {model_name}")
            return None
        
        y_test = results['y_test']
        y_test_encoded = results['y_test_encoded']
        y_pred = results['predictions']
        y_pred_proba = results['probabilities']
        le = results['label_encoder']
        
        # Get class names
        class_names = le.classes_
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=class_names))
        
        # Confusion matrix
        print(f"\nGenerating confusion matrix...")
        plot_confusion_matrix(y_test_encoded, y_pred, class_names, model_name, output_dir)
        
        # ROC curves
        print(f"Generating ROC curves...")
        roc_auc = plot_roc_curves(y_test_encoded, y_pred_proba, class_names, model_name, output_dir)
        print(f"  ROC AUC scores:")
        for i, class_name in enumerate(class_names):
            print(f"    {class_name}: {roc_auc[i]:.4f}")
        print(f"    Micro-average: {roc_auc['micro']:.4f}")
        
        # Precision-Recall curves
        print(f"Generating Precision-Recall curves...")
        avg_precision = plot_precision_recall_curves(y_test_encoded, y_pred_proba, class_names, model_name, output_dir)
        print(f"  Average Precision scores:")
        for i, class_name in enumerate(class_names):
            print(f"    {class_name}: {avg_precision[i]:.4f}")
        
        return {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'classification_report': classification_report(y_test_encoded, y_pred, target_names=class_names, output_dict=True)
        }
        
    except Exception as e:
        print(f"  ✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_all_models(model_names, model_dir='../models', output_dir='../figures',
                      features_file='../data/processed/arsenal_features.csv'):
    """
    Evaluate and compare all models.
    
    Args:
        model_names: List of model names to evaluate
        model_dir: Directory containing models
        output_dir: Directory to save visualizations
        features_file: Path to features CSV
    """
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    all_results = []
    
    for model_name in model_names:
        result = evaluate_model(model_name, model_dir, output_dir, features_file)
        if result:
            all_results.append(result)
    
    # Create comparison summary
    if all_results:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        comparison_data = []
        for result in all_results:
            roc_auc = result['roc_auc']
            comparison_data.append({
                'Model': result['model_name'],
                'ROC AUC (Win)': roc_auc.get(0, np.nan),
                'ROC AUC (Draw)': roc_auc.get(1, np.nan),
                'ROC AUC (Loss)': roc_auc.get(2, np.nan),
                'ROC AUC (Micro-avg)': roc_auc.get('micro', np.nan),
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nROC AUC Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path / 'model_evaluation_comparison.csv', index=False)
        print(f"\n✓ Saved evaluation comparison: {output_path / 'model_evaluation_comparison.csv'}")
    
    return all_results


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Models to evaluate
    models_to_evaluate = [
        'logreg_baseline',
        'random_forest',
        'neural_network',
        'neural_network_weighted',
        'lstm',
        'hybrid_lstm'
    ]
    
    # Evaluate all models
    results = compare_all_models(models_to_evaluate)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nAll visualizations saved to: ../figures/")
