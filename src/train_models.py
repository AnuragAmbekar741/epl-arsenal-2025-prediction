"""
Model Training Module

This module trains multiple models for predicting Arsenal match outcomes:
1. Baseline: Logistic Regression
2. Traditional ML: Random Forest, XGBoost
3. Deep Learning: Feedforward Neural Network, LSTM

All models use time-based train/test split to avoid data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    DL_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow/Keras not available. Deep learning models will be skipped.")
    DL_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns


def load_features(file_path='../data/processed/arsenal_features.csv'): 
    """
    Load the feature matrix.
    
    Args:
        file_path: Path to features CSV
        
    Returns:
        DataFrame: Feature matrix
    """
    features_file = Path(file_path)
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_csv(features_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} matches")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Seasons: {sorted(df['Season'].unique())}")
    
    return df


def create_train_test_split(df, test_season='2024-25', validation_season=None):
    """
    Create time-based train/test split using seasons.
    
    Args:
        df: DataFrame with features
        test_season: Season to use as test set (default: '2024-25')
        validation_season: Optional season for validation (e.g., '2023-24')
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_val, y_val)
               X_val and y_val are None if validation_season is None
    """
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Separate features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['Result', 'Season', 'Date', 'Opponent']]
    
    X = df[feature_cols].copy()
    y = df['Result'].copy()
    
    # Handle missing values more robustly
    print(f"\nHandling missing values...")
    print(f"  Missing values before: {X.isnull().sum().sum()}")
    
    # Drop columns that are completely empty
    cols_before = len(X.columns)
    X = X.dropna(axis=1, how='all')  # Drop columns with all NaN
    cols_after = len(X.columns)
    if cols_before != cols_after:
        print(f"  Dropped {cols_before - cols_after} columns with all NaN values")
    
    # Fill remaining NaN values
    # For numeric columns: fill with median, or 0 if median is NaN
    for col in X.select_dtypes(include=[np.number]).columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(median_val)
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)  # Fill any remaining NaN (from inf replacement) with 0
    
    print(f"  Missing values after: {X.isnull().sum().sum()}")
    print(f"  Infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Verify no NaN or inf remain
    if X.isnull().sum().sum() > 0:
        print(f"  Warning: Still have {X.isnull().sum().sum()} NaN values!")
    if np.isinf(X.select_dtypes(include=[np.number])).sum().sum() > 0:
        print(f"  Warning: Still have infinite values!")
    
    # Split by season
    train_mask = df['Season'] != test_season
    test_mask = df['Season'] == test_season
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    
    print(f"\nTraining set:")
    print(f"  Matches: {len(X_train):,}")
    print(f"  Seasons: {sorted(df[train_mask]['Season'].unique())}")
    print(f"  Date range: {df[train_mask]['Date'].min()} to {df[train_mask]['Date'].max()}")
    
    print(f"\nTest set:")
    print(f"  Matches: {len(X_test):,}")
    print(f"  Season: {test_season}")
    
    # Optional validation set
    X_val = None
    y_val = None
    if validation_season:
        val_mask = df['Season'] == validation_season
        X_val = X[val_mask].copy()
        y_val = y[val_mask].copy()
        print(f"\nValidation set:")
        print(f"  Matches: {len(X_val):,}")
        print(f"  Season: {validation_season}")
    
    # Check for class distribution
    print(f"\nClass distribution (Training):")
    print(y_train.value_counts())
    
    return X_train, X_test, y_train, y_test, X_val, y_val, feature_cols


def train_logistic_regression(X_train, y_train, X_test, y_test, 
                              model_dir='../models'):
    """
    Train Logistic Regression baseline model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_dir: Directory to save model
        
    Returns:
        dict: Model and metrics
    """
    print("\n" + "=" * 60)
    print("BASELINE: LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Final check for NaN/inf before scaling
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("  Warning: Found NaN values. Filling with 0...")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    # Replace infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train model
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    model.fit(X_train_scaled, y_train_encoded)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    print(f"\n✓ Model trained")
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save model
    model_path = Path(model_dir) / 'logreg_baseline.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }, model_path)
    print(f"\n✓ Saved model: {model_path}")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_random_forest(X_train, y_train, X_test, y_test, 
                       model_dir='../models', n_estimators=100, max_depth=10):
    """
    Train Random Forest model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_dir: Directory to save model
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        dict: Model and metrics
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train model
    print(f"\nTraining Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_encoded)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    print(f"\n✓ Model trained")
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save model
    model_path = Path(model_dir) / 'random_forest.pkl'
    joblib.dump({
        'model': model,
        'label_encoder': le
    }, model_path)
    print(f"\n✓ Saved model: {model_path}")
    
    return {
        'model': model,
        'label_encoder': le,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_neural_network(X_train, y_train, X_test, y_test, 
                        model_dir='../models', epochs=100, batch_size=32):
    """
    Train Feedforward Neural Network.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        dict: Model and metrics
    """
    if not DL_AVAILABLE:
        print("Skipping Neural Network (TensorFlow not available)")
        return None
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING: FEEDFORWARD NEURAL NETWORK")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels to categorical
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    y_train_cat = to_categorical(y_train_encoded, num_classes=3)
    y_test_cat = to_categorical(y_test_encoded, num_classes=3)
    
    # Build model
    n_features = X_train_scaled.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        str(Path(model_dir) / 'neural_network_best.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    print(f"\n✓ Model trained")
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save model
    model_path = Path(model_dir) / 'neural_network.h5'
    model.save(model_path)
    joblib.dump({'scaler': scaler, 'label_encoder': le}, 
                Path(model_dir) / 'neural_network_preprocessing.pkl')
    print(f"\n✓ Saved model: {model_path}")
    
    # Plot training history
    plot_training_history(history, 'neural_network', model_dir)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_neural_network_with_class_weights(X_train, y_train, X_test, y_test, 
                                           model_dir='../models', epochs=100, batch_size=32):
    """
    Train Feedforward Neural Network with class weights to handle imbalance.
    """
    if not DL_AVAILABLE:
        return None
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING: NEURAL NETWORK (WITH CLASS WEIGHTS)")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    y_train_cat = to_categorical(y_train_encoded, num_classes=3)
    y_test_cat = to_categorical(y_test_encoded, num_classes=3)
    
    # Calculate class weights (inverse frequency)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    # Build model (deeper/wider)
    n_features = X_train_scaled.shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        str(Path(model_dir) / 'neural_network_weighted_best.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train with class weights
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    
    print(f"\n✓ Model trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save
    model_path = Path(model_dir) / 'neural_network_weighted.h5'
    model.save(model_path)
    joblib.dump({'scaler': scaler, 'label_encoder': le}, 
                Path(model_dir) / 'neural_network_weighted_preprocessing.pkl')
    
    plot_training_history(history, 'neural_network_weighted', model_dir)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_lstm(X_train, y_train, X_test, y_test, 
              model_dir='../models', sequence_length=10, epochs=100, batch_size=32):
    """
    Train LSTM model for sequential pattern learning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_dir: Directory to save model
        sequence_length: Number of previous matches to use as sequence
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        dict: Model and metrics
    """
    if not DL_AVAILABLE:
        print("Skipping LSTM (TensorFlow not available)")
        return None
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING: LSTM MODEL")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    print(f"\nCreating sequences (length={sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
    
    print(f"  Training sequences: {len(X_train_seq):,}")
    print(f"  Test sequences: {len(X_test_seq):,}")
    
    # Encode labels
    le = LabelEncoder()
    y_train_cat = to_categorical(le.fit_transform(y_train_seq), num_classes=3)
    y_test_cat = to_categorical(le.transform(y_test_seq), num_classes=3)
    
    # Build LSTM model
    n_features = X_train_seq.shape[2]
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        str(Path(model_dir) / 'lstm_model_best.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_encoded = le.transform(y_test_seq)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    print(f"\n✓ Model trained")
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save model
    model_path = Path(model_dir) / 'lstm_model.h5'
    model.save(model_path)
    joblib.dump({
        'scaler': scaler,
        'label_encoder': le,
        'sequence_length': sequence_length
    }, Path(model_dir) / 'lstm_preprocessing.pkl')
    print(f"\n✓ Saved model: {model_path}")
    
    # Plot training history
    plot_training_history(history, 'lstm_model', model_dir)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_hybrid_lstm_model(X_train, y_train, X_test, y_test, 
                           model_dir='../models', sequence_length=10, epochs=100):
    """
    Train Hybrid LSTM model: Combines sequential patterns (LSTM) with match features (Dense).
    """
    if not DL_AVAILABLE:
        return None
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING: HYBRID LSTM + FEATURE MODEL")
    print("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
    
    # Encode labels
    le = LabelEncoder()
    y_train_cat = to_categorical(le.fit_transform(y_train_seq), num_classes=3)
    y_test_cat = to_categorical(le.transform(y_test_seq), num_classes=3)
    
    # Get current match features (last element of sequence)
    X_train_current = X_train_seq[:, -1, :]  # Last match in sequence
    X_test_current = X_test_seq[:, -1, :]
    
    # Build hybrid model: Two parallel branches
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Concatenate
    
    # Branch 1: LSTM for sequential patterns
    lstm_input = Input(shape=(sequence_length, X_train_seq.shape[2]))
    lstm_branch = LSTM(64, return_sequences=True)(lstm_input)
    lstm_branch = Dropout(0.3)(lstm_branch)
    lstm_branch = LSTM(32)(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    
    # Branch 2: Dense for current match features
    feature_input = Input(shape=(X_train_current.shape[1],))
    feature_branch = Dense(64, activation='relu')(feature_input)
    feature_branch = Dropout(0.3)(feature_branch)
    feature_branch = Dense(32, activation='relu')(feature_branch)
    
    # Merge branches
    merged = Concatenate()([lstm_branch, feature_branch])
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(3, activation='softmax')(merged)
    
    model = Model(inputs=[lstm_input, feature_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        str(Path(model_dir) / 'hybrid_lstm_best.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        [X_train_seq, X_train_current], y_train_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict([X_test_seq, X_test_current])
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_encoded = le.transform(y_test_seq)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average=None)
    recall = recall_score(y_test_encoded, y_pred, average=None)
    f1 = f1_score(y_test_encoded, y_pred, average=None)
    
    print(f"\n✓ Model trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-score: {f1}")
    
    # Save
    model_path = Path(model_dir) / 'hybrid_lstm.h5'
    model.save(model_path)
    joblib.dump({
        'scaler': scaler,
        'label_encoder': le,
        'sequence_length': sequence_length
    }, Path(model_dir) / 'hybrid_lstm_preprocessing.pkl')
    
    plot_training_history(history, 'hybrid_lstm', model_dir)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def train_gru_model(X_train, y_train, X_test, y_test, 
                   model_dir='../models', sequence_length=10, epochs=100):
    """
    Train GRU (Gated Recurrent Unit) model - simpler than LSTM.
    """
    if not DL_AVAILABLE:
        return None
    
    from tensorflow.keras.layers import GRU
    
    # Similar to LSTM but use GRU instead
    # ... (implementation similar to train_lstm but with GRU layers)
    pass # Placeholder for GRU implementation


def create_sequences(X, y, sequence_length):
    """
    Create sequences from time series data.
    
    Args:
        X: Feature matrix
        y: Target labels
        sequence_length: Length of each sequence
        
    Returns:
        tuple: (X_sequences, y_sequences)
    """
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
    
    return np.array(X_seq), np.array(y_seq)


def plot_training_history(history, model_name, output_dir='../figures'):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Keras training history
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved training history plot: {output_path / f'{model_name}_training_history.png'}")


def compare_models(results_dict, output_dir='../reports'):
    """
    Create comprehensive model comparison.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        output_dir: Directory to save comparison report
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison = []
    for model_name, results in results_dict.items():
        if results is None:
            continue
        
        comparison.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision (Win)': results['precision'][0] if len(results['precision']) > 0 else np.nan,
            'Precision (Draw)': results['precision'][1] if len(results['precision']) > 1 else np.nan,
            'Precision (Loss)': results['precision'][2] if len(results['precision']) > 2 else np.nan,
            'Recall (Win)': results['recall'][0] if len(results['recall']) > 0 else np.nan,
            'Recall (Draw)': results['recall'][1] if len(results['recall']) > 1 else np.nan,
            'Recall (Loss)': results['recall'][2] if len(results['recall']) > 2 else np.nan,
            'F1 (Win)': results['f1'][0] if len(results['f1']) > 0 else np.nan,
            'F1 (Draw)': results['f1'][1] if len(results['f1']) > 1 else np.nan,
            'F1 (Loss)': results['f1'][2] if len(results['f1']) > 2 else np.nan,
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)
    print(f"\n✓ Saved comparison: {output_path / 'model_comparison.csv'}")
    
    return comparison_df


if __name__ == '__main__':
    # Load features
    print("=" * 60)
    print("ARSENAL MATCH PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    df = load_features()
    
    # Create train/test split
    X_train, X_test, y_train, y_test, X_val, y_val, feature_cols = create_train_test_split(
        df, test_season='2024-25', validation_season='2023-24'
    )
    
    # Train all models
    results = {}
    
    # 1. Baseline: Logistic Regression
    results['Logistic Regression'] = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # 2. Random Forest
    results['Random Forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    # 3. Neural Network
    if DL_AVAILABLE:
        results['Neural Network'] = train_neural_network(X_train, y_train, X_test, y_test)
        
        # 4. Neural Network with Class Weights
        results['Neural Network (Class Weights)'] = train_neural_network_with_class_weights(X_train, y_train, X_test, y_test)
        
        # 5. LSTM
        results['LSTM'] = train_lstm(X_train, y_train, X_test, y_test)
        
        # 6. Hybrid LSTM + Feature Model
        results['Hybrid LSTM + Feature Model'] = train_hybrid_lstm_model(X_train, y_train, X_test, y_test)
        
        # 7. GRU Model
        results['GRU Model'] = train_gru_model(X_train, y_train, X_test, y_test)
    
    # Compare all models
    comparison_df = compare_models(results)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nBest model by accuracy:")
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"  {best_model['Model']}: {best_model['Accuracy']:.4f}")
