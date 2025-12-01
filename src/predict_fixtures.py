"""
Predictions for Upcoming Fixtures

This module generates predictions for future Arsenal matches using the best trained model.
It constructs features based on the most recent historical data and outputs probabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model loading
import joblib
try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import feature engineering functions
import sys
sys.path.append(str(Path(__file__).parent))
from features import (
    create_rolling_features, create_match_context_features,
    create_opponent_features, create_statistical_features, calculate_points
)


def load_best_model(model_dir='../models'):
    """
    Load the best model (Neural Network with Class Weights).
    
    Args:
        model_dir: Directory containing models
        
    Returns:
        dict: Model, scaler, label encoder, and metadata
    """
    print("=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)
    
    model_path = Path(model_dir)
    
    # Load Neural Network with Class Weights (best performing model)
    if not DL_AVAILABLE:
        raise ImportError("TensorFlow not available. Cannot load neural network model.")
    
    model = keras.models.load_model(model_path / 'neural_network_weighted.h5')
    preprocess_data = joblib.load(model_path / 'neural_network_weighted_preprocessing.pkl')
    
    scaler = preprocess_data['scaler']
    le = preprocess_data['label_encoder']
    
    print("  ✓ Loaded Neural Network (Class Weights)")
    print(f"  Model input shape: {model.input_shape}")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }


def load_historical_data(features_file='../data/processed/arsenal_features.csv',
                        cleaned_file='../data/processed/arsenal_labeled.csv'):
    """
    Load historical Arsenal match data for feature construction.
    
    Args:
        features_file: Path to features CSV
        cleaned_file: Path to cleaned Arsenal matches CSV
        
    Returns:
        tuple: (features_df, matches_df)
    """
    print("\n" + "=" * 60)
    print("LOADING HISTORICAL DATA")
    print("=" * 60)
    
    # Load features (for getting recent match data)
    features_df = pd.read_csv(features_file)
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    features_df = features_df.sort_values('Date').reset_index(drop=True)
    
    # Load cleaned matches (for constructing new features)
    matches_df = pd.read_csv(cleaned_file)
    matches_df['Date'] = pd.to_datetime(matches_df['Date'])
    matches_df = matches_df.sort_values('Date').reset_index(drop=True)
    
    print(f"  ✓ Loaded {len(features_df)} historical matches")
    print(f"  Date range: {features_df['Date'].min()} to {features_df['Date'].max()}")
    
    return features_df, matches_df


def parse_arsenal_fixtures_file(fixtures_file):
    """
    Parse Arsenal fixtures from the epl-arsenal-2025-26.csv format.
    Filters for future matches (those without results).
    
    Args:
        fixtures_file: Path to fixtures CSV file
        
    Returns:
        DataFrame: Future fixtures with Date, Opponent, Home columns
    """
    df = pd.read_csv(fixtures_file)
    
    # Filter for Arsenal matches
    arsenal_matches = df[
        (df['Home Team'] == 'Arsenal') | (df['Away Team'] == 'Arsenal')
    ].copy()
    
    # Extract opponent and determine home/away
    fixtures_list = []
    for _, row in arsenal_matches.iterrows():
        if row['Home Team'] == 'Arsenal':
            opponent = row['Away Team']
            is_home = 1
        else:
            opponent = row['Home Team']
            is_home = 0
        
        # Parse date (format: DD/MM/YYYY HH:MM)
        date_str = str(row['Date'])
        try:
            if ' ' in date_str:
                date_part = date_str.split(' ')[0]
                date = pd.to_datetime(date_part, format='%d/%m/%Y')
            else:
                date = pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            date = pd.to_datetime(date_str, dayfirst=True)
        
        fixtures_list.append({
            'Date': date,
            'Opponent': opponent,
            'Home': is_home,
            'Result': row.get('Result', '')  # Keep result for filtering
        })
    
    fixtures_df = pd.DataFrame(fixtures_list)
    
    # Filter for future matches (those without results or empty results)
    if 'Result' in fixtures_df.columns:
        future_fixtures = fixtures_df[
            (fixtures_df['Result'].isna()) | 
            (fixtures_df['Result'] == '') |
            (fixtures_df['Result'].str.strip() == '')
        ].copy()
        print(f"  Found {len(fixtures_df)} total Arsenal matches")
        print(f"  {len(future_fixtures)} future matches (no result yet)")
        print(f"  {len(fixtures_df) - len(future_fixtures)} already played (with results)")
    else:
        future_fixtures = fixtures_df.copy()
    
    # Drop Result column
    if 'Result' in future_fixtures.columns:
        future_fixtures = future_fixtures.drop(columns=['Result'])
    
    return future_fixtures


def prepare_future_fixtures(fixtures_data=None, fixtures_file=None):
    """
    Prepare future fixtures data.
    
    Can accept:
    - List of dicts with 'Date', 'Opponent', 'Home' keys
    - CSV file path with fixtures
    - If None, creates a template for user to fill
    
    Args:
        fixtures_data: List of fixture dicts or DataFrame
        fixtures_file: Path to fixtures CSV file
        
    Returns:
        DataFrame: Future fixtures with required columns
    """
    print("\n" + "=" * 60)
    print("PREPARING FUTURE FIXTURES")
    print("=" * 60)
    
    if fixtures_file:
        # Check if it's the epl-arsenal format (has Home Team and Away Team columns)
        try:
            test_df = pd.read_csv(fixtures_file, nrows=1)
            if 'Home Team' in test_df.columns and 'Away Team' in test_df.columns:
                # Use special parser for epl-arsenal format
                fixtures_df = parse_arsenal_fixtures_file(fixtures_file)
                print(f"  ✓ Loaded {len(fixtures_df)} fixtures from epl-arsenal format file")
            else:
                # Standard format (Date, Opponent, Home columns)
                fixtures_df = pd.read_csv(fixtures_file)
                fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
                print(f"  ✓ Loaded {len(fixtures_df)} fixtures from standard format file")
        except Exception as e:
            print(f"  Error loading file: {e}")
            raise
    elif fixtures_data:
        if isinstance(fixtures_data, list):
            fixtures_df = pd.DataFrame(fixtures_data)
        else:
            fixtures_df = fixtures_data
        fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
        print(f"  ✓ Loaded {len(fixtures_df)} fixtures from data")
    else:
        # Create template
        print("  No fixtures provided. Creating template...")
        print("  Please provide fixtures as:")
        print("    - CSV file with columns: Date, Opponent, Home (1=Home, 0=Away)")
        print("    - Or list of dicts: [{'Date': '2024-08-17', 'Opponent': 'Liverpool', 'Home': 1}, ...]")
        
        # Create example template
        fixtures_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-08-17', '2024-08-24', '2024-08-31']),
            'Opponent': ['Liverpool', 'Chelsea', 'Manchester City'],
            'Home': [1, 0, 1]
        })
        print(f"\n  Created template with {len(fixtures_df)} example fixtures")
    
    # Ensure required columns
    required_cols = ['Date', 'Opponent', 'Home']
    for col in required_cols:
        if col not in fixtures_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert Home to Arsenal_Home format
    if 'Home' in fixtures_df.columns:
        fixtures_df['Arsenal_Home'] = fixtures_df['Home'].astype(int)
    
    fixtures_df = fixtures_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n  Fixtures to predict: {len(fixtures_df)}")
    print(f"  Date range: {fixtures_df['Date'].min()} to {fixtures_df['Date'].max()}")
    
    return fixtures_df


def construct_features_for_fixtures(fixtures_df, historical_features_df, historical_matches_df):
    """
    Construct features for future fixtures based on most recent historical data.
    
    Args:
        fixtures_df: DataFrame with future fixtures
        historical_features_df: Historical features (for recent form)
        historical_matches_df: Historical matches (for feature construction)
        
    Returns:
        DataFrame: Features for future fixtures
    """
    print("\n" + "=" * 60)
    print("CONSTRUCTING FEATURES FOR FUTURE FIXTURES")
    print("=" * 60)
    
    # Get the most recent historical matches (for rolling features)
    recent_matches = historical_matches_df.sort_values('Date').tail(10).copy()
    
    # Initialize features DataFrame
    fixture_features = fixtures_df.copy()
    
    # For each fixture, construct features
    for idx, fixture in fixtures_df.iterrows():
        match_date = fixture['Date']
        opponent = fixture['Opponent']
        is_home = fixture['Arsenal_Home'] == 1
        
        # Get historical matches up to this date (for rolling features)
        historical_up_to_date = historical_matches_df[
            historical_matches_df['Date'] < match_date
        ].sort_values('Date').tail(10)
        
        if len(historical_up_to_date) == 0:
            # Use most recent matches if no historical data before this date
            historical_up_to_date = recent_matches
        
        # 1. Match Context Features
        fixture_features.loc[idx, 'Arsenal_Home'] = 1 if is_home else 0
        
        # 2. Rolling Features (from last 5 matches)
        if len(historical_up_to_date) >= 5:
            last_5 = historical_up_to_date.tail(5)
        else:
            last_5 = historical_up_to_date
        
        # Calculate rolling statistics
        last_5['Arsenal_Goals'] = last_5.apply(
            lambda row: row['FTHG'] if row['Arsenal_Home'] == 1 else row['FTAG'], axis=1
        )
        last_5['Opponent_Goals'] = last_5.apply(
            lambda row: row['FTAG'] if row['Arsenal_Home'] == 1 else row['FTHG'], axis=1
        )
        last_5['Arsenal_Shots'] = last_5.apply(
            lambda row: row['HS'] if row['Arsenal_Home'] == 1 else row['AS'], axis=1
        )
        last_5['Opponent_Shots'] = last_5.apply(
            lambda row: row['AS'] if row['Arsenal_Home'] == 1 else row['HS'], axis=1
        )
        last_5['Arsenal_Shots_Target'] = last_5.apply(
            lambda row: row['HST'] if row['Arsenal_Home'] == 1 else row['AST'], axis=1
        )
        last_5['Opponent_Shots_Target'] = last_5.apply(
            lambda row: row['AST'] if row['Arsenal_Home'] == 1 else row['HST'], axis=1
        )
        last_5['Points'] = last_5['Result'].apply(calculate_points)
        
        # Rolling averages
        fixture_features.loc[idx, 'avg_goals_scored_last5'] = last_5['Arsenal_Goals'].mean()
        fixture_features.loc[idx, 'avg_goals_conceded_last5'] = last_5['Opponent_Goals'].mean()
        fixture_features.loc[idx, 'avg_points_last5'] = last_5['Points'].mean()
        fixture_features.loc[idx, 'avg_shots_last5'] = last_5['Arsenal_Shots'].mean()
        fixture_features.loc[idx, 'avg_shots_against_last5'] = last_5['Opponent_Shots'].mean()
        fixture_features.loc[idx, 'avg_shots_target_last5'] = last_5['Arsenal_Shots_Target'].mean()
        fixture_features.loc[idx, 'avg_shots_target_against_last5'] = last_5['Opponent_Shots_Target'].mean()
        
        # Additional rolling features
        fixture_features.loc[idx, 'goal_diff_last5'] = (
            fixture_features.loc[idx, 'avg_goals_scored_last5'] - 
            fixture_features.loc[idx, 'avg_goals_conceded_last5']
        )
        fixture_features.loc[idx, 'win_rate_last5'] = (last_5['Result'] == 'Win').mean()
        fixture_features.loc[idx, 'form_points_last5'] = last_5['Points'].sum()
        
        # 3. Opponent Features
        past_vs_opponent = historical_matches_df[
            (historical_matches_df['Opponent'] == opponent) &
            (historical_matches_df['Date'] < match_date)
        ]
        
        if len(past_vs_opponent) > 0:
            fixture_features.loc[idx, 'historical_win_rate_vs_opponent'] = (
                past_vs_opponent['Result'] == 'Win'
            ).mean()
            fixture_features.loc[idx, 'previous_meetings'] = len(past_vs_opponent)
        else:
            fixture_features.loc[idx, 'historical_win_rate_vs_opponent'] = 0.5
            fixture_features.loc[idx, 'previous_meetings'] = 0
        
        # 4. Statistical Features
        if len(last_5) > 0:
            goals_last5 = last_5['Arsenal_Goals'].mean()
            shots_target_last5 = last_5['Arsenal_Shots_Target'].mean()
            fixture_features.loc[idx, 'shot_conversion_rate'] = (
                goals_last5 / shots_target_last5 if shots_target_last5 > 0 else 0
            )
            fixture_features.loc[idx, 'defensive_strength'] = last_5['Opponent_Goals'].mean()
            fixture_features.loc[idx, 'attacking_strength'] = goals_last5
        else:
            fixture_features.loc[idx, 'shot_conversion_rate'] = 0
            fixture_features.loc[idx, 'defensive_strength'] = 0
            fixture_features.loc[idx, 'attacking_strength'] = 0
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(fixtures_df)} fixtures...")
    
    print(f"\n  ✓ Constructed features for {len(fixtures_df)} fixtures")
    
    return fixture_features


def generate_predictions(fixtures_with_features, model_data, output_dir='../reports'):
    """
    Generate predictions using the trained model.
    
    Args:
        fixtures_with_features: DataFrame with fixtures and features
        model_data: Dictionary with model, scaler, label_encoder
        output_dir: Directory to save predictions
        
    Returns:
        DataFrame: Predictions with probabilities
    """
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    
    model = model_data['model']
    scaler = model_data['scaler']
    le = model_data['label_encoder']
    
    # Select feature columns (same as training)
    feature_cols = [
        'avg_goals_scored_last5', 'avg_goals_conceded_last5', 'avg_points_last5',
        'avg_shots_last5', 'avg_shots_against_last5',
        'avg_shots_target_last5', 'avg_shots_target_against_last5',
        'goal_diff_last5', 'win_rate_last5', 'form_points_last5',
        'Arsenal_Home',
        'historical_win_rate_vs_opponent', 'previous_meetings',
        'shot_conversion_rate', 'defensive_strength', 'attacking_strength'
    ]
    
    # Ensure all features exist
    available_features = [col for col in feature_cols if col in fixtures_with_features.columns]
    missing_features = [col for col in feature_cols if col not in fixtures_with_features.columns]
    
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        # Fill missing with 0 or median
        for feat in missing_features:
            fixtures_with_features[feat] = 0
    
    # Extract features
    X_future = fixtures_with_features[feature_cols].copy()
    
    # Handle missing values
    X_future = X_future.fillna(X_future.median())
    X_future = X_future.replace([np.inf, -np.inf], 0)
    
    # Align features with scaler
    if hasattr(scaler, 'feature_names_in_'):
        X_future = X_future[scaler.feature_names_in_]
    
    # Scale features
    X_future_scaled = scaler.transform(X_future)
    
    # Generate predictions
    print("  Generating predictions...")
    probabilities = model.predict(X_future_scaled, verbose=0)
    
    # Generate predictions with thresholds
    predicted_class = []
    for i, probs in enumerate(probabilities):
        win_prob = probs[0]
        draw_prob = probs[1]
        loss_prob = probs[2]
        
        # Only predict Win if confidence is high
        if win_prob > 0.45:
            predicted_class.append('Win')
        elif loss_prob > 0.40:  # High loss probability
            predicted_class.append('Loss')
        elif draw_prob > 0.35:  # Draw threshold
            predicted_class.append('Draw')
        else:
            # Default to highest probability
            predicted_class.append(le.classes_[np.argmax(probs)])
    
    predicted_class = np.array(predicted_class)
    
    # Create predictions DataFrame
    predictions_df = fixtures_with_features[['Date', 'Opponent', 'Arsenal_Home']].copy()
    predictions_df['Predicted_Result'] = predicted_class
    predictions_df['P_Win'] = probabilities[:, 0]
    predictions_df['P_Draw'] = probabilities[:, 1]
    predictions_df['P_Loss'] = probabilities[:, 2]
    
    # Add confidence (max probability)
    predictions_df['Confidence'] = probabilities.max(axis=1)
    
    # Format probabilities as percentages
    predictions_df['P_Win_%'] = (predictions_df['P_Win'] * 100).round(1)
    predictions_df['P_Draw_%'] = (predictions_df['P_Draw'] * 100).round(1)
    predictions_df['P_Loss_%'] = (predictions_df['P_Loss'] * 100).round(1)
    predictions_df['Confidence_%'] = (predictions_df['Confidence'] * 100).round(1)
    
    # Add home/away label
    predictions_df['Venue'] = predictions_df['Arsenal_Home'].apply(
        lambda x: 'Home' if x == 1 else 'Away'
    )
    
    # Add "Most Likely" but emphasize probabilities
    predictions_df['Most_Likely'] = predicted_class
    predictions_df['Second_Most_Likely'] = [
        le.classes_[np.argsort(probabilities[i])[-2]] 
        for i in range(len(probabilities))
    ]
    
    # Reorder columns
    predictions_df = predictions_df[[
        'Date', 'Opponent', 'Venue', 'Predicted_Result',
        'P_Win_%', 'P_Draw_%', 'P_Loss_%', 'Confidence_%',
        'P_Win', 'P_Draw', 'P_Loss', 'Confidence'
    ]]
    
    print(f"\n  ✓ Generated predictions for {len(predictions_df)} fixtures")
    print("\n  Sample predictions:")
    print(predictions_df[['Date', 'Opponent', 'Venue', 'Predicted_Result', 'P_Win_%', 'P_Draw_%', 'P_Loss_%']].head().to_string(index=False))
    
    # Save predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path / 'arsenal_2024_25_predictions.csv', index=False)
    print(f"\n  ✓ Saved predictions: {output_path / 'arsenal_2024_25_predictions.csv'}")
    
    return predictions_df


def visualize_predictions(predictions_df, output_dir='../figures'):
    """
    Create visualizations of predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        output_dir: Directory to save figures
    """
    print("\n" + "=" * 60)
    print("CREATING PREDICTION VISUALIZATIONS")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Probability bar chart for each fixture
    fig, ax = plt.subplots(figsize=(14, max(8, len(predictions_df) * 0.4)))
    
    x_pos = np.arange(len(predictions_df))
    width = 0.25
    
    ax.barh(x_pos - width, predictions_df['P_Win_%'], width, 
            label='Win', color='green', alpha=0.7)
    ax.barh(x_pos, predictions_df['P_Draw_%'], width, 
            label='Draw', color='orange', alpha=0.7)
    ax.barh(x_pos + width, predictions_df['P_Loss_%'], width, 
            label='Loss', color='red', alpha=0.7)
    
    # Create fixture labels
    fixture_labels = [
        f"{row['Opponent']} ({'H' if row['Venue'] == 'Home' else 'A'})"
        for _, row in predictions_df.iterrows()
    ]
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(fixture_labels, fontsize=9)
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Arsenal Fixtures - Predicted Outcome Probabilities', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path / 'arsenal_fixtures_predictions.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'arsenal_fixtures_predictions.png'}")
    plt.close()
    
    # 2. Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence histogram
    axes[0].hist(predictions_df['Confidence_%'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Confidence (%)', fontsize=11)
    axes[0].set_ylabel('Number of Fixtures', fontsize=11)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Predicted results distribution
    result_counts = predictions_df['Predicted_Result'].value_counts()
    colors = {'Win': 'green', 'Draw': 'orange', 'Loss': 'red'}
    axes[1].bar(result_counts.index, result_counts.values, 
                color=[colors.get(r, 'gray') for r in result_counts.index], alpha=0.7)
    axes[1].set_xlabel('Predicted Result', fontsize=11)
    axes[1].set_ylabel('Number of Fixtures', fontsize=11)
    axes[1].set_title('Predicted Results Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'arsenal_predictions_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'arsenal_predictions_summary.png'}")
    plt.close()
    
    # 3. High-risk matches (high loss probability)
    high_risk = predictions_df[predictions_df['P_Loss_%'] > 40].sort_values('P_Loss_%', ascending=False)
    if len(high_risk) > 0:
        print(f"\n  ⚠️  High-risk matches (P(Loss) > 40%):")
        for _, match in high_risk.iterrows():
            print(f"    {match['Date'].strftime('%Y-%m-%d')}: {match['Opponent']} ({match['Venue']}) - "
                  f"Loss: {match['P_Loss_%']:.1f}%")


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("=" * 60)
    print("ARSENAL FUTURE FIXTURES PREDICTIONS")
    print("=" * 60)
    
    # Load best model
    model_data = load_best_model()
    
    # Load historical data
    historical_features, historical_matches = load_historical_data()
    
    # Prepare future fixtures - Use your 2025-26 fixtures file
    fixtures_df = prepare_future_fixtures(fixtures_file='../epl-arsenal-2025-26.csv')
    
    # Option 2: Provide as list of dicts
    # fixtures_data = [
    #     {'Date': '2024-08-17', 'Opponent': 'Liverpool', 'Home': 1},
    #     {'Date': '2024-08-24', 'Opponent': 'Chelsea', 'Home': 0},
    #     # ... more fixtures
    # ]
    # fixtures_df = prepare_future_fixtures(fixtures_data=fixtures_data)
    
    # Option 3: Create template (for now)
    # fixtures_df = prepare_future_fixtures()
    
    # Construct features
    fixtures_with_features = construct_features_for_fixtures(
        fixtures_df, historical_features, historical_matches
    )
    
    # Generate predictions
    predictions_df = generate_predictions(fixtures_with_features, model_data)
    
    # Visualize predictions
    visualize_predictions(predictions_df)
    
    print("\n" + "=" * 60)
    print("PREDICTIONS COMPLETE")
    print("=" * 60)
    print("\nTo predict your own fixtures:")
    print("  1. Create a CSV file with columns: Date, Opponent, Home")
    print("  2. Update the fixtures_file path in the script")
    print("  3. Or provide fixtures_data as a list of dictionaries")
    print("\nPredictions saved to: ../reports/arsenal_2024_25_predictions.csv")
    print("Visualizations saved to: ../figures/")
