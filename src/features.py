"""
Feature Engineering Module

This module creates features for predicting Arsenal match outcomes.
All features use only past information to avoid data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_points(result):
    """
    Convert match result to points.
    
    Args:
        result: Match result ('Win', 'Draw', 'Loss')
        
    Returns:
        int: Points (3 for Win, 1 for Draw, 0 for Loss)
    """
    if result == 'Win':
        return 3
    elif result == 'Draw':
        return 1
    else:  # Loss
        return 0


def create_rolling_features(df, window=5):
    """
    Create rolling window features from past N matches.
    
    For each match, calculates statistics from the previous N matches.
    This ensures no data leakage (only uses past information).
    
    Args:
        df: DataFrame with Arsenal matches, sorted by date
        window: Number of previous matches to use (default: 5)
        
    Returns:
        DataFrame: DataFrame with rolling features added
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate goals scored and conceded from Arsenal's perspective
    df['Arsenal_Goals'] = df.apply(
        lambda row: row['FTHG'] if row['Arsenal_Home'] == 1 else row['FTAG'],
        axis=1
    )
    df['Opponent_Goals'] = df.apply(
        lambda row: row['FTAG'] if row['Arsenal_Home'] == 1 else row['FTHG'],
        axis=1
    )
    
    # Calculate shots from Arsenal's perspective
    df['Arsenal_Shots'] = df.apply(
        lambda row: row['HS'] if row['Arsenal_Home'] == 1 else row['AS'],
        axis=1
    )
    df['Opponent_Shots'] = df.apply(
        lambda row: row['AS'] if row['Arsenal_Home'] == 1 else row['HS'],
        axis=1
    )
    
    # Calculate shots on target
    df['Arsenal_Shots_Target'] = df.apply(
        lambda row: row['HST'] if row['Arsenal_Home'] == 1 else row['AST'],
        axis=1
    )
    df['Opponent_Shots_Target'] = df.apply(
        lambda row: row['AST'] if row['Arsenal_Home'] == 1 else row['HST'],
        axis=1
    )
    
    # Calculate points
    df['Points'] = df['Result'].apply(calculate_points)
    
    # Rolling averages (using only past matches - shift by 1 to avoid data leakage)
    rolling_cols = {
        'Arsenal_Goals': 'avg_goals_scored',
        'Opponent_Goals': 'avg_goals_conceded',
        'Points': 'avg_points',
        'Arsenal_Shots': 'avg_shots',
        'Opponent_Shots': 'avg_shots_against',
        'Arsenal_Shots_Target': 'avg_shots_target',
        'Opponent_Shots_Target': 'avg_shots_target_against'
    }
    
    for col, prefix in rolling_cols.items():
        # Calculate rolling mean
        rolling_mean = df[col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{prefix}_last{window}'] = rolling_mean
        
        # Calculate rolling sum (for total goals, points)
        if col in ['Arsenal_Goals', 'Opponent_Goals', 'Points']:
            rolling_sum = df[col].shift(1).rolling(window=window, min_periods=1).sum()
            df[f'{prefix.replace("avg", "total")}_last{window}'] = rolling_sum
    
    # Goal difference in last N matches
    df['goal_diff_last5'] = (
        df['Arsenal_Goals'].shift(1).rolling(window=window, min_periods=1).sum() -
        df['Opponent_Goals'].shift(1).rolling(window=window, min_periods=1).sum()
    )
    
    # Win rate in last N matches
    df['wins_last5'] = (df['Result'].shift(1) == 'Win').rolling(window=window, min_periods=1).sum()
    df['win_rate_last5'] = df['wins_last5'] / window
    
    # Form (points in last 5 matches)
    df['form_points_last5'] = df['Points'].shift(1).rolling(window=window, min_periods=1).sum()
    
    return df


def create_match_context_features(df):
    """
    Create match context features (home/away only).
    Removed: days_since_last_match, matchday, month, day_of_week
    (These don't account for other competitions like Champions League, FA Cup)
    
    Args:
        df: DataFrame with Arsenal matches
        
    Returns:
        DataFrame: DataFrame with context features added
    """
    df = df.copy()
    
    # Home/Away indicator
    if 'Arsenal_Home' not in df.columns:
        df['Arsenal_Home'] = (df['HomeTeam'] == 'Arsenal').astype(int)
    
    return df


def create_opponent_features(df):
    """
    Create features related to opponent strength.
    
    Note: This is simplified. In a full implementation, you might calculate
    opponent's league position, recent form, etc.
    
    Args:
        df: DataFrame with Arsenal matches
        
    Returns:
        DataFrame: DataFrame with opponent features added
    """
    df = df.copy()
    
    # Opponent name (already exists, but ensure it's there)
    if 'Opponent' not in df.columns:
        df['Opponent'] = df.apply(
            lambda row: row['AwayTeam'] if row['Arsenal_Home'] == 1 
            else row['HomeTeam'],
            axis=1
        )
    
    # Historical performance vs opponent (from past matches only)
    # This calculates Arsenal's win rate vs each opponent in previous meetings
    opponent_win_rates = {}
    
    for idx, row in df.iterrows():
        opponent = row['Opponent']
        date = row['Date']
        
        # Get past matches vs this opponent
        past_matches = df[
            (df['Opponent'] == opponent) & 
            (df['Date'] < date)
        ]
        
        if len(past_matches) > 0:
            win_rate = (past_matches['Result'] == 'Win').mean()
            opponent_win_rates[idx] = win_rate
        else:
            opponent_win_rates[idx] = 0.5  # Default if no past matches
    
    df['historical_win_rate_vs_opponent'] = pd.Series(opponent_win_rates)
    
    # Number of previous meetings with opponent
    df['previous_meetings'] = df.apply(
        lambda row: len(df[
            (df['Opponent'] == row['Opponent']) & 
            (df['Date'] < row['Date'])
        ]),
        axis=1
    )
    
    return df


def add_opponent_league_position(df, standings_file='../data/processed/league_standings.csv'):
    """
    Add opponent's league position before the match.
    Uses league standings data to get opponent's position at the time of the match.
    
    Args:
        df: DataFrame with Arsenal matches
        standings_file: Path to league standings CSV
        
    Returns:
        DataFrame: DataFrame with opponent_league_position added
    """
    df = df.copy()
    
    # Load league standings
    standings_path = Path(standings_file)
    if not standings_path.exists():
        print(f"  Warning: {standings_path} not found. Skipping opponent league position.")
        df['opponent_league_position'] = np.nan
        return df
    
    standings_df = pd.read_csv(standings_path)
    standings_df['Date'] = pd.to_datetime(standings_df['Date']) if 'Date' in standings_df.columns else None
    
    # For each match, find opponent's league position before the match
    opponent_positions = []
    
    for idx, row in df.iterrows():
        opponent = row['Opponent']
        match_date = row['Date']
        season = row['Season']
        
        # Get standings for this season up to this date
        season_standings = standings_df[
            (standings_df['Season'] == season) & 
            (standings_df['Date'] < match_date)
        ] if 'Date' in standings_df.columns else standings_df[standings_df['Season'] == season]
        
        # Find opponent's most recent position
        opponent_standings = season_standings[season_standings['Team'] == opponent]
        
        if len(opponent_standings) > 0:
            # Get the most recent position
            if 'Date' in standings_df.columns:
                latest = opponent_standings.sort_values('Date').iloc[-1]
            else:
                latest = opponent_standings.iloc[0]
            opponent_positions.append(latest['Position'])
        else:
            # If no standings found, use NaN
            opponent_positions.append(np.nan)
    
    df['opponent_league_position'] = opponent_positions
    
    return df


def create_statistical_features(df):
    """
    Create additional statistical features from match stats.
    
    Args:
        df: DataFrame with Arsenal matches
        
    Returns:
        DataFrame: DataFrame with statistical features added
    """
    df = df.copy()
    
    # Shot conversion rate (goals per shot on target) - from past matches
    df['shots_target_last5'] = df['Arsenal_Shots_Target'].shift(1).rolling(
        window=5, min_periods=1
    ).mean()
    
    df['goals_last5'] = df['Arsenal_Goals'].shift(1).rolling(
        window=5, min_periods=1
    ).mean()
    
    df['shot_conversion_rate'] = np.where(
        df['shots_target_last5'] > 0,
        df['goals_last5'] / df['shots_target_last5'],
        0
    )
    
    # Defensive strength (goals conceded per match)
    df['defensive_strength'] = df['Opponent_Goals'].shift(1).rolling(
        window=5, min_periods=1
    ).mean()
    
    # Attacking strength (goals scored per match)
    df['attacking_strength'] = df['Arsenal_Goals'].shift(1).rolling(
        window=5, min_periods=1
    ).mean()
    
    return df


def create_all_features(df):
    """
    Main function to create all features.
    
    Args:
        df: DataFrame with Arsenal matches (from arsenal_labeled.csv)
        
    Returns:
        DataFrame: DataFrame with all features added
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (critical for rolling features)
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nStarting with {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Step 1: Rolling features
    print("\n[1/5] Creating rolling window features (last 5 matches)...")
    df = create_rolling_features(df, window=5)
    print("  ✓ Rolling features created")
    
    # Step 2: Match context features (simplified - only home/away)
    print("\n[2/5] Creating match context features...")
    df = create_match_context_features(df)
    print("  ✓ Context features created")
    
    # Step 3: Opponent features
    print("\n[3/5] Creating opponent features...")
    df = create_opponent_features(df)
    print("  ✓ Opponent features created")
    
    # Step 4: Opponent league position
    print("\n[4/5] Adding opponent league position...")
    df = add_opponent_league_position(df)
    print("  ✓ Opponent league position added")
    
    # Step 5: Statistical features
    print("\n[5/5] Creating statistical features...")
    df = create_statistical_features(df)
    print("  ✓ Statistical features created")
    
    # Select feature columns (FINALIZED FEATURE SET)
    feature_columns = [
        # Rolling features (last 5 matches)
        'avg_goals_scored_last5', 'avg_goals_conceded_last5', 'avg_points_last5',
        'avg_shots_last5', 'avg_shots_against_last5',
        'avg_shots_target_last5', 'avg_shots_target_against_last5',
        'goal_diff_last5', 'win_rate_last5', 'form_points_last5',
        
        # Match context (simplified)
        'Arsenal_Home',
        
        # Opponent features
        'historical_win_rate_vs_opponent', 'previous_meetings', 'opponent_league_position',
        
        # Statistical features
        'shot_conversion_rate', 'defensive_strength', 'attacking_strength',
        
        # Target and metadata
        'Result', 'Season', 'Date', 'Opponent'
    ]
    
    # Only keep columns that exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix
    feature_df = df[available_features].copy()
    
    print(f"\n" + "=" * 60)
    print(f"FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nTotal features created: {len(available_features) - 4}")  # Exclude Result, Season, Date, Opponent
    print(f"Feature columns:")
    for i, col in enumerate([c for c in available_features if c not in ['Result', 'Season', 'Date', 'Opponent']], 1):
        print(f"  {i:2d}. {col}")
    
    return feature_df


def save_features(df, output_path='../data/processed/arsenal_features.csv'):
    """
    Save feature matrix to CSV.
    
    Args:
        df: DataFrame with features
        output_path: Path to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved feature matrix: {output_path}")
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")


if __name__ == '__main__':
    # Load Arsenal matches
    arsenal_file = Path('../data/processed/arsenal_labeled.csv')
    
    if not arsenal_file.exists():
        print(f"Error: {arsenal_file} not found!")
        print("Please run data_preprocessing.py first.")
    else:
        print("Loading Arsenal matches...")
        df = pd.read_csv(arsenal_file)
        print(f"Loaded {len(df)} Arsenal matches")
        
        # Create features
        feature_df = create_all_features(df)
        
        # Save features
        save_features(feature_df)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total matches: {len(feature_df):,}")
        print(f"Total features: {len(feature_df.columns) - 3}")  # Exclude Result, Season, Date
        print(f"Date range: {feature_df['Date'].min()} to {feature_df['Date'].max()}")
        print(f"\nResult distribution:")
        print(feature_df['Result'].value_counts())
