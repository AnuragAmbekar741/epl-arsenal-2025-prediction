"""
Data Preprocessing Module

This module handles loading, cleaning, and preparing EPL match data for modeling.
It addresses all data quality issues identified in the exploration phase.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_all_seasons(data_dir):
    """
    Load all EPL season CSV files with robust error handling.
    
    Handles:
    - Encoding errors (2004-05)
    - Parsing errors (2003-04, 2004-05)
    - Missing values
    
    Args:
        data_dir: Path to raw data directory
        
    Returns:
        list: List of DataFrames (one per season)
        list: List of season names
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob('epl-*.csv'))
    
    dataframes = []
    season_names = []
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    for file in csv_files:
        try:
            # Extract season name from filename
            season = file.stem.replace('epl-', '')
            
            # Special handling for problematic files
            if '2003-04' in file.name:
                # Handle parsing error - use Python engine and skip bad lines
                df = pd.read_csv(file, engine='python', on_bad_lines='skip')
                print(f"  ✓ Loaded {file.name}: {len(df)} matches (some bad lines skipped)")
                
            elif '2004-05' in file.name:
                # Handle BOTH encoding AND parsing errors
                try:
                    df = pd.read_csv(file, encoding='latin-1', engine='python', on_bad_lines='skip')
                except Exception:
                    try:
                        df = pd.read_csv(file, encoding='cp1252', engine='python', on_bad_lines='skip')
                    except Exception:
                        df = pd.read_csv(file, encoding='utf-8', errors='replace', 
                                       engine='python', on_bad_lines='skip')
                print(f"  ✓ Loaded {file.name}: {len(df)} matches (encoding/parsing fixed)")
                
            else:
                # Normal loading for other files
                df = pd.read_csv(file)
                print(f"  ✓ Loaded {file.name}: {len(df)} matches")
            
            # Add season column
            df['Season'] = season
            dataframes.append(df)
            season_names.append(season)
            
        except Exception as e:
            print(f"  ✗ Error loading {file.name}: {e}")
    
    print(f"\nTotal seasons loaded: {len(dataframes)}")
    return dataframes, season_names


def remove_empty_columns(df):
    """
    Remove columns that are completely empty or have only missing values.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame: DataFrame with empty columns removed
    """
    # Remove columns that are completely empty
    df = df.dropna(axis=1, how='all')
    
    # Remove columns with 'Unnamed' in the name (usually empty index columns)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def select_essential_columns(df):
    """
    Select only essential columns for modeling, removing all betting odds.
    
    Keeps:
    - Basic match info (Date, Teams, Season)
    - Match results (Goals, Results)
    - Match statistics (Shots, Corners, Cards, Fouls, etc.)
    - Match metadata (Referee, Attendance, Time)
    
    Removes:
    - All betting odds columns
    
    Args:
        df: DataFrame to filter
        
    Returns:
        DataFrame: DataFrame with only essential columns
    """
    # Essential columns to keep
    essential_columns = [
        # Basic match information
        'Div', 'Date', 'HomeTeam', 'AwayTeam', 'Season',
        
        # Match results
        'FTHG', 'FTAG', 'FTR',  # Full-time
        'HTHG', 'HTAG', 'HTR',  # Half-time
        
        # Match statistics
        'HS', 'AS',           # Shots
        'HST', 'AST',         # Shots on Target
        'HF', 'AF',           # Fouls
        'HC', 'AC',           # Corners
        'HY', 'AY',           # Yellow Cards
        'HR', 'AR',           # Red Cards
        'HO', 'AO',           # Offsides
        'HHW', 'AHW',         # Hit Woodwork
        'HBP', 'ABP',         # Ball Possession (%)
        
        # Match metadata
        'Attendance', 'Referee', 'Time'
    ]
    
    # Only keep columns that exist in the DataFrame
    columns_to_keep = [col for col in essential_columns if col in df.columns]
    
    # Select only these columns
    df_filtered = df[columns_to_keep].copy()
    
    print(f"  Selected {len(columns_to_keep)} essential columns (removed {len(df.columns) - len(columns_to_keep)} betting/other columns)")
    
    return df_filtered


def standardize_team_names(df):
    """
    Standardize team names across all seasons.
    
    Common variations to handle:
    - Man United / Manchester United
    - Man City / Manchester City
    - etc.
    
    Args:
        df: DataFrame with HomeTeam and AwayTeam columns
        
    Returns:
        DataFrame: DataFrame with standardized team names
    """
    # Dictionary of team name mappings (add more as needed)
    team_name_mapping = {
        # Manchester United variations
        'Man United': 'Man United',
        'Manchester United': 'Man United',
        'Man Utd': 'Man United',
        
        # Manchester City variations
        'Man City': 'Man City',
        'Manchester City': 'Man City',
        
        # Other common variations (add as you discover them)
        # 'Tottenham': 'Tottenham',
        # 'Spurs': 'Tottenham',
    }
    
    # Apply mappings to HomeTeam and AwayTeam
    if 'HomeTeam' in df.columns:
        df['HomeTeam'] = df['HomeTeam'].replace(team_name_mapping)
    if 'AwayTeam' in df.columns:
        df['AwayTeam'] = df['AwayTeam'].replace(team_name_mapping)
    
    return df


def convert_dates(df):
    """
    Convert date column to datetime format.
    
    Handles different date formats:
    - DD/MM/YY (e.g., "19/08/00")
    - DD/MM/YYYY (e.g., "19/08/2000")
    
    Args:
        df: DataFrame with Date column
        
    Returns:
        DataFrame: DataFrame with Date as datetime
    """
    if 'Date' not in df.columns:
        return df
    
    # Try different date formats
    date_formats = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d']
    
    for fmt in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
            # If most dates converted successfully, break
            if df['Date'].notna().sum() > len(df) * 0.8:
                break
        except:
            continue
    
    # If still not converted, try automatic parsing
    if df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    
    return df


def filter_arsenal_matches(df, arsenal_name='Arsenal'):
    """
    Filter DataFrame to only include matches where Arsenal played.
    
    Args:
        df: DataFrame with HomeTeam and AwayTeam columns
        arsenal_name: Name of Arsenal team in the data
        
    Returns:
        DataFrame: Filtered DataFrame with only Arsenal matches
    """
    # Find Arsenal name variations
    home_teams = set(df['HomeTeam'].dropna().unique())
    away_teams = set(df['AwayTeam'].dropna().unique())
    all_teams = home_teams.union(away_teams)
    
    # Find Arsenal variations
    arsenal_variations = [team for team in all_teams if 'Arsenal' in str(team)]
    
    if not arsenal_variations:
        print(f"Warning: Could not find '{arsenal_name}' in team names")
        return pd.DataFrame()
    
    # Use the first match (usually just 'Arsenal')
    actual_arsenal_name = arsenal_variations[0]
    
    # Filter for Arsenal matches
    arsenal_matches = df[
        (df['HomeTeam'] == actual_arsenal_name) | 
        (df['AwayTeam'] == actual_arsenal_name)
    ].copy()
    
    print(f"Found {len(arsenal_matches)} Arsenal matches")
    return arsenal_matches


def create_arsenal_labels(df, arsenal_name='Arsenal'):
    """
    Create target labels for Arsenal matches (Win/Draw/Loss from Arsenal's perspective).
    
    Args:
        df: DataFrame with Arsenal matches
        arsenal_name: Name of Arsenal team
        
    Returns:
        DataFrame: DataFrame with 'Result' column added
    """
    df = df.copy()
    
    # Find actual Arsenal name in data
    home_teams = set(df['HomeTeam'].dropna().unique())
    away_teams = set(df['AwayTeam'].dropna().unique())
    all_teams = home_teams.union(away_teams)
    arsenal_variations = [team for team in all_teams if 'Arsenal' in str(team)]
    actual_arsenal_name = arsenal_variations[0] if arsenal_variations else arsenal_name
    
    # Create Result column from Arsenal's perspective
    def get_result(row):
        if row['HomeTeam'] == actual_arsenal_name:
            # Arsenal is home team
            if row['FTR'] == 'H':
                return 'Win'
            elif row['FTR'] == 'A':
                return 'Loss'
            else:
                return 'Draw'
        else:
            # Arsenal is away team
            if row['FTR'] == 'A':
                return 'Win'
            elif row['FTR'] == 'H':
                return 'Loss'
            else:
                return 'Draw'
    
    df['Result'] = df.apply(get_result, axis=1)
    
    # Add home/away flag for Arsenal
    df['Arsenal_Home'] = (df['HomeTeam'] == actual_arsenal_name).astype(int)
    df['Opponent'] = df.apply(
        lambda row: row['AwayTeam'] if row['HomeTeam'] == actual_arsenal_name 
        else row['HomeTeam'], 
        axis=1
    )
    
    return df


def calculate_league_standings(df, season, up_to_date=None):
    """
    Calculate league standings for a given season up to a specific date.
    
    This ensures we only use information from matches played BEFORE the current match.
    
    Args:
        df: DataFrame with all EPL matches
        season: Season to calculate standings for (e.g., '2000-01')
        up_to_date: Only include matches up to this date (datetime)
        
    Returns:
        DataFrame: League standings with columns: Position, Team, Points, GoalDiff, GoalsFor, GoalsAgainst, Matches
    """
    # Filter for the season
    season_df = df[df['Season'] == season].copy()
    
    # Filter for matches up to the specified date (if provided)
    if up_to_date is not None:
        season_df = season_df[season_df['Date'] < up_to_date]
    
    # Initialize standings dictionary
    standings = {}
    
    # Process each match
    for _, match in season_df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        home_goals = match['FTHG']
        away_goals = match['FTAG']
        result = match['FTR']
        
        # Skip matches with missing critical data
        if pd.isna(home_goals) or pd.isna(away_goals) or pd.isna(result):
            continue
        
        # Initialize teams if not seen before
        if home_team not in standings:
            standings[home_team] = {
                'Points': 0,
                'GoalsFor': 0,
                'GoalsAgainst': 0,
                'Matches': 0
            }
        if away_team not in standings:
            standings[away_team] = {
                'Points': 0,
                'GoalsFor': 0,
                'GoalsAgainst': 0,
                'Matches': 0
            }
        
        # Update home team (convert to int to ensure no float issues)
        standings[home_team]['GoalsFor'] += int(home_goals)
        standings[home_team]['GoalsAgainst'] += int(away_goals)
        standings[home_team]['Matches'] += 1
        
        if result == 'H':
            standings[home_team]['Points'] += 3
        elif result == 'D':
            standings[home_team]['Points'] += 1
            standings[away_team]['Points'] += 1
        else:  # result == 'A'
            standings[away_team]['Points'] += 3
        
        # Update away team (convert to int to ensure no float issues)
        standings[away_team]['GoalsFor'] += int(away_goals)
        standings[away_team]['GoalsAgainst'] += int(home_goals)
        standings[away_team]['Matches'] += 1
    
    # Convert to DataFrame
    standings_df = pd.DataFrame(standings).T
    standings_df['Team'] = standings_df.index
    standings_df['GoalDiff'] = standings_df['GoalsFor'] - standings_df['GoalsAgainst']
    
    # Sort by standings rules: Points (desc), GoalDiff (desc), GoalsFor (desc)
    standings_df = standings_df.sort_values(
        ['Points', 'GoalDiff', 'GoalsFor'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Add position
    standings_df['Position'] = range(1, len(standings_df) + 1)
    
    # Ensure integer types for numeric columns (fill NaN with 0 first)
    standings_df['Points'] = standings_df['Points'].fillna(0).astype(int)
    standings_df['GoalDiff'] = standings_df['GoalDiff'].fillna(0).astype(int)
    standings_df['GoalsFor'] = standings_df['GoalsFor'].fillna(0).astype(int)
    standings_df['GoalsAgainst'] = standings_df['GoalsAgainst'].fillna(0).astype(int)
    standings_df['Matches'] = standings_df['Matches'].fillna(0).astype(int)
    standings_df['Position'] = standings_df['Position'].astype(int)
    
    # Reorder columns
    standings_df = standings_df[['Position', 'Team', 'Points', 'GoalDiff', 'GoalsFor', 'GoalsAgainst', 'Matches']]
    
    return standings_df


def calculate_all_season_standings(df, output_path='../data/processed/league_standings.csv'):
    """
    Calculate final league standings for all seasons and save to CSV.
    
    Args:
        df: DataFrame with all EPL matches (cleaned)
        output_path: Path to save the standings CSV
        
    Returns:
        DataFrame: Combined standings for all seasons
    """
    print("\n" + "=" * 60)
    print("CALCULATING LEAGUE STANDINGS FOR ALL SEASONS")
    print("=" * 60)
    
    df['Date'] = pd.to_datetime(df['Date'])
    all_standings = []
    
    seasons = sorted(df['Season'].unique())
    
    for season in seasons:
        print(f"\nCalculating standings for {season}...")
        
        # Calculate final standings (no date filter = all matches in season)
        standings = calculate_league_standings(df, season, up_to_date=None)
        standings['Season'] = season
        
        all_standings.append(standings)
        
        # Show top 5 teams
        print(f"  Top 5 teams:")
        for _, row in standings.head(5).iterrows():
            print(f"    {int(row['Position']):2d}. {row['Team']:20s} - {int(row['Points']):3d} pts (GD: {int(row['GoalDiff']):+3d})")
    
    # Combine all seasons
    combined_standings = pd.concat(all_standings, ignore_index=True)
    
    # Reorder columns
    combined_standings = combined_standings[['Season', 'Position', 'Team', 'Points', 'GoalDiff', 
                                             'GoalsFor', 'GoalsAgainst', 'Matches']]
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_standings.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved league standings: {output_path}")
    print(f"  Total seasons: {len(seasons)}")
    print(f"  Total rows: {len(combined_standings):,}")
    
    return combined_standings


def preprocess_data(raw_data_dir='../data/raw', output_dir='../data/processed'):
    """
    Main preprocessing function that orchestrates all cleaning steps.
    
    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Path to save processed data
        
    Returns:
        tuple: (cleaned_all_matches, arsenal_matches)
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load all seasons
    print("\n[1/6] Loading all season files...")
    all_dataframes, season_names = load_all_seasons(raw_data_dir)
    
    if len(all_dataframes) == 0:
        print("Error: No data loaded!")
        return None, None
    
    # Step 2: Combine all seasons
    print("\n[2/6] Combining all seasons...")
    master_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    print(f"  Combined DataFrame: {len(master_df):,} rows × {len(master_df.columns)} columns")
    
    # Step 3: Remove empty columns
    print("\n[3/7] Removing empty columns...")
    initial_cols = len(master_df.columns)
    master_df = remove_empty_columns(master_df)
    removed_cols = initial_cols - len(master_df.columns)
    print(f"  Removed {removed_cols} empty/unnamed columns")
    print(f"  Remaining columns: {len(master_df.columns)}")
    
    # Step 4: Select only essential columns (remove betting odds)
    print("\n[4/7] Selecting essential columns (removing betting odds)...")
    master_df = select_essential_columns(master_df)
    print(f"  Essential columns: {len(master_df.columns)}")
    print(f"  Columns kept: {', '.join(master_df.columns.tolist()[:10])}...")
    
    # Step 5: Standardize team names
    print("\n[5/7] Standardizing team names...")
    master_df = standardize_team_names(master_df)
    print("  Team names standardized")
    
    # Step 6: Convert dates
    print("\n[6/7] Converting dates to datetime...")
    master_df = convert_dates(master_df)
    date_converted = master_df['Date'].notna().sum() if 'Date' in master_df.columns else 0
    print(f"  Converted {date_converted:,} dates to datetime format")
    
    # Step 7: Filter for Arsenal matches
    print("\n[7/8] Filtering for Arsenal matches...")
    arsenal_matches = filter_arsenal_matches(master_df)
    
    if len(arsenal_matches) > 0:
        # Create labels for Arsenal matches
        arsenal_matches = create_arsenal_labels(arsenal_matches)
        print(f"  Created Result labels: {arsenal_matches['Result'].value_counts().to_dict()}")
    
    # Step 8: Calculate league standings for all seasons
    print("\n[8/8] Calculating league standings for all seasons...")
    
    # Convert output_dir to Path and create directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate and save standings
    standings_df = calculate_all_season_standings(master_df, 
                                                  output_path=output_path / 'league_standings.csv')
    
    # Print summary
    total_seasons = len(standings_df['Season'].unique()) if 'Season' in standings_df.columns else len(standings_df)
    print(f"  ✓ Calculated standings for {total_seasons} seasons")
    print(f"  ✓ Saved to: {output_path / 'league_standings.csv'}")
    
    # Save processed data
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    # Save cleaned master data
    master_output = output_path / 'epl_cleaned.csv'
    master_df.to_csv(master_output, index=False)
    print(f"\n✓ Saved cleaned EPL data: {master_output}")
    print(f"  Rows: {len(master_df):,}, Columns: {len(master_df.columns)}")
    
    # Save Arsenal matches
    if len(arsenal_matches) > 0:
        arsenal_output = output_path / 'arsenal_labeled.csv'
        arsenal_matches.to_csv(arsenal_output, index=False)
        print(f"\n✓ Saved Arsenal matches: {arsenal_output}")
        print(f"  Rows: {len(arsenal_matches):,}, Columns: {len(arsenal_matches.columns)}")
    
    # Standings already saved in calculate_all_season_standings function
    print(f"\n✓ League standings already saved: {output_path / 'league_standings.csv'}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return master_df, arsenal_matches


if __name__ == '__main__':
    # Run preprocessing
    cleaned_data, arsenal_data = preprocess_data()
    
    if cleaned_data is not None:
        print("\nSummary:")
        print(f"  Total matches: {len(cleaned_data):,}")
        print(f"  Arsenal matches: {len(arsenal_data):,}")
        print(f"  Date range: {cleaned_data['Date'].min()} to {cleaned_data['Date'].max()}")
        print(f"\n✓ All processed files saved to data/processed/:")
        print(f"  - epl_cleaned.csv")
        print(f"  - arsenal_labeled.csv")
        print(f"  - league_standings.csv")
