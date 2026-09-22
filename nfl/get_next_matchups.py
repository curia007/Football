#!/usr/bin/env python3
"""
Generate and save upcoming NFL game matchups to a CSV file in the data directory.

This script identifies the next upcoming NFL game matchups based on the schedule
and current date/unplayed status, enriches them with team details and betting lines,
and exports the dataset to data/next_matchups.csv.
"""

import os
import argparse
import datetime
import ssl
import urllib.request
from typing import Optional
import pandas as pd

# NFL Teams Reference Registry
NFL_TEAMS = {
    # AFC East
    'BUF': {'name': 'Buffalo Bills', 'conference': 'AFC', 'division': 'East', 'city': 'Buffalo'},
    'MIA': {'name': 'Miami Dolphins', 'conference': 'AFC', 'division': 'East', 'city': 'Miami'},
    'NE':  {'name': 'New England Patriots', 'conference': 'AFC', 'division': 'East', 'city': 'Foxborough'},
    'NYJ': {'name': 'New York Jets', 'conference': 'AFC', 'division': 'East', 'city': 'East Rutherford'},
    # AFC North
    'BAL': {'name': 'Baltimore Ravens', 'conference': 'AFC', 'division': 'North', 'city': 'Baltimore'},
    'CIN': {'name': 'Cincinnati Bengals', 'conference': 'AFC', 'division': 'North', 'city': 'Cincinnati'},
    'CLE': {'name': 'Cleveland Browns', 'conference': 'AFC', 'division': 'North', 'city': 'Cleveland'},
    'PIT': {'name': 'Pittsburgh Steelers', 'conference': 'AFC', 'division': 'North', 'city': 'Pittsburgh'},
    # AFC South
    'HOU': {'name': 'Houston Texans', 'conference': 'AFC', 'division': 'South', 'city': 'Houston'},
    'IND': {'name': 'Indianapolis Colts', 'conference': 'AFC', 'division': 'South', 'city': 'Indianapolis'},
    'JAX': {'name': 'Jacksonville Jaguars', 'conference': 'AFC', 'division': 'South', 'city': 'Jacksonville'},
    'TEN': {'name': 'Tennessee Titans', 'conference': 'AFC', 'division': 'South', 'city': 'Nashville'},
    # AFC West
    'DEN': {'name': 'Denver Broncos', 'conference': 'AFC', 'division': 'West', 'city': 'Denver'},
    'KC':  {'name': 'Kansas City Chiefs', 'conference': 'AFC', 'division': 'West', 'city': 'Kansas City'},
    'LV':  {'name': 'Las Vegas Raiders', 'conference': 'AFC', 'division': 'West', 'city': 'Las Vegas'},
    'LAC': {'name': 'Los Angeles Chargers', 'conference': 'AFC', 'division': 'West', 'city': 'Los Angeles'},
    # NFC East
    'DAL': {'name': 'Dallas Cowboys', 'conference': 'NFC', 'division': 'East', 'city': 'Arlington'},
    'NYG': {'name': 'New York Giants', 'conference': 'NFC', 'division': 'East', 'city': 'East Rutherford'},
    'PHI': {'name': 'Philadelphia Eagles', 'conference': 'NFC', 'division': 'East', 'city': 'Philadelphia'},
    'WAS': {'name': 'Washington Commanders', 'conference': 'NFC', 'division': 'East', 'city': 'Landover'},
    # NFC North
    'CHI': {'name': 'Chicago Bears', 'conference': 'NFC', 'division': 'North', 'city': 'Chicago'},
    'DET': {'name': 'Detroit Lions', 'conference': 'NFC', 'division': 'North', 'city': 'Detroit'},
    'GB':  {'name': 'Green Bay Packers', 'conference': 'NFC', 'division': 'North', 'city': 'Green Bay'},
    'MIN': {'name': 'Minnesota Vikings', 'conference': 'NFC', 'division': 'North', 'city': 'Minneapolis'},
    # NFC South
    'ATL': {'name': 'Atlanta Falcons', 'conference': 'NFC', 'division': 'South', 'city': 'Atlanta'},
    'CAR': {'name': 'Carolina Panthers', 'conference': 'NFC', 'division': 'South', 'city': 'Charlotte'},
    'NO':  {'name': 'New Orleans Saints', 'conference': 'NFC', 'division': 'South', 'city': 'New Orleans'},
    'TB':  {'name': 'Tampa Bay Buccaneers', 'conference': 'NFC', 'division': 'South', 'city': 'Tampa'},
    # NFC West
    'ARI': {'name': 'Arizona Cardinals', 'conference': 'NFC', 'division': 'West', 'city': 'Glendale'},
    'LA':  {'name': 'Los Angeles Rams', 'conference': 'NFC', 'division': 'West', 'city': 'Inglewood'},
    'SF':  {'name': 'San Francisco 49ers', 'conference': 'NFC', 'division': 'West', 'city': 'Santa Clara'},
    'SEA': {'name': 'Seattle Seahawks', 'conference': 'NFC', 'division': 'West', 'city': 'Seattle'},
}

TEAM_ALIASES = {
    'SD': 'LAC',    # San Diego Chargers -> Los Angeles Chargers
    'STL': 'LA',    # St. Louis Rams -> Los Angeles Rams
    'LAR': 'LA',    # Los Angeles Rams alt alias
    'OAK': 'LV',    # Oakland Raiders -> Las Vegas Raiders
    'LVR': 'LV',    # Las Vegas Raiders alt alias
    'WSH': 'WAS',   # Washington alt alias
}

DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DEFAULT_OUTPUT_PATH = os.path.join("../data", "next_matchups.csv")


def normalize_team(team_str: str) -> str:
    """Normalize team abbreviations and full names to canonical codes."""
    if not isinstance(team_str, str):
        return team_str
    team_str = team_str.strip().upper()
    if team_str in TEAM_ALIASES:
        return TEAM_ALIASES[team_str]
    if team_str in NFL_TEAMS:
        return team_str
    for abbr, info in NFL_TEAMS.items():
        if team_str.lower() in info['name'].lower() or team_str.lower() in info['city'].lower():
            return abbr
    return team_str


def load_games_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads NFL games dataset from local file or remote nflverse repository.
    """
    candidate_paths = [
        data_path,
        os.path.join("../data", "games.csv"),
        os.path.join("../..", "data", "games.csv"),
        os.path.join("../notebooks", "games.csv"),
        "games.csv"
    ]
    
    for path in candidate_paths:
        if path and os.path.exists(path):
            print(f"📖 Loaded games dataset from local cache: {path}")
            return pd.read_csv(str(path))
            
    print(f"🌐 Fetching dataset from {DATA_URL} ...")
    try:
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_ctx = ssl._create_unverified_context()
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as response:
            df = pd.read_csv(response)
            try:
                os.makedirs("../data", exist_ok=True)
                df.to_csv(os.path.join("../data", "games.csv"), index=False)
            except Exception:
                pass
            return df
    except Exception as e:
        raise RuntimeError(f"Could not load NFL games data: {e}")


def get_next_matchups(
    df: Optional[pd.DataFrame] = None,
    as_of_date: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    all_upcoming: bool = False
) -> pd.DataFrame:
    """
    Extracts and enriches upcoming NFL game matchups.
    
    Parameters:
        df: DataFrame containing NFL games schedule/data.
        as_of_date: Reference date string (YYYY-MM-DD). Defaults to current date.
        season: Specific NFL season (e.g., 2026).
        week: Specific NFL week number (e.g., 3).
        all_upcoming: If True, returns all remaining unplayed matchups.
                      If False, returns only the immediate next week's slate.
                      
    Returns:
        pd.DataFrame containing formatted matchup records.
    """
    if df is None:
        df = load_games_dataset()
        
    df = df.copy()
    
    # Normalize team columns
    df['home_team'] = df['home_team'].apply(normalize_team)
    df['away_team'] = df['away_team'].apply(normalize_team)
    
    # Convert gameday to string YYYY-MM-DD
    df['gameday'] = df['gameday'].astype(str)
    
    # Determine reference date
    if as_of_date is None:
        as_of_date = datetime.date.today().strftime('%Y-%m-%d')
        
    # Unplayed games are those without recorded scores or outcomes
    is_unplayed = df['home_score'].isna() | df['result'].isna()
    is_future_or_today = df['gameday'] >= as_of_date
    
    upcoming_df = df[is_unplayed & is_future_or_today].copy()
    if upcoming_df.empty:
        # Fallback to any unplayed games if all dates have passed
        upcoming_df = df[is_unplayed].copy()
        
    if upcoming_df.empty:
        print("⚠️ No unplayed games found in the dataset.")
        return pd.DataFrame()
        
    # Filter by season if specified
    if season is not None:
        upcoming_df = upcoming_df[upcoming_df['season'] == season]
    else:
        next_season = upcoming_df['season'].min()
        upcoming_df = upcoming_df[upcoming_df['season'] == next_season]
        
    # Filter by week if specified, otherwise pick next upcoming week
    if week is not None:
        upcoming_df = upcoming_df[upcoming_df['week'] == week]
    elif not all_upcoming:
        next_week = upcoming_df['week'].min()
        upcoming_df = upcoming_df[upcoming_df['week'] == next_week]
        
    # Sort chronologically
    upcoming_df = upcoming_df.sort_values(by=['gameday', 'gametime', 'game_id']).reset_index(drop=True)
    
    # Enrich with team franchise metadata
    upcoming_df['away_name'] = upcoming_df['away_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('name', x))
    upcoming_df['home_name'] = upcoming_df['home_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('name', x))
    upcoming_df['away_conference'] = upcoming_df['away_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('conference', ''))
    upcoming_df['home_conference'] = upcoming_df['home_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('conference', ''))
    upcoming_df['away_division'] = upcoming_df['away_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('division', ''))
    upcoming_df['home_division'] = upcoming_df['home_team'].apply(lambda x: NFL_TEAMS.get(x, {}).get('division', ''))
    
    upcoming_df['matchup'] = upcoming_df['away_name'] + " @ " + upcoming_df['home_name']
    upcoming_df['matchup_short'] = upcoming_df['away_team'] + " @ " + upcoming_df['home_team']
    
    # Divisional game calculation if not present
    if 'div_game' not in upcoming_df.columns or upcoming_df['div_game'].isna().all():
        upcoming_df['div_game'] = (
            (upcoming_df['away_conference'] == upcoming_df['home_conference']) & 
            (upcoming_df['away_division'] == upcoming_df['home_division'])
        ).astype(int)
        
    # Select and order key columns
    desired_columns = [
        'game_id', 'season', 'week', 'gameday', 'weekday', 'gametime',
        'away_team', 'away_name', 'home_team', 'home_name',
        'matchup', 'matchup_short',
        'spread_line', 'total_line', 'away_moneyline', 'home_moneyline',
        'div_game', 'location', 'roof', 'surface', 'stadium'
    ]
    
    # Keep desired columns that exist in the dataframe
    output_cols = [col for col in desired_columns if col in upcoming_df.columns]
    # Append any other existing columns from source
    for col in upcoming_df.columns:
        if col not in output_cols and col not in ['away_score', 'home_score', 'result', 'total', 'overtime']:
            output_cols.append(col)
            
    return upcoming_df[output_cols]


def predict_matchups(matchups_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Predicts the winner of each matchup in matchups_df using the MLX model logic.
    """
    import main
    return main.predict_matchups(matchups_df, verbose=verbose)


def predict_next_matchups(
    matchups_df: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
    as_of_date: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    all_upcoming: bool = False,
    output_csv: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extracts upcoming matchups and predicts the winner of each matchup using MLX logic.
    """
    import main
    return main.predict_next_matchups(
        matchups_df=matchups_df,
        data_path=data_path,
        as_of_date=as_of_date,
        season=season,
        week=week,
        all_upcoming=all_upcoming,
        output_csv=output_csv,
        verbose=verbose
    )


def save_next_matchups_csv(
    output_path: str = DEFAULT_OUTPUT_PATH,
    as_of_date: Optional[str] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    all_upcoming: bool = False,
    predict: bool = False
) -> pd.DataFrame:
    """
    Extracts next NFL matchups, optionally predicts winners with MLX, and saves them to the specified CSV file.
    """
    # Ensure destination directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    if predict:
        return predict_next_matchups(
            as_of_date=as_of_date,
            season=season,
            week=week,
            all_upcoming=all_upcoming,
            output_csv=output_path,
            verbose=True
        )
        
    matchups_df = get_next_matchups(
        as_of_date=as_of_date,
        season=season,
        week=week,
        all_upcoming=all_upcoming
    )
    
    if not matchups_df.empty:
        cleaned_df = matchups_df.copy()
        for col in cleaned_df.columns:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna('N/A')
        cleaned_df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(matchups_df)} next NFL matchups to: {output_path}")
    else:
        print("⚠️ No matchups were generated to save.")
        
    return matchups_df


def main():
    parser = argparse.ArgumentParser(description="Produce a CSV containing the next NFL game matchups and optionally predict winners.")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--predict", "-p",
        action="store_true",
        help="Predict the winner of each matchup using Apple MLX model logic"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Reference date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        default=None,
        help="Specific NFL season (e.g. 2026)"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        default=None,
        help="Specific NFL week (e.g. 3)"
    )
    parser.add_argument(
        "--all-upcoming", "-a",
        action="store_true",
        help="Include all remaining unplayed matchups in the season instead of only the next week"
    )
    
    args = parser.parse_args()
    
    df = save_next_matchups_csv(
        output_path=args.output,
        as_of_date=args.date,
        season=args.season,
        week=args.week,
        all_upcoming=args.all_upcoming,
        predict=args.predict
    )
    
    if not df.empty and not args.predict:
        print("\n" + "=" * 95)
        print(f"🏈 NEXT NFL MATCHUPS SUMMARY (Season {df['season'].iloc[0]}, Week {df['week'].iloc[0]})")
        print("=" * 95)
        display_cols = ['gameday', 'weekday', 'gametime', 'matchup_short', 'matchup']
        if 'spread_line' in df.columns:
            display_cols.append('spread_line')
        elif 'spread' in df.columns:
            display_cols.append('spread')
        if 'total_line' in df.columns:
            display_cols.append('total_line')
        formatted_summary = df[display_cols].fillna("N/A")
        print(formatted_summary.to_string(index=False))
        print("=" * 95)


if __name__ == "__main__":
    main()
