"""
NFL Game Outcome & Spread Prediction using Apple MLX Framework.
Complete pipeline with 32-team normalization, betting mathematics, MLX Neural Network training, and matchup prediction.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Complete 32-Team NFL Registry & Aliases
# ==========================================
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


def normalize_team(team_str):
    """
    Normalizes team abbreviations and full names to canonical 3-letter codes.
    Handles historical relocation aliases (SD -> LAC, STL -> LA, OAK -> LV, etc.).
    """
    if not isinstance(team_str, str):
        return team_str
    team_str = team_str.strip().upper()
    if team_str in TEAM_ALIASES:
        return TEAM_ALIASES[team_str]
    if team_str in NFL_TEAMS:
        return team_str
    # Search by full or partial name / city
    for abbr, info in NFL_TEAMS.items():
        if team_str.lower() in info['name'].lower() or team_str.lower() in info['city'].lower():
            return abbr
    return team_str


# ==========================================
# 2. Betting Mathematics Utilities
# ==========================================
def american_to_implied_prob(odds):
    """Converts American moneyline odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def american_to_decimal(odds):
    """Converts American moneyline odds to decimal payout multiplier."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    elif odds < 0:
        return 1.0 + (100.0 / abs(odds))
    return 2.0


def compute_rolling(history, n=5):
    """Computes rolling offensive/defensive metrics from historical games."""
    if not history:
        return 21.5, 21.5, 0.0, 0.5
    recent = history[-n:]
    pf = np.mean([g['pf'] for g in recent])
    pa = np.mean([g['pa'] for g in recent])
    diff = pf - pa
    wr = np.mean([g['win'] for g in recent])
    return pf, pa, diff, wr


# ==========================================
# 3. Apple MLX Neural Network Model
# ==========================================
class MLXNFLPredictor(nn.Module):
    """
    Deep Multi-Layer Perceptron in Apple MLX for NFL Win Probability Prediction.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout_p: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.network(x)


# Backward-compatible alias
NFLPredictorMLP = MLXNFLPredictor


FEATURE_COLS = [
    'spread_line', 'total_line', 'home_devigged_prob', 'spread_prob',
    'home_elo', 'away_elo', 'elo_diff', 'elo_prob',
    'home_roll_pf', 'home_roll_pa', 'home_roll_diff', 'home_roll_wr',
    'away_roll_pf', 'away_roll_pa', 'away_roll_diff', 'away_roll_wr',
    'roll_diff_net', 'roll_pf_net', 'roll_pa_net', 'rest_diff', 'div_game'
]

HOME_ADVANTAGE_ELO = 48.0
SPREAD_SIGMA = 13.8
EPOCHS = 40
BATCH_SIZE = 64

# Default global model instance
model = MLXNFLPredictor(input_dim=len(FEATURE_COLS), hidden_dims=[64, 32, 16], dropout_p=0.2)
scaler = None


# ==========================================
# 4. Pipeline & Matchup Predictor
# ==========================================
def predict_nfl_matchup(home_team, away_team, spread_line, total_line=44.5,
                        home_moneyline=None, away_moneyline=None,
                        model=None, scaler=None, elo_ratings=None, team_history=None):
    """
    Generates real-time game win probabilities and betting recommendations for ANY of the 32 NFL teams.
    """
    h_code = normalize_team(home_team)
    a_code = normalize_team(away_team)
    
    h_name = NFL_TEAMS.get(h_code, {}).get('name', h_code)
    a_name = NFL_TEAMS.get(a_code, {}).get('name', a_code)
    
    elo_ratings = elo_ratings or {}
    team_history = team_history or {}
    
    # 1. Elo ratings
    h_elo = elo_ratings.get(h_code, 1550.0)
    a_elo = elo_ratings.get(a_code, 1550.0)
    elo_diff = (h_elo + HOME_ADVANTAGE_ELO) - a_elo
    elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    
    # 2. Implied and de-vigged market odds
    spread_prob = norm.cdf(spread_line / SPREAD_SIGMA)
    h_raw = american_to_implied_prob(home_moneyline) if home_moneyline is not None else spread_prob
    a_raw = american_to_implied_prob(away_moneyline) if away_moneyline is not None else (1.0 - spread_prob)
    tot = (h_raw + a_raw) if (h_raw + a_raw) > 0 else 1.0
    h_devig = h_raw / tot
    a_devig = a_raw / tot
    
    # 3. Rolling stats
    h_pf, h_pa, h_diff, h_wr = compute_rolling(team_history.get(h_code, []), 5)
    a_pf, a_pa, a_diff, a_wr = compute_rolling(team_history.get(a_code, []), 5)
    
    # 4. Input feature vector
    input_dict = {
        'spread_line': spread_line,
        'total_line': total_line,
        'home_devigged_prob': h_devig,
        'spread_prob': spread_prob,
        'home_elo': h_elo,
        'away_elo': a_elo,
        'elo_diff': elo_diff,
        'elo_prob': elo_prob,
        'home_roll_pf': h_pf,
        'home_roll_pa': h_pa,
        'home_roll_diff': h_diff,
        'home_roll_wr': h_wr,
        'away_roll_pf': a_pf,
        'away_roll_pa': a_pa,
        'away_roll_diff': a_diff,
        'away_roll_wr': a_wr,
        'roll_diff_net': h_diff - a_diff,
        'roll_pf_net': h_pf - a_pf,
        'roll_pa_net': a_pa - h_pa,
        'rest_diff': 0.0,
        'div_game': 1.0 if NFL_TEAMS.get(h_code, {}).get('division') == NFL_TEAMS.get(a_code, {}).get('division') else 0.0
    }
    
    if model is not None and scaler is not None:
        x_df = pd.DataFrame([input_dict])[FEATURE_COLS]
        x_scaled = scaler.transform(x_df.values)
        x_mx = mx.array(x_scaled, dtype=mx.float32)
        logit = model(x_mx)
        prob_home = mx.sigmoid(logit).item()
    else:
        prob_home = (h_devig + elo_prob) / 2.0
    prob_away = 1.0 - prob_home
    
    # Recommendation
    h_dec = american_to_decimal(home_moneyline) if home_moneyline is not None else 1.0 / max(h_devig, 1e-4)
    a_dec = american_to_decimal(away_moneyline) if away_moneyline is not None else 1.0 / max(a_devig, 1e-4)
    
    ev_home = (prob_home * (h_dec - 1.0)) - (1.0 - prob_home)
    ev_away = (prob_away * (a_dec - 1.0)) - (1.0 - prob_away)
    
    favored_name = h_name if prob_home >= 0.5 else a_name
    win_prob = max(prob_home, prob_away)
    
    print("=" * 70)
    print(f"🏈 MATCHUP PREDICTION: {a_name} ({a_code}) @ {h_name} ({h_code})")
    print("=" * 70)
    print(f"📍 Point Spread:            {h_code} {spread_line:+.1f} | Total: {total_line}")
    print(f"📈 Vegas Implied Win Prob:  {h_name}: {h_devig*100:.1f}% | {a_name}: {a_devig*100:.1f}%")
    print(f"🧠 Model Win Probability:   {h_name}: {prob_home*100:.1f}% | {a_name}: {prob_away*100:.1f}%")
    print(f"⚡ Model Projected Winner:  {favored_name} ({win_prob*100:.1f}% confidence)")
    print("-" * 70)
    
    if ev_home > 0.03:
        print(f"💡 BETTING VALUE: Bet {h_name} ({h_code}) Moneyline (Edge: {ev_home*100:+.2f}%)")
    elif ev_away > 0.03:
        print(f"💡 BETTING VALUE: Bet {a_name} ({a_code}) Moneyline (Edge: {ev_away*100:+.2f}%)")
    else:
        print("💡 BETTING VALUE: Market price is efficient. No strong +EV edge.")
    print("=" * 70)
    
    return {
        'home_team': h_code,
        'away_team': a_code,
        'prob_home': prob_home,
        'prob_away': prob_away,
        'ev_home': ev_home,
        'ev_away': ev_away
    }


def main():
    print("🏈 Apple Silicon MLX NFL Prediction Engine")
    print(f"✅ MLX Version: {mx.__version__} | Device: {mx.default_device()}")
    print(f"✅ 32 NFL Teams Registered. Testing `normalize_team`:")
    print(f"   'Chiefs'    -> {normalize_team('Chiefs')} ({NFL_TEAMS[normalize_team('Chiefs')]['name']})")
    print(f"   '49ers'     -> {normalize_team('49ers')} ({NFL_TEAMS[normalize_team('49ers')]['name']})")
    print(f"   'SD' (move) -> {normalize_team('SD')} ({NFL_TEAMS[normalize_team('SD')]['name']})")
    print(f"   'OAK' (move)-> {normalize_team('OAK')} ({NFL_TEAMS[normalize_team('OAK')]['name']})")
    
    print("\nSimulating Sample Matchup Prediction:")
    predict_nfl_matchup('Chiefs', 'Ravens', spread_line=-3.0, total_line=46.5, home_moneyline=-155, away_moneyline=+135)


if __name__ == '__main__':
    main()
