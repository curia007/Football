"""
NFL Game Outcome & Spread Prediction using Apple MLX Framework.
Complete pipeline with 32-team normalization, betting mathematics, MLX Neural Network training, and matchup prediction.
"""

import os
import time
import ssl
import urllib.request
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


def prob_to_american(p: float) -> str:
    """Converts a win probability (0.0 to 1.0) to American moneyline string format (e.g. -150 or +130)."""
    if p is None or pd.isna(p) or p <= 0.0 or p >= 1.0:
        return "N/A"
    if p >= 0.5:
        ml = -int(round((p / (1.0 - p)) * 100))
        return f"{ml:+d}"
    else:
        ml = int(round(((1.0 - p) / p) * 100))
        return f"{ml:+d}"


def compute_rolling(history, n=5):
    """Computes rolling offensive/defensive metrics from historical games."""
    if not history:
        return 21.5, 21.5, 0.0, 0.5
    recent = history[-n:]
    pf = float(np.mean([g['pf'] for g in recent]))
    pa = float(np.mean([g['pa'] for g in recent]))
    diff = pf - pa
    wr = float(np.mean([g['win'] for g in recent]))
    return pf, pa, diff, wr


# ==========================================
# 3. Apple MLX Neural Network Model
# ==========================================
class MLXNFLPredictor(nn.Module):
    """
    Deep Multi-Layer Perceptron in Apple MLX for NFL Win Probability Prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 16)
        self.out = nn.Linear(16, 1)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dropout(self.act(self.ln1(self.fc1(x))))
        x = self.dropout(self.act(self.ln2(self.fc2(x))))
        x = self.act(self.fc3(x))
        return self.out(x)


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
INITIAL_ELO = 1500.0
K_BASE = 20.0
SPREAD_SIGMA = 13.86
EPOCHS = 40
BATCH_SIZE = 64

# Default global model instance
model = MLXNFLPredictor(input_dim=len(FEATURE_COLS), hidden_dim=64, dropout_rate=0.2)
scaler = None


# ==========================================
# 4. Training Pipeline & Feature Extraction
# ==========================================
def load_historical_games(data_path=None):
    """Loads historical NFL games from local data or nflverse."""
    candidate_paths = [
        data_path,
        os.path.join("data", "games.csv"),
        os.path.join("..", "data", "games.csv"),
        os.path.join("notebooks", "games.csv"),
        "games.csv"
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return pd.read_csv(str(path))
    
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=20) as r:
        df = pd.read_csv(r)
        try:
            os.makedirs("data", exist_ok=True)
            df.to_csv(os.path.join("data", "games.csv"), index=False)
        except Exception:
            pass
        return df


def train_mlx_pipeline(games_df=None, data_path=None, epochs=40, batch_size=64, learning_rate=1.5e-3, verbose=True):
    """
    Trains the Apple Silicon MLX Neural Network on historical games and computes up-to-date Elo ratings.
    """
    if games_df is None:
        games_df = load_historical_games(data_path)
        
    df = games_df.copy()
    df['home_team'] = df['home_team'].apply(normalize_team)
    df['away_team'] = df['away_team'].apply(normalize_team)
    
    # Filter completed games
    completed = df[df['home_score'].notna() & df['away_score'].notna()].copy()
    completed['gameday'] = pd.to_datetime(completed['gameday'])
    completed = completed.sort_values(['gameday', 'game_id']).reset_index(drop=True)
    
    # Target and market odds
    completed['home_win'] = (completed['home_score'] > completed['away_score']).astype(np.float32)
    
    # Impute missing spread / total
    completed['spread_line'] = completed['spread_line'].fillna(0.0)
    completed['total_line'] = completed['total_line'].fillna(44.5)
    
    # Probability transformations
    completed['spread_prob'] = norm.cdf(completed['spread_line'] / SPREAD_SIGMA)
    completed['home_ml_prob'] = completed['home_moneyline'].apply(american_to_implied_prob) if 'home_moneyline' in completed.columns else np.nan
    completed['away_ml_prob'] = completed['away_moneyline'].apply(american_to_implied_prob) if 'away_moneyline' in completed.columns else np.nan
    
    completed['home_raw_prob'] = completed['home_ml_prob'].fillna(completed['spread_prob'])
    completed['away_raw_prob'] = completed['away_ml_prob'].fillna(1.0 - completed['spread_prob'])
    sum_raw = completed['home_raw_prob'] + completed['away_raw_prob']
    completed['home_devigged_prob'] = completed['home_raw_prob'] / sum_raw.replace(0, 1.0)
    
    # Dynamic Elo and Rolling stats tracking
    team_history = {team: [] for team in NFL_TEAMS}
    elo_ratings = {team: INITIAL_ELO for team in NFL_TEAMS}
    
    home_elo_list, away_elo_list = [], []
    h_pf_list, h_pa_list, h_diff_list, h_wr_list = [], [], [], []
    a_pf_list, a_pa_list, a_diff_list, a_wr_list = [], [], [], []
    
    for idx, row in completed.iterrows():
        ht = row['home_team']
        at = row['away_team']
        hs = row['home_score']
        as_ = row['away_score']
        
        h_elo = elo_ratings.get(ht, INITIAL_ELO)
        a_elo = elo_ratings.get(at, INITIAL_ELO)
        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)
        
        h_pf, h_pa, h_diff, h_wr = compute_rolling(team_history.get(ht, []), 5)
        a_pf, a_pa, a_diff, a_wr = compute_rolling(team_history.get(at, []), 5)
        
        h_pf_list.append(h_pf)
        h_pa_list.append(h_pa)
        h_diff_list.append(h_diff)
        h_wr_list.append(h_wr)
        
        a_pf_list.append(a_pf)
        a_pa_list.append(a_pa)
        a_diff_list.append(a_diff)
        a_wr_list.append(a_wr)
        
        # Update Elo post game
        h_adv = HOME_ADVANTAGE_ELO if row.get('location', 'Home') == 'Home' else 0.0
        e_diff = (h_elo + h_adv) - a_elo
        e_home = 1.0 / (1.0 + 10.0 ** (-e_diff / 400.0))
        s_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        mov = abs(hs - as_)
        mov_mult = np.log(max(mov, 1) + 1.0) * (2.2 / ((e_diff if s_home == 1.0 else -e_diff) * 0.001 + 2.2))
        k = K_BASE * mov_mult
        
        elo_ratings[ht] = h_elo + k * (s_home - e_home)
        elo_ratings[at] = a_elo + k * ((1.0 - s_home) - (1.0 - e_home))
        
        if ht not in team_history: team_history[ht] = []
        if at not in team_history: team_history[at] = []
        team_history[ht].append({'pf': hs, 'pa': as_, 'win': 1.0 if hs > as_ else 0.0})
        team_history[at].append({'pf': as_, 'pa': hs, 'win': 1.0 if as_ > hs else 0.0})
        
    completed['home_elo'] = home_elo_list
    completed['away_elo'] = away_elo_list
    completed['elo_diff'] = (completed['home_elo'] + HOME_ADVANTAGE_ELO) - completed['away_elo']
    completed['elo_prob'] = 1.0 / (1.0 + 10.0 ** (-completed['elo_diff'] / 400.0))
    
    completed['home_roll_pf'] = h_pf_list
    completed['home_roll_pa'] = h_pa_list
    completed['home_roll_diff'] = h_diff_list
    completed['home_roll_wr'] = h_wr_list
    
    completed['away_roll_pf'] = a_pf_list
    completed['away_roll_pa'] = a_pa_list
    completed['away_roll_diff'] = a_diff_list
    completed['away_roll_wr'] = a_wr_list
    
    completed['roll_diff_net'] = completed['home_roll_diff'] - completed['away_roll_diff']
    completed['roll_pf_net'] = completed['home_roll_pf'] - completed['away_roll_pf']
    completed['roll_pa_net'] = completed['away_roll_pa'] - completed['home_roll_pa']
    completed['rest_diff'] = (completed['home_rest'].fillna(7.0) - completed['away_rest'].fillna(7.0)) if 'home_rest' in completed.columns else 0.0
    completed['div_game'] = completed['div_game'].fillna(0.0).astype(np.float32) if 'div_game' in completed.columns else 0.0
    
    # Train / Val split
    train_mask = completed['season'] < 2024
    if train_mask.sum() == 0:
        train_mask = np.ones(len(completed), dtype=bool)
        
    train_df = completed[train_mask]
    val_df = completed[~train_mask] if (~train_mask).sum() > 0 else train_df
    
    trained_scaler = StandardScaler()
    X_train = trained_scaler.fit_transform(train_df[FEATURE_COLS].values)
    y_train = train_df['home_win'].values.reshape(-1, 1).astype(np.float32)
    
    X_val_mx = mx.array(trained_scaler.transform(val_df[FEATURE_COLS].values), dtype=mx.float32)
    y_val_mx = mx.array(val_df['home_win'].values.reshape(-1, 1).astype(np.float32), dtype=mx.float32)
    
    trained_model = MLXNFLPredictor(input_dim=len(FEATURE_COLS), hidden_dim=64, dropout_rate=0.2)
    mx.eval(trained_model.parameters())
    
    def loss_fn(m, x, y):
        logits = m(x)
        return mx.mean(nn.losses.binary_cross_entropy(logits, y, with_logits=True))
        
    loss_and_grad_fn = nn.value_and_grad(trained_model, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    
    n_train = len(X_train)
    if verbose:
        print(f"🧠 Training Apple MLX Neural Network on {n_train:,} historical games across {len(NFL_TEAMS)} teams...")
    
    start_t = time.time()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = mx.array(X_train[batch_idx], dtype=mx.float32)
            yb = mx.array(y_train[batch_idx], dtype=mx.float32)
            loss, grads = loss_and_grad_fn(trained_model, xb, yb)
            optimizer.update(trained_model, grads)
            mx.eval(trained_model.parameters(), optimizer.state)
            
    val_logits = trained_model(X_val_mx)
    val_preds = (mx.sigmoid(val_logits) > 0.5).astype(mx.float32)
    val_acc = mx.mean(val_preds == y_val_mx).item()
    
    if verbose:
        dur = time.time() - start_t
        print(f"✅ MLX Training Complete ({dur:.2f}s) | Validation Accuracy: {val_acc*100:.2f}%")
        
    return trained_model, trained_scaler, elo_ratings, team_history


# ==========================================
# 4. Pipeline & Matchup Predictor
# ==========================================
def predict_nfl_matchup(home_team, away_team, spread_line=None, total_line=44.5,
                        home_moneyline=None, away_moneyline=None,
                        model=None, scaler=None, elo_ratings=None, team_history=None, verbose=True):
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
    if spread_line is None or pd.isna(spread_line):
        spread_line = float(norm.ppf(np.clip(elo_prob, 0.01, 0.99)) * SPREAD_SIGMA)
    else:
        spread_line = float(spread_line)
        
    if total_line is None or pd.isna(total_line):
        total_line = 44.5
    else:
        total_line = float(total_line)
        
    spread_prob = norm.cdf(spread_line / SPREAD_SIGMA)
    h_raw = american_to_implied_prob(home_moneyline) if (home_moneyline is not None and not pd.isna(home_moneyline)) else spread_prob
    a_raw = american_to_implied_prob(away_moneyline) if (away_moneyline is not None and not pd.isna(away_moneyline)) else (1.0 - spread_prob)
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
        prob_home = float(mx.sigmoid(logit).item())
    else:
        prob_home = float((h_devig + elo_prob) / 2.0)
    prob_away = 1.0 - prob_home
    
    # Recommendation
    h_dec = american_to_decimal(home_moneyline) if (home_moneyline is not None and not pd.isna(home_moneyline)) else 1.0 / max(h_devig, 1e-4)
    a_dec = american_to_decimal(away_moneyline) if (away_moneyline is not None and not pd.isna(away_moneyline)) else 1.0 / max(a_devig, 1e-4)
    
    ev_home = (prob_home * (h_dec - 1.0)) - (1.0 - prob_home)
    ev_away = (prob_away * (a_dec - 1.0)) - (1.0 - prob_away)
    
    favored_code = h_code if prob_home >= 0.5 else a_code
    favored_name = h_name if prob_home >= 0.5 else a_name
    win_prob = max(prob_home, prob_away)
    
    # Formatted spread representation (e.g. KC -3.5 or GB -6.5 or EVEN)
    # Note: In NFL convention, negative spread for home team means home is favored by that amount.
    if spread_line < 0:
        formatted_spread = f"{h_code} {spread_line:+.1f}"
    elif spread_line > 0:
        formatted_spread = f"{a_code} {-spread_line:+.1f}"
    else:
        formatted_spread = "EVEN (0.0)"
        
    # Projected winner spread (from winner's perspective)
    if favored_code == h_code:
        winner_spread = spread_line
    else:
        winner_spread = -spread_line
    
    if ev_home > 0.03:
        rec = f"Bet {h_name} ({h_code}) Moneyline (Edge: {ev_home*100:+.2f}%)"
    elif ev_away > 0.03:
        rec = f"Bet {a_name} ({a_code}) Moneyline (Edge: {ev_away*100:+.2f}%)"
    else:
        rec = "Market price is efficient. No strong +EV edge."
        
    if verbose:
        print("=" * 70)
        print(f"🏈 MATCHUP PREDICTION: {a_name} ({a_code}) @ {h_name} ({h_code})")
        print("=" * 70)
        print(f"📍 Point Spread:            {formatted_spread} (Home Spread: {spread_line:+.1f}) | Total: {total_line}")
        print(f"📈 Vegas Implied Win Prob:  {h_name}: {h_devig*100:.1f}% | {a_name}: {a_devig*100:.1f}%")
        print(f"🧠 Model Win Probability:   {h_name}: {prob_home*100:.1f}% | {a_name}: {prob_away*100:.1f}%")
        print(f"⚡ Model Projected Winner:  {favored_name} ({win_prob*100:.1f}% possible win)")
        print(f"🎯 Spread Projection:       {formatted_spread}")
        print("-" * 70)
        print(f"💡 BETTING VALUE: {rec}")
        print("=" * 70)
    
    h_ml_str = f"{int(home_moneyline):+d}" if (home_moneyline is not None and not pd.isna(home_moneyline)) else prob_to_american(prob_home)
    a_ml_str = f"{int(away_moneyline):+d}" if (away_moneyline is not None and not pd.isna(away_moneyline)) else prob_to_american(prob_away)

    return {
        'home_team': h_code,
        'home_name': h_name,
        'away_team': a_code,
        'away_name': a_name,
        'possible_win': favored_code,
        'possible_win_name': favored_name,
        'possible_win_pct': f"{win_prob*100:.1f}%",
        'predicted_winner': favored_code,
        'predicted_winner_name': favored_name,
        'win_probability': round(win_prob, 4),
        'confidence_pct': f"{win_prob*100:.1f}%",
        'prob_home': round(prob_home, 4),
        'prob_away': round(prob_away, 4),
        'spread': round(spread_line, 1),
        'spread_line': round(spread_line, 1),
        'formatted_spread': formatted_spread,
        'winner_spread': round(winner_spread, 1),
        'total_line': round(total_line, 1),
        'home_moneyline': h_ml_str,
        'away_moneyline': a_ml_str,
        'home_elo': round(h_elo, 1),
        'away_elo': round(a_elo, 1),
        'elo_diff': round(elo_diff, 1),
        'ev_home': round(ev_home, 4),
        'ev_away': round(ev_away, 4),
        'recommendation': rec
    }


def predict_matchups(matchups_df, model=None, scaler=None, elo_ratings=None, team_history=None, verbose=True):
    """
    Predicts the winner for each matchup in a DataFrame using MLX model logic.
    """
    if matchups_df.empty:
        print("⚠️ No matchups provided for prediction.")
        return matchups_df
        
    results = []
    enriched_df = matchups_df.copy()
    
    for idx, row in enriched_df.iterrows():
        ht = row['home_team']
        at = row['away_team']
        sp = row.get('spread_line', None)
        tot = row.get('total_line', None)
        h_ml = row.get('home_moneyline', None)
        a_ml = row.get('away_moneyline', None)
        
        pred = predict_nfl_matchup(
            home_team=ht,
            away_team=at,
            spread_line=sp,
            total_line=tot,
            home_moneyline=h_ml,
            away_moneyline=a_ml,
            model=model,
            scaler=scaler,
            elo_ratings=elo_ratings,
            team_history=team_history,
            verbose=False
        )
        results.append(pred)
        
    preds_df = pd.DataFrame(results)
    
    # Assign prediction fields to enriched_df
    enriched_df['possible_win'] = preds_df['possible_win'].values
    enriched_df['possible_win_name'] = preds_df['possible_win_name'].values
    enriched_df['possible_win_pct'] = preds_df['possible_win_pct'].values
    enriched_df['predicted_winner'] = preds_df['predicted_winner'].values
    enriched_df['predicted_winner_name'] = preds_df['predicted_winner_name'].values
    enriched_df['win_probability'] = preds_df['win_probability'].values
    enriched_df['confidence_pct'] = preds_df['confidence_pct'].values
    enriched_df['home_win_prob'] = preds_df['prob_home'].values
    enriched_df['away_win_prob'] = preds_df['prob_away'].values
    enriched_df['spread'] = preds_df['spread'].values
    enriched_df['spread_line'] = preds_df['spread_line'].values
    enriched_df['formatted_spread'] = preds_df['formatted_spread'].values
    enriched_df['winner_spread'] = preds_df['winner_spread'].values
    enriched_df['projected_spread'] = preds_df['spread_line'].values
    enriched_df['total_line'] = preds_df['total_line'].values
    enriched_df['home_moneyline'] = preds_df['home_moneyline'].values
    enriched_df['away_moneyline'] = preds_df['away_moneyline'].values
    enriched_df['home_elo'] = preds_df['home_elo'].values
    enriched_df['away_elo'] = preds_df['away_elo'].values
    enriched_df['elo_diff'] = preds_df['elo_diff'].values
    enriched_df['recommendation'] = preds_df['recommendation'].values
    
    # Clean any remaining NaNs across all columns for proper CSV and table presentation
    for col in enriched_df.columns:
        if enriched_df[col].isna().any():
            enriched_df[col] = enriched_df[col].fillna('N/A')
    
    if verbose:
        print("\n" + "=" * 105)
        print("🏈 MLX NFL MATCHUP WINNER & SPREAD PREDICTIONS")
        print("=" * 105)
        display_cols = [
            'gameday', 'matchup_short', 'possible_win', 'possible_win_name',
            'possible_win_pct', 'formatted_spread', 'home_win_prob', 'away_win_prob'
        ]
        available_cols = [c for c in display_cols if c in enriched_df.columns]
        
        # Format probabilities for clean table display
        formatted_table = enriched_df[available_cols].fillna("N/A").copy()
        if 'home_win_prob' in formatted_table.columns:
            formatted_table['home_win_prob'] = formatted_table['home_win_prob'].apply(lambda x: f"{x*100:.1f}%")
        if 'away_win_prob' in formatted_table.columns:
            formatted_table['away_win_prob'] = formatted_table['away_win_prob'].apply(lambda x: f"{x*100:.1f}%")
            
        print(formatted_table.to_string(index=False))
        print("=" * 105)
        
    return enriched_df


def predict_next_matchups(matchups_df=None, data_path=None, as_of_date=None, season=None, week=None,
                          all_upcoming=False, output_csv=None, verbose=True):
    """
    End-to-end pipeline: Ingests upcoming matchups from get_next_matchups, trains MLX model,
    and predicts the winner for each matchup.
    """
    from nfl import get_next_matchups

    # 1. Obtain matchups if not provided
    if matchups_df is None:
        games_df = get_next_matchups.load_games_dataset(data_path)
        matchups_df = get_next_matchups.get_next_matchups(
            df=games_df,
            as_of_date=as_of_date,
            season=season,
            week=week,
            all_upcoming=all_upcoming
        )
    else:
        games_df = None
        
    if matchups_df.empty:
        print("⚠️ No upcoming matchups found to predict.")
        return matchups_df
        
    # 2. Train MLX neural network and compute current Elo / histories
    trained_model, trained_scaler, elo_ratings, team_history = train_mlx_pipeline(
        games_df=games_df,
        data_path=data_path,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=verbose
    )
    
    # 3. Predict all matchups
    predicted_df = predict_matchups(
        matchups_df=matchups_df,
        model=trained_model,
        scaler=trained_scaler,
        elo_ratings=elo_ratings,
        team_history=team_history,
        verbose=verbose
    )
    
    # 4. Save to output CSV if specified
    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        predicted_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"💾 Saved {len(predicted_df)} matchup predictions to: {output_csv}")
            
    return predicted_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NFL Game Prediction using Apple MLX Framework")
    parser.add_argument("--predict-next", action="store_true", help="Predict all next upcoming NFL matchups")
    parser.add_argument("--season", type=int, default=None, help="NFL season (e.g. 2026)")
    parser.add_argument("--week", type=int, default=None, help="NFL week (e.g. 3)")
    parser.add_argument("--all-upcoming", "-a", action="store_true", help="Predict all remaining upcoming games")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path for predictions")
    parser.add_argument("--home", default=None, help="Home team for single matchup prediction")
    parser.add_argument("--away", default=None, help="Away team for single matchup prediction")
    parser.add_argument("--spread", type=float, default=None, help="Spread line for single matchup")
    parser.add_argument("--total", type=float, default=44.5, help="Total line for single matchup")
    parser.add_argument("--home-ml", type=float, default=None, help="Home moneyline")
    parser.add_argument("--away-ml", type=float, default=None, help="Away moneyline")
    
    args = parser.parse_args()
    
    print("🏈 Apple Silicon MLX NFL Prediction Engine")
    print(f"✅ MLX Version: {mx.__version__} | Device: {mx.default_device()}")
    
    if args.home and args.away:
        print(f"\nPredicting Custom Matchup: {args.away} @ {args.home}")
        trained_model, trained_scaler, elo_ratings, team_history = train_mlx_pipeline(verbose=False)
        predict_nfl_matchup(
            home_team=args.home,
            away_team=args.away,
            spread_line=args.spread,
            total_line=args.total,
            home_moneyline=args.home_ml,
            away_moneyline=args.away_ml,
            model=trained_model,
            scaler=trained_scaler,
            elo_ratings=elo_ratings,
            team_history=team_history,
            verbose=True
        )
    else:
        # Default: Run full pipeline on the next NFL matchups
        output_path = args.output if args.output else os.path.join("data", "next_matchups_predictions.csv")
        predict_next_matchups(
            season=args.season,
            week=args.week,
            all_upcoming=args.all_upcoming,
            output_csv=output_path,
            verbose=True
        )


if __name__ == '__main__':
    main()
