# 🏈 NFL Game Outcome & Spread Prediction using Apple MLX

> **High-Performance NFL Quantitative Analytics & Betting Intelligence powered by Apple Silicon (MLX) and Vegas Closing Market Insights.**

---

## 📌 Overview

Predicting NFL match outcomes and point spreads is one of the most challenging problems in quantitative sports analytics. Modern sports betting markets aggregate massive volumes of public and sharp information, making betting lines (spreads, moneylines, over/under totals) exceptionally efficient baselines.

This project implements an end-to-end Machine Learning pipeline utilizing **[Apple MLX](https://github.com/ml-explore/mlx)** (`mlx.core`, `mlx.nn`, `mlx.optimizers`) designed specifically for Apple Silicon (M-series unified memory architecture) with Metal acceleration. It blends market-implied probabilities, dynamic FiveThirtyEight-style Elo ratings, rolling team efficiency metrics, and deep neural networks to identify positive expected value ($+EV$) betting opportunities.

---

## 🚀 Key Features

1. **Complete 32-Team NFL Registry & Franchise Normalization**:
   - Covers all 32 active NFL franchises across all 8 divisions and both conferences (AFC / NFC).
   - Seamlessly handles historical relocations and naming aliases (`SD` $\rightarrow$ `LAC`, `STL` $\rightarrow$ `LA`, `OAK` $\rightarrow$ `LV`, `WSH` $\rightarrow$ `WAS`).

2. **Automated Betting & Match Data Ingestion**:
   - Ingests historical NFL game logs, scores, and closing betting lines (spreads, moneylines, over/under totals) from the **nflverse** open analytics consortium (1999–present).
   - Robust caching and offline fallback mechanisms.

3. **Betting Mathematics & Advanced Feature Engineering**:
   - **De-vigging Odds**: Strips bookmaker juice (overround) to extract true consensus market win probabilities.
   - **Spread-to-Probability Mapping**: Gaussian cumulative distribution modeling ($\sigma \approx 13.86$) to convert point spreads to win probabilities.
   - **Dynamic Elo Engine**: Historical Elo tracking with Margin of Victory (MOV) multipliers and home-field advantage modeling.
   - **Rolling Performance Metrics**: Zero-leakage exponential and rolling offensive/defensive scoring rates, point differentials, and rest differentials.

4. **Deep Neural Network in Apple MLX**:
   - Custom Deep Multi-Layer Perceptron (MLP) architecture using `mlx.nn.Module`, `LayerNorm`, `GELU` activations, and `Dropout`.
   - Hardware-accelerated GPU training on Apple Silicon via `nn.value_and_grad` and `optim.AdamW`.

5. **Chronological Backtesting & +EV Simulation**:
   - Strict walk-forward / chronological train-validation-test split (Train: 1999–2021, Val: 2022–2023, Test: 2024–2026) with zero lookahead bias.
   - Out-of-sample evaluation vs. Vegas closing lines and Elo baselines (Accuracy, ROC AUC, Brier Score, Log Loss).
   - Real-money simulation using Flat Staking and Fractional Kelly Criterion bankroll growth tracking.

6. **Interactive All-Teams Matchup Predictor**:
   - Real-time prediction tool (`predict_nfl_matchup`) to evaluate any matchup between any two NFL teams given current lines and moneylines.

---

## 📁 Repository Structure

```text
├── main.py                         # Standalone pipeline and CLI predictor
├── nfl_prediction_mlx.py           # Full MLX pipeline script with Jupyter cell markers (# %%)
├── build_notebook.py               # Script to build the executable Jupyter notebook
├── notebooks/
│   └── nfl_prediction_mlx.ipynb    # Rich interactive Jupyter Notebook with charts & outputs
├── data/
│   └── games.csv                   # Historical NFL match & betting dataset (nflverse)
├── README.md                       # Project documentation
└── LICENSE                         # License file
```

---

## ⚙️ Installation & Requirements

### System Requirements
- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4 series recommended for MLX Metal acceleration)
- **Python**: Python 3.10+ (tested on Python 3.12)

### Dependencies
Install the required Python packages:

```bash
pip install mlx numpy pandas scipy scikit-learn matplotlib seaborn certifi nbformat
```

---

## 🚦 Quickstart & Usage

### 1. Run the Prediction Pipeline via CLI
Execute `main.py` to test team normalization, inspect MLX device status, and run sample matchup predictions:

```bash
python main.py
```

### 2. Run the Full MLX Training Script
Execute `nfl_prediction_mlx.py` for end-to-end data processing, feature engineering, model training, evaluation, and +EV betting backtests:

```bash
python nfl_prediction_mlx.py
```

### 3. Open the Interactive Jupyter Notebook
Open the notebook in JupyterLab, VS Code, or PyCharm:

```bash
jupyter lab notebooks/nfl_prediction_mlx.ipynb
```

*(Optional)* Rebuild the notebook programmatically at any time:
```bash
python build_notebook.py
```

---

## 💡 Programmatic Usage: Matchup Predictor

You can analyze any matchup by calling `predict_nfl_matchup()` in Python:

```python
from main import predict_nfl_matchup

# Example: Chiefs vs. Ravens
predict_nfl_matchup(
    home_team='Chiefs',
    away_team='Ravens',
    spread_line=-3.0,
    total_line=46.5,
    home_moneyline=-155,
    away_moneyline=+135
)
```

**Sample Output:**
```text
======================================================================
🏈 MATCHUP PREDICTION: Baltimore Ravens (BAL) @ Kansas City Chiefs (KC)
======================================================================
📍 Point Spread:            KC -3.0 | Total: 46.5
📈 Vegas Implied Win Prob:  Kansas City Chiefs: 58.2% | Baltimore Ravens: 41.8%
🧠 Model Win Probability:   Kansas City Chiefs: 61.4% | Baltimore Ravens: 38.6%
⚡ Model Projected Winner:  Kansas City Chiefs (61.4% confidence)
----------------------------------------------------------------------
💡 BETTING VALUE: Market price is efficient. No strong +EV edge.
======================================================================
```

---

## 🧠 Methodology & Machine Learning Architecture

```
                               ┌────────────────────────┐
                               │  nflverse Data Ingest   │
                               │  (Games, Odds, Lines)  │
                               └───────────┬────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  Feature Engineering   │
                               │  - De-vigged Odds      │
                               │  - Gaussian Spread CDF │
                               │  - Dynamic Elo Tracker │
                               │  - Rolling Stats (5G)  │
                               └───────────┬────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
             ┌──────────▼──────────┐               ┌──────────▼───��──────┐
             │ Train / Val Split   │               │ Out-of-Sample Test  │
             │ (1999 - 2023)       │               │ (2024 - 2026)       │
             └──────────┬──────────┘               └──────────┬──────────┘
                        │                                     │
             ┌──────────▼──────────┐                          │
             │ Apple MLX MLP Model │                          │
             │ Metal GPU Training  │                          │
             └──────────┬──────────┘                          │
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  Model Evaluation &    │
                               │  +EV Betting Backtest  │
                               └────────────────────────┘
```

### Feature Vector Composition
The model consumes a 21-dimensional normalized feature vector for each game:
- **Market Features**: `spread_line`, `total_line`, `home_devigged_prob`, `spread_prob`
- **Elo Features**: `home_elo`, `away_elo`, `elo_diff`, `elo_prob`
- **Rolling Form**: `home_roll_pf`, `home_roll_pa`, `home_roll_diff`, `home_roll_wr`, `away_roll_pf`, `away_roll_pa`, `away_roll_diff`, `away_roll_wr`
- **Net Differentials & Context**: `roll_diff_net`, `roll_pf_net`, `roll_pa_net`, `rest_diff`, `div_game`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
