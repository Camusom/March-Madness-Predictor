# March-Madness-Predictor
An XGBoost model that predicts the margin of victory for NCAA Tournament  matchups using publicly available efficiency data. Finished 68.3%  directionally accurate across 63 March Madness 2026 games, hitting 78%  in the Round of 64.

## How It Works

The model is trained on every regular season game, validated on conference 
tournaments, and evaluated on March Madness matchups. For each game, it 
predicts the margin of victory (MOV) from the perspective of team A — 
a positive value means team A wins, negative means team B wins.

### Features (all computed as team A minus team B differentials)

OffRtg_diff (Offensive rating differential)
DefRtg_diff (Defensive rating differential) 
ThreePA_diff (Three point attempts differential) 
TOV_diff (Turnover differential) 
ORB_diff (Offensive rebound differential)
DRB_diff (Defensive rebound differential)
RecentForm_diff (Rolling 5-game MOV average differential)
SOS_diff (Strength of schedule differential)
Pace_diff (Pace of play differential)

SOS ended up being the strongest single predictor. When OffRtg and DefRtg 
both strongly favor one team by roughly equal magnitudes, the model's 
margin predictions were sometimes accurate to within 1-2 points.

---

## Data Sources

- **Game logs / efficiency stats** — [Stathead Basketball](https://stathead.com/basketball/)
- **Strength of schedule + Pace** — [TeamRankings](https://www.teamrankings.com/)

Data is copy-pasted from Stathead into CSV format and scraped from 
TeamRankings, then merged and cleaned via the data pipeline.

---

## Project Files
data_cleaning.py # Full data pipeline (clean, merge, roll, split)
modeling.py # XGBoost training with Optuna optimization, generate predictions here
data1.csv # Raw Stathead export
teamrankings_SOS_raw.csv # Raw SOS from TeamRankings
teamrankings_Pace_raw.csv # Raw Pace from TeamRankings
stathead_cleaned.csv # Cleaned game logs
supplements_cleaned.csv # Cleaned SOS + Pace
games_rolling.csv # Rolling features per team
train.csv # Regular season training data
val.csv # Conference tournament validation





---

## Setup

```bash
git clone https://github.com/Camusom/march-madness-model
cd march-madness-model
pip install -r requirements.txt

Requirements:
text
pandas
numpy
xgboost
optuna
scikit-learn

Running The Pipeline
Step 1 — Collect Data
Download game logs from Stathead (all regular season + tournament games)
Save as data/data1.csv
Download SOS and Pace CSVs from TeamRankings
Save as data/teamrankings_SOS_raw.csv and data/teamrankings_Pace_raw.csv

Step 2 — Clean and Build Dataset
bash
python data_cleaning.py
This runs the full pipeline: cleans Stathead data, merges TeamRankings
supplements, computes rolling features, builds matchup rows, and outputs
train/val splits.

Step 3 — Train Model
bash
python modeling.py
Trains XGBoost with Bayesian hyperparameter optimization via Optuna.
Outputs the trained model and validation metrics.

Step 4 — Predict a Matchup
bash
python predict.py
Enter any two team names and the script generates a predicted margin of
victory with key matchup factor breakdowns.

Model Performance
Metric	Value
Validation MAE	~8.8 points
Tournament directional accuracy	68.3% (63 games)
Round of 64 accuracy	78.1% (32 games)

Notable calls:
Texas over Gonzaga — model: Texas -9.8, Vegas: Gonzaga -6.5, Actual: Texas -6
Illinois over Houston — model: Illinois -8.2, Vegas: Houston -2.5, Actual: Illinois -10
Iowa over Nebraska — model: Iowa -6.2, Vegas: Nebraska -1.5, Actual: Iowa -6 (exact)
Michigan over Tennessee — model: Michigan -12.1, Vegas: Michigan -7.5, Actual: Michigan -33
Known limitation: The model structurally undervalues teams with elite
coaching systems and tournament experience (see: UConn, 4/4 beating model
favorites all tournament). These factors don't appear in box score data.

What's Next
Adapting this for NBA playoffs with additional features:

Coaching tenure / playoff record
Player availability / injury flags
Rest day differentials
Series momentum
Home court advantage
Built With
Python · XGBoost · Optuna · Pandas · NumPy · Scikit-learn

Data is manually collected from public sources. This project is for
educational purposes only.

