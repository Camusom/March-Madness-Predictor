import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

SAVE_DIR = r"C:\Users\Massimo Camuso\Desktop\march madness model"
#'SOS_diff',
FEATURE_COLS = [
    'OffRtg_diff',
    'DefRtg_diff',
    'ThreePA_diff',
    'TOV_diff',
    'ORB_diff',
    'DRB_diff',
    'RecentForm_diff',
    'SOS_diff',
    'Pace_diff',
    'location'
]

TARGET_COL = 'MOV'

NAME_MAP = {
    # A
    'Abilene Christian'         : 'Abl Christian',
    'Alabama State'             : 'Alabama St',
    'Albany (NY)'               : 'Albany',
    'Alcorn State'              : 'Alcorn St',
    'Appalachian State'         : 'App State',
    'Arizona State'             : 'Arizona St',
    'Arkansas State'            : 'Arkansas St',
    'Arkansas–Pine Bluff'       : 'AR-Pine Bluff',
    'Arkansas-Pine Bluff'       : 'AR-Pine Bluff',

    # B
    'Ball State'                : 'Ball St',
    'Bethune-Cookman'           : 'Bethune',
    'Boise State'               : 'Boise St',
    'Boston University'         : 'Boston U',
    'Bowling Green State'       : 'Bowling Green',
    'Brigham Young'             : 'BYU',

    # C
    'California Baptist'        : 'Cal Baptist',
    'Cal State Bakersfield'     : 'CS Bakersfield',
    'Cal State Fullerton'       : 'CS Fullerton',
    'Cal State Northridge'      : 'CS Northridge',
    'Central Arkansas'          : 'C Arkansas',
    'Central Connecticut State' : 'C Connecticut',
    'Central Michigan'          : 'C Michigan',
    'Charleston Southern'       : 'Charleston So',
    'Cleveland State'           : 'Cleveland St',
    'Coastal Carolina'          : 'Coastal Car',
    'Colorado State'            : 'Colorado St',
    'Connecticut'               : 'UConn',
    'Chicago State'             : 'Chicago St',
    'College of Charleston'     : 'Charleston',
    'Coppin State'              : 'Coppin St',

    # D
    'Delaware State'            : 'Delaware St',

    # E
    'East Carolina'             : 'E Carolina',
    'East Tennessee State'      : 'E Tennessee St',
    'East Texas A&M'            : 'E Texas A&M',
    'Eastern Illinois'          : 'E Illinois',       # check txt if still fails
    'Eastern Kentucky'          : 'E Kentucky',       # check txt if still fails
    'Eastern Michigan'          : 'E Michigan',
    'Eastern Washington'        : 'E Washington',

    # F
    'FDU'                       : 'F Dickinson',
    'Florida Atlantic'          : 'Florida Atlantic',
    'Florida Gulf Coast'        : 'FGCU',
    'Florida International'     : 'Florida Intl',
    'Florida State'             : 'Florida St',
    'Fort Wayne'                : 'Purdue FW',
    'Fresno State'              : 'Fresno St',

    # G
    'Gardner-Webb'              : 'Gardner-Webb',
    'George Mason'              : 'George Mason',
    'George Washington'         : 'G Washington',
    'Georgia Southern'          : 'Georgia So',
    'Georgia State'             : 'Georgia St',
    'Grambling State'           : 'Grambling',
    'Grand Canyon'              : 'Grand Canyon',

    # H
    'Hawaii'                    : "Hawai'i",
    'Houston Baptist'           : 'Hou Christian',
    'Houston Christian'         : 'Hou Christian',

    # I
    'Idaho State'               : 'Idaho St',
    'Illinois–Chicago'          : 'Illinois Chicago',
    'Illinois State'            : 'Illinois St',
    'Indiana State'             : 'Indiana St',
    'Iowa State'                : 'Iowa St',

    # J
    'Jackson State'             : 'Jackson St',
    'Jacksonville State'        : 'Jacksonville St',
    'James Madison'             : 'J Madison',

    # K
    'Kansas State'              : 'Kansas St',
    'Kennesaw State'            : 'Kennesaw St',
    'Kent State'                : 'Kent St',

    # L
    'Le Moyne'                  : 'Le Moyne',
    'LIU'                       : 'LIU',
    'Long Beach State'          : 'Long Beach St',
    'Louisiana Monroe'          : 'UL Monroe',
    'Louisiana State'           : 'LSU',
    'Louisiana Tech'            : 'Louisiana Tech',
    'Louisiana–Lafayette'       : 'Louisiana',
    'Loyola (MD)'               : 'Loyola MD',
    'Loyola Marymount'          : 'Loyola Mymt',
    'Loyola Chicago'            : 'Loyola Chi',

    # M
    'Maryland Eastern Shore'    : 'Maryland ES',
    'Massachusetts'             : 'UMass',
    'McNeese State'             : 'McNeese',
    'Miami (FL)'                : 'Miami',
    'Miami (OH)'                : 'Miami OH',
    'Michigan State'            : 'Michigan St',
    'Middle Tennessee'          : 'Middle Tenn',
    'Mississippi State'         : 'Mississippi St',
    'Mississippi Valley State'  : 'Miss Valley St',
    'Missouri State'            : 'Missouri St',
    'Montana State'             : 'Montana St',
    'Morehead State'            : 'Morehead St',
    'Morgan State'              : 'Morgan St',
    'Mount St. Mary\'s'         : 'Mt St Mary\'s',
    'Murray State'              : 'Murray St',

    # N
    'Nebraska Omaha'            : 'Omaha',
    'Nevada–Las Vegas'          : 'UNLV',
    'Nevada-Las Vegas'          : 'UNLV',
    'New Mexico State'          : 'New Mexico St',
    'Nicholls State'            : 'Nicholls',
    'Norfolk State'             : 'Norfolk St',
    'North Alabama'             : 'N Alabama',
    'North Carolina A&T'        : 'NC A&T',
    'North Carolina Central'    : 'NC Central',
    'North Carolina State'      : 'NC State',
    'North Dakota State'        : 'N Dakota St',
    'North Florida'             : 'N Florida',
    'North Texas'               : 'N Texas',
    'Northern Arizona'          : 'N Arizona',
    'Northern Colorado'         : 'N Colorado',
    'Northern Illinois'         : 'N Illinois',
    'Northern Iowa'             : 'N Iowa',
    'Northern Kentucky'         : 'N Kentucky',
    'Northwestern State'        : 'NW State',

    # O
    'Ohio State'                : 'Ohio St',
    'Oklahoma State'            : 'Oklahoma St',
    'Ole Miss'                  : 'Mississippi',
    'Oregon State'              : 'Oregon St',

    # P
    'Penn State'                : 'Penn St',
    'Portland State'            : 'Portland St',
    'Prairie View A&M'          : 'Prairie View',
    'Purdue Fort Wayne'         : 'Purdue FW',
    'Pennsylvania'              : 'Penn',


    # Q
    'Queens (NC)'               : 'Queens',

    # S
    'Sacramento State'          : 'Sacramento St',
    'Saint Francis (PA)'        : 'St Francis PA',
    'Saint Joseph\'s'           : "Saint Joseph's",
    'St. Mary\'s (CA)'          : 'Saint Mary\'s',

    'Sam Houston State'         : 'Sam Houston',
    'San Diego State'           : 'San Diego St',
    'San José State'            : 'San Jose St',
    'San Jose State'            : 'San Jose St',
    'Savannah State'            : 'Savannah St',
    'Seattle University'        : 'Seattle U',
    'SIU Edwardsville'          : 'SIU Edward',
    'South Alabama'             : 'S Alabama',
    'South Carolina State'      : 'S Carolina St',
    'South Dakota State'        : 'S Dakota St',
    'South Florida'             : 'S Florida',
    'Southeast Missouri State'  : 'SE Missouri St',
    'Southeastern Louisiana'    : 'SE Louisiana',
    'Southern Illinois'         : 'Southern',
    'Southern Indiana'          : 'S Indiana',
    'Southern Mississippi'      : 'Southern Miss',
    'Southern Utah'             : 'S Utah',
    'St. Bonaventure'           : 'St Bonaventure',
    'St. John\'s (NY)'          : 'St John\'s',
    'St. Thomas'                : 'St Thomas',
    'Stephen F. Austin'         : 'SF Austin',
    'Southern California'       : 'USC',


    # T
    'Tarleton State'            : 'Tarleton St',
    'Tennessee State'           : 'Tennessee St',
    'Tennessee Tech'            : 'Tenn Tech',
    'Texas A&M–Corpus Christi'  : 'Texas A&M-CC',
    'Texas Southern'            : 'Texas So',
    'Texas State'               : 'Texas St',
    'Texas–Rio Grande Valley'   : 'UT Rio Grande',

    # U
    'UC San Diego'              : 'UCSD',
    'UC Santa Barbara'          : 'UCSB',
    'UNC Asheville'             : 'NC Asheville',
    'UNC Greensboro'            : 'NC Greensboro',
    'UNC Wilmington'            : 'NC Wilmington',
    'USC Upstate'               : 'SC Upstate',
    'UT Arlington'              : 'UT Arlington',
    'UT Martin'                 : 'UT Martin',
    'UTSA'                      : 'UTSA',
    'Utah State'                : 'Utah St',

    # V
    'Virginia Tech'             : 'Virginia Tech',

    # W
    'Washington State'          : 'Washington St',
    'Weber State'               : 'Weber St',
    'West Georgia'              : 'W Georgia',
    'Western Carolina'          : 'W Carolina',
    'Western Illinois'          : 'W Illinois',
    'Western Kentucky'          : 'W Kentucky',
    'Western Michigan'          : 'W Michigan',
    'Wichita State'             : 'Wichita St',
    'Winston-Salem State'       : 'Winston-Salem St',
    'Wisconsin–Milwaukee'       : 'Milwaukee',
    'Wright State'              : 'Wright St',


    # Y
    'Youngstown State'          : 'Youngstown St',
}

def normalize_stathead_name(name):
    """
    Convert Stathead team name to TeamRankings equivalent
    Falls back to original if no mapping found
    """
    name = str(name).strip()
    return NAME_MAP.get(name, name)


def load_data():
    train = pd.read_csv(os.path.join(SAVE_DIR, 'train.csv'), parse_dates=['date'])
    train = train.sort_values('date').reset_index(drop=True)

    # Val might be empty - handle gracefully
    val_path = os.path.join(SAVE_DIR, 'val.csv')
    val = pd.read_csv(val_path, parse_dates=['date'])
    val = val.sort_values('date').reset_index(drop=True)

    print(f"Train: {len(train)} games")
    print(f"Val:   {len(val)} games {'(empty - skipping validation)' if len(val) == 0 else ''}")

    return train, val


def objective(trial, X, y):
    """
    Optuna objective - minimize MAE using TimeSeriesSplit CV
    """
    params = {
        'max_depth'        : trial.suggest_int('max_depth', 2, 6),
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 600),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.5, 5.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 2.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'objective'        : 'reg:squarederror',
        'verbosity'        : 0,
        'random_state'     : 42
    }

    # TimeSeriesSplit respects temporal ordering
    # 5 folds on training data
    tscv     = TimeSeriesSplit(n_splits=5)
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False
        )

        preds = model.predict(X_vl)
        mae   = mean_absolute_error(y_vl, preds)
        fold_maes.append(mae)

    return np.mean(fold_maes)


def train_model(train_df, val_df, n_trials=50):

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    # Optuna hyperparameter search
    print("\nRunning Optuna search...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\nBest CV MAE:  {study.best_value:.2f} points")
    print(f"Best params:  {study.best_params}")

    # Retrain on full training set with best params
    best_params = study.best_params.copy()
    best_params['objective']    = 'reg:squarederror'
    best_params['verbosity']    = 0
    best_params['random_state'] = 42

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # ---- Only evaluate on val if it has rows ----
    if val_df is not None and len(val_df) > 0:
        X_val  = val_df[FEATURE_COLS].values
        y_val  = val_df[TARGET_COL].values

        val_preds = final_model.predict(X_val)
        val_mae   = mean_absolute_error(y_val, val_preds)
        dir_acc   = ((val_preds > 0) == (y_val > 0)).mean()

        print(f"\n{'='*50}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"MAE:                {val_mae:.2f} points")
        print(f"Direction accuracy: {dir_acc*100:.1f}%")

        print(f"\nSample predictions vs actual:")
        print(f"{'Predicted':>12} {'Actual':>8} {'Error':>8} {'Correct':>8}")
        for pred, actual in zip(val_preds[:15], y_val[:15]):
            correct = '✓' if (pred > 0) == (actual > 0) else '✗'
            print(f"{pred:>+12.1f} {actual:>+8.1f} {abs(pred-actual):>8.1f} {correct:>8}")
    else:
        print("\nNo validation set - skipping validation metrics")
        print(f"Model trained on {len(train_df)} games")
        print("Proceeding directly to predictions")

    # Feature importance always prints regardless of val
    importance = pd.DataFrame({
        'feature'   : FEATURE_COLS,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nFeature importances:")
    print(importance.to_string(index=False))

    return final_model, study


def predict_matchup(model, games_rolling_df, supplements_df,
                    team_a, team_b, location=0):
    """
    Predict MOV for a tournament matchup

    team_a: string name as it appears in Stathead data
    team_b: string name as it appears in Stathead data
    location: always 0 for NCAA tournament (neutral site)

    Returns predicted MOV from team_a perspective
    Positive = team_a wins, negative = team_b wins
    """

    def get_team_features(team_name, df, supplements):
        # Get most recent rolling stats for this team
        team_rows = df[df['team'] == team_name].sort_values('date')

        if len(team_rows) == 0:
            print(f"WARNING: {team_name} not found in rolling data")
            return None

        latest = team_rows.iloc[-1]

        # Get SOS and Pace from supplements
        norm_name  = normalize_stathead_name(team_name)
        supp_row   = supplements[supplements['Team'] == norm_name]

        sos  = supp_row['SOS'].values[0]  if len(supp_row) > 0 else 0
        pace = supp_row['Pace'].values[0] if len(supp_row) > 0 else 0

        return {
            'OffRtg'     : latest['roll_OffRtg'],
            'DefRtg'     : latest['roll_DefRtg'],
            'ThreePA'    : latest['roll_ThreePA'],
            'TOV'        : latest['roll_TOV'],
            'ORB'        : latest['roll_ORB'],
            'DRB'        : latest['roll_DRB'],
            'RecentForm' : latest['recent_form'],
            'SOS'        : sos,
            'Pace'       : pace,
        }

    stats_a = get_team_features(team_a, games_rolling_df, supplements_df)
    stats_b = get_team_features(team_b, games_rolling_df, supplements_df)

    if stats_a is None or stats_b is None:
        return None

    # Build prediction row as differences
    row = {
        'OffRtg_diff'    : stats_a['OffRtg']     - stats_b['DefRtg'],
        'DefRtg_diff'    : stats_a['DefRtg']      - stats_b['OffRtg'],
        'ThreePA_diff'   : stats_a['ThreePA']     - stats_b['ThreePA'],
        'TOV_diff'       : stats_a['TOV']         - stats_b['TOV'],
        'ORB_diff'       : stats_a['ORB']         - stats_b['ORB'],
        'DRB_diff'       : stats_a['DRB']         - stats_b['DRB'],
        'RecentForm_diff': stats_a['RecentForm']  - stats_b['RecentForm'],
        'SOS_diff'       : stats_a['SOS']         - stats_b['SOS'],
        'Pace_diff'      : stats_a['Pace']        - stats_b['Pace'],
        'location'       : location
    }

    X = pd.DataFrame([row])[FEATURE_COLS].values
    predicted_mov = model.predict(X)[0]

    # Print readable output
    winner   = team_a if predicted_mov > 0 else team_b
    margin   = abs(predicted_mov)

    print(f"\n{'='*50}")
    print(f"  {team_a} vs {team_b}")
    print(f"{'='*50}")
    print(f"  Predicted winner: {winner}")
    print(f"  Predicted margin: {margin:.1f} points")
    print(f"  Predicted MOV ({team_a}): {predicted_mov:+.1f}")
    print(f"\n  Key matchup factors:")
    print(f"  OffRtg diff:     {row['OffRtg_diff']:+.1f}")
    print(f"  DefRtg diff:     {row['DefRtg_diff']:+.1f}")
    print(f"  RecentForm diff: {row['RecentForm_diff']:+.1f}")
    print(f"  SOS diff:        {row['SOS_diff']:+.1f}")

    return predicted_mov

def save_model(model, study):
    model_path = os.path.join(SAVE_DIR, 'xgb_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":

    # Load data
    train_df, val_df = load_data()

    # Train model
    model, study = train_model(train_df, val_df, n_trials=50)

    # Save model
    save_model(model, study)

    # Example predictions - Sweet 16 matchups
    # Load rolling data for predictions
    games_rolling = pd.read_csv(
        os.path.join(SAVE_DIR, 'games_rolling.csv'),
        parse_dates=['date']
    )
    supplements = pd.read_csv(
        os.path.join(SAVE_DIR, 'supplements_cleaned.csv')
    )

    print("\n" + "="*50)
    print("PREDICTIONS")
    print("="*50)

    # Replace with actual matchups as they're announced
    matchups = [
        ('TCU', 'Ohio State'),
        ('Nebraska', 'Troy'),
        ('South Florida', 'Louisville'),
        ('Wisconsin', 'High Point'),
        ('Duke', 'Siena'),
        ('McNeese State', 'Vanderbilt'),
        ('Michigan State', 'North Dakota State'),
        ('Arkansas', 'Hawaii'),
        ('North Carolina', 'VCU'),
        ('Michigan', 'Howard'),
        ('Texas', 'Brigham Young'),
        ('Texas A&M', 'St. Mary\'s (CA)'),
        ('Illinois', 'Pennsylvania'),
        ('Saint Louis', 'Georgia'),
        ('Idaho', 'Houston'),
        ('Gonzaga', 'Kennesaw State'),
        ('Kentucky', 'Santa Clara'),
        ('Texas Tech', 'Akron'),
        ('LIU', 'Arizona'),
        ('Wright State', 'Virginia'),
        ('Tennessee State', 'Iowa State'),
        ('Hofstra', 'Alabama'),
        ('Utah State', 'Villanova'),
        ('Tennessee', 'Miami (OH)')
          
    ]

    if len(matchups) == 0:
        print("\nNo matchups specified - skipping predictions")
        print("Fill in matchups list once satisfied with validation results")
    else:
        results = []
        for team_a, team_b in matchups:
            mov = predict_matchup(
                model, games_rolling, supplements,
                team_a, team_b, location=0
            )
            if mov is not None:
                results.append({
                    'team_a'   : team_a,
                    'team_b'   : team_b,
                    'pred_mov' : round(mov, 1),
                    'winner'   : team_a if mov > 0 else team_b,
                    'margin'   : round(abs(mov), 1)
                })

        results_df = pd.DataFrame(results)
        print(f"\nAll predictions:")
        print(results_df.to_string(index=False))
        results_df.to_csv(os.path.join(SAVE_DIR, 'predictions.csv'), index=False)