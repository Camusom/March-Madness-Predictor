import pandas as pd
import numpy as np
import os
import re

SAVE_DIR = r"C:\Users\Massimo Camuso\Desktop\march madness model"



# Stathead name -> TeamRankings name
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


def run_mismatch_diagnostic(games_df, supplements_df):
    """
    Run this AFTER applying name normalization
    to see how many mismatches remain
    Paste remaining mismatches here and we add them to NAME_MAP
    """
    stathead_teams    = set(games_df['team_normalized'].unique())
    teamrankings_teams = set(supplements_df['Team'].unique())
    
    remaining_mismatches = stathead_teams - teamrankings_teams
    
    print(f"\nRemaining mismatches after normalization: {len(remaining_mismatches)}")
    if remaining_mismatches:
        print("Teams in Stathead not found in TeamRankings:")
        for t in sorted(remaining_mismatches):
            print(f"  '{t}'")
    else:
        print("All teams matched successfully")
    
    return remaining_mismatches
# ============================================================
# STEP 1: CLEAN STATHEAD DATA
# ============================================================

def clean_stathead(filepath):
    
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    
    # ---- Keep only what we need ----
    cols_to_keep = {
        'Team'        : 'team',
        'Date'        : 'date',
        'Unnamed: 9'  : 'location_raw',
        'Opp'         : 'opponent',
        'Result'      : 'result_raw',
        'ORtg'        : 'OffRtg',
        'DRtg'        : 'DefRtg',
        '3PA'         : 'ThreePA',
        'TOV'         : 'TOV',
        'ORB'         : 'ORB',
        'DRB'         : 'DRB',
        'PTS'         : 'team_score',
        'PTS.1'       : 'opp_score',
    }
    
    # Only keep columns that exist
    existing = {k: v for k, v in cols_to_keep.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)
    
    print(f"After column selection: {df.shape}")
    
    # ---- Drop header rows that repeat inside the data ----
    # Stathead sometimes repeats header rows every 200 rows
    df = df[df['team'] != 'Team'].copy()
    df = df[df['team'].notna()].copy()
    
    
    
    # ---- Parse date - handle both formats ----
    # Regular season: "3/25/2026"
    # NCAA Tournament: "2026-03-26 NCAA"
    # NIT:            "3/25/2026" (same as regular season)
    
    def parse_stathead_date(val):
        if pd.isna(val):
            return pd.NaT
        
        val = str(val).strip()
        
        # Strip any tournament suffix
        for suffix in [' NCAA', ' NIT', ' CBI', ' CIT', ' NIT2']:
            val = val.replace(suffix, '').strip()
        
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT
        
    df['date'] = df['date'].apply(parse_stathead_date)
    
    # Report how many dates failed to parse
    failed = df['date'].isna().sum()
    if failed > 0:
        print(f"WARNING: {failed} rows failed date parsing")
        print("Sample failed dates:")
        print(df[df['date'].isna()]['date_raw'].head(10).tolist())
    
    df = df[df['date'].notna()].copy()


    # ---- Encode location ----
    # @ = away (team is visiting), N = neutral, blank = home
    def encode_location(val):
        if pd.isna(val) or str(val).strip() == '':
            return 1    # home
        elif str(val).strip() == '@':
            return -1   # away
        elif str(val).strip() == 'N':
            return 0    # neutral
        else:
            return 1    # default to home for anything unexpected
    
    df['location'] = df['location_raw'].apply(encode_location)
    
    # ---- Parse MOV from Result column ----
    # Result format is typically "W 75-60" or "L 55-71"
    
    
    df['team_score'] = pd.to_numeric(df['team_score'], errors='coerce')
    df['opp_score']  = pd.to_numeric(df['opp_score'],  errors='coerce')
    df['MOV']        = df['team_score'] - df['opp_score']
    
    # ---- Convert stat columns to numeric ----
    stat_cols = ['OffRtg', 'DefRtg', 'ThreePA', 'TOV', 'ORB', 'DRB']
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ---- Clean team and opponent names ----
    # Remove records like "(15-3)" and whitespace
    for col in ['team', 'opponent']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s*\(\d+-\d+\)', '', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # ---- Drop rows missing critical data ----
    df = df.dropna(subset=['team', 'date', 'MOV', 'OffRtg', 'DefRtg'])
    
    # ---- Sort by date ----
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    
    # ---- Drop raw helper columns ----
    df = df.drop(columns=['location_raw', 'result_raw', 
                           'team_score', 'opp_score'], errors='ignore')
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique teams: {df['team'].nunique()}")
    print(f"MOV range: {df['MOV'].min():.0f} to {df['MOV'].max():.0f}")
    print(f"Null counts:\n{df.isnull().sum()}")
    print(df.head(3).to_string())

    # ---- Verify tournament games included ----
    tourney_games = df[df['date'] >= '2026-03-19']
    print(f"NCAA Tournament games included: {len(tourney_games)}")
    if len(tourney_games) > 0:
        print(f"Sample tournament games:")
        print(tourney_games[['team', 'date', 'opponent', 'MOV']].head(10).to_string())
    
    return df


# ============================================================
# STEP 2: CLEAN TEAMRANKINGS SOS AND PACE
# ============================================================

def clean_teamrankings(sos_path, pace_path):
    
    # ---- SOS ----
    sos_df = pd.read_csv(sos_path)
    print(f"\nSOS raw: {sos_df.shape}")
    
    sos_clean = sos_df[['Team', 'Rating']].copy()
    sos_clean.columns = ['Team', 'SOS']
    
    # ---- Pace ----
    pace_df = pd.read_csv(pace_path)
    print(f"Pace raw: {pace_df.shape}")
    
    # '2025' column is current season pace
    pace_clean = pace_df[['Team', '2025']].copy()
    pace_clean.columns = ['Team', 'Pace']
    
    # ---- Clean team names on both ----
    for df in [sos_clean, pace_clean]:
        df['Team'] = df['Team'].astype(str).str.strip()
        # Remove records like "(31-3)"
        df['Team'] = df['Team'].str.replace(r'\s*\(\d+-\d+\)', '', regex=True)
        df['Team'] = df['Team'].str.replace(r'\s+', ' ', regex=True)
    
    # ---- Convert to numeric ----
    sos_clean['SOS']   = pd.to_numeric(sos_clean['SOS'],  errors='coerce')
    pace_clean['Pace'] = pd.to_numeric(pace_clean['Pace'], errors='coerce')
    
    # ---- Merge into one supplements dataframe ----
    supplements = sos_clean.merge(pace_clean, on='Team', how='outer')
    supplements = supplements.dropna(subset=['Team'])
    supplements = supplements.drop_duplicates(subset=['Team'])
    
    print(f"\nSupplements shape: {supplements.shape}")
    print(f"Sample:\n{supplements.head(5).to_string()}")
    print(f"Null counts: SOS={supplements['SOS'].isna().sum()}, "
          f"Pace={supplements['Pace'].isna().sum()}")
    
    return supplements


# ============================================================
# STEP 3: COMPUTE ROLLING FEATURES PER TEAM
# ============================================================

def compute_rolling_features(games_df, recent_form_window=5, min_games=3):
    """
    For each game row, compute rolling averages using only
    games BEFORE that game. shift(1) prevents leakage.
    
    min_games: drop rows where team has fewer than N prior games
               early season games have noisy/NaN rolling features
    """
    
    games_df = games_df.sort_values(['team', 'date']).copy()
    
    rolling_stat_cols = ['OffRtg', 'DefRtg', 'ThreePA', 'TOV', 'ORB', 'DRB']
    
    all_teams = []
    
    for team, group in games_df.groupby('team'):
        group = group.sort_values('date').copy()
        
        # Rolling season average (expanding, excludes current game)
        for stat in rolling_stat_cols:
            if stat in group.columns:
                group[f'roll_{stat}'] = (
                    group[stat]
                    .shift(1)
                    .expanding(min_periods=1)
                    .mean()
                )
        
        # Recent form: last N games MOV average
        group['recent_form'] = (
            group['MOV']
            .shift(1)
            .rolling(window=recent_form_window, min_periods=1)
            .mean()
        )
        
        # Track how many games played before this one
        group['games_played'] = range(len(group))
        
        all_teams.append(group)
    
    result = pd.concat(all_teams, ignore_index=True)
    
    # Drop early games where rolling features are unreliable
    before = len(result)
    result = result[result['games_played'] >= min_games].copy()
    after  = len(result)
    
    print(f"\nRolling features computed")
    print(f"Dropped {before - after} rows with < {min_games} prior games")
    print(f"Remaining: {after} rows")
    
    return result


# ============================================================
# STEP 4: BUILD MATCHUP ROWS (self-join)
# ============================================================

def build_matchup_rows_v2(games_df, supplements_df):
    """
    Updated version with name normalization applied before join
    """
    
    games_df = games_df.copy()
    
    # ---- Normalize team names before joining ----
    games_df['team_normalized'] = games_df['team'].apply(normalize_stathead_name)
    
    # ---- Diagnostic: check remaining mismatches ----
    remaining = run_mismatch_diagnostic(games_df, supplements_df)
    
    if len(remaining) > 0:
        print("\nThese teams will have null SOS/Pace after join")
        print("Add them to NAME_MAP or accept the data loss")
    
    # ---- Join supplements using normalized name ----
    games_df = games_df.merge(
        supplements_df.rename(columns={'Team': 'team_normalized'}),
        on='team_normalized',
        how='left'
    )
    
    missing_after = games_df['SOS'].isna().sum()
    total = len(games_df)
    print(f"\nAfter normalized join:")
    print(f"  Missing SOS: {missing_after} / {total} ({100*missing_after/total:.1f}%)")
    
    # ---- Build opponent lookup ----
    roll_cols = ['roll_OffRtg', 'roll_DefRtg', 'roll_ThreePA',
                 'roll_TOV', 'roll_ORB', 'roll_DRB', 'recent_form',
                 'SOS', 'Pace']
    available_roll_cols = [c for c in roll_cols if c in games_df.columns]
    
    # Opponent lookup uses ORIGINAL team name to match
    # because games_df['opponent'] uses original Stathead names
    opponent_lookup = games_df[['team', 'date'] + available_roll_cols].copy()
    opponent_lookup.columns = (
        ['opponent', 'date'] +
        [f'opp_{c}' for c in available_roll_cols]
    )
    
    # ---- Self join ----
    matchups = games_df.merge(
        opponent_lookup,
        on=['date', 'opponent'],
        how='inner'
    )
    
    print(f"\nAfter self-join: {len(matchups)} rows")
    
    # Deduplicate
    matchups = matchups[matchups['team'] > matchups['opponent']].copy()
    print(f"After deduplication: {len(matchups)} unique games")
    
    # ---- Difference features ----
    feature_map = {
        'OffRtg_diff'    : ('roll_OffRtg',  'opp_roll_OffRtg'),
        'DefRtg_diff'    : ('roll_DefRtg',  'opp_roll_DefRtg'),
        'ThreePA_diff'   : ('roll_ThreePA', 'opp_roll_ThreePA'),
        'TOV_diff'       : ('roll_TOV',     'opp_roll_TOV'),
        'ORB_diff'       : ('roll_ORB',     'opp_roll_ORB'),
        'DRB_diff'       : ('roll_DRB',     'opp_roll_DRB'),
        'RecentForm_diff': ('recent_form',  'opp_recent_form'),
        'SOS_diff'       : ('SOS',          'opp_SOS'),
        'Pace_diff'      : ('Pace',         'opp_Pace'),
    }
    
    for new_col, (col_a, col_b) in feature_map.items():
        if col_a in matchups.columns and col_b in matchups.columns:
            matchups[new_col] = matchups[col_a] - matchups[col_b]
        else:
            print(f"WARNING: Cannot compute {new_col}")
    
    feature_cols = [c for c in feature_map.keys() if c in matchups.columns]
    final_cols   = ['date', 'team', 'opponent', 'location', 'MOV'] + feature_cols
    final_df     = matchups[final_cols].copy()
    
    # Drop only where core efficiency features are null
    # SOS/Pace nulls are less critical - can fill with 0
    core_features = ['OffRtg_diff', 'DefRtg_diff', 'RecentForm_diff']
    final_df = final_df.dropna(subset=core_features)
    
    # Fill remaining nulls (SOS/Pace mismatches) with 0
    # 0 = no difference, neutral assumption
    final_df[feature_cols] = final_df[feature_cols].fillna(0)
    
    print(f"\nFinal dataset: {final_df.shape}")
    print(f"Date range: {final_df['date'].min().date()} "
          f"to {final_df['date'].max().date()}")
    
    return final_df


# ============================================================
# STEP 5: TRAIN / VAL / TEST SPLIT
# ============================================================

def split_data(matchups_df):
    
    matchups_df['date'] = pd.to_datetime(matchups_df['date'])
    
    print(f"\nFull date range: {matchups_df['date'].min().date()} "
          f"to {matchups_df['date'].max().date()}")
    
    # Move cutoff forward to include everything played so far
    # into training, leave nothing for val
    start = pd.Timestamp('2026-04-06')  # adjust to actual date
    
    train = matchups_df[matchups_df['date'] <  start]
    val   = matchups_df[matchups_df['date'] >= start]  # empty until desired round plays
    test  = pd.DataFrame()  # we build test rows manually from matchup pairs
    
    print(f"\nTrain: {len(train):,} games  "
          f"({train['date'].min().date()} - {train['date'].max().date()})")
    print(f"Val:   {len(val):,} games")
    
    return train, val, test


# ============================================================
# MAIN
# ============================================================
'''
use this main block if running for first time
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ---- Paths ----
    stathead_path = os.path.join(SAVE_DIR, 'data1.csv')
    sos_path      = os.path.join(SAVE_DIR, 'teamrankings_SOS_raw.csv')
    pace_path     = os.path.join(SAVE_DIR, 'teamrankings_Pace_raw.csv')
    
    # ---- Step 1: Clean Stathead ----
    print("="*60)
    print("STEP 1: Cleaning Stathead data")
    print("="*60)
    games_df = clean_stathead(stathead_path)
    games_df.to_csv(os.path.join(SAVE_DIR, 'stathead_cleaned.csv'), index=False)
    
    # ---- Step 2: Clean TeamRankings ----
    print("\n" + "="*60)
    print("STEP 2: Cleaning TeamRankings supplements")
    print("="*60)
    supplements_df = clean_teamrankings(sos_path, pace_path)
    supplements_df.to_csv(os.path.join(SAVE_DIR, 'supplements_cleaned.csv'), index=False)
    
    # ---- Step 3: Rolling features ----
    print("\n" + "="*60)
    print("STEP 3: Computing rolling features")
    print("="*60)
    games_rolling = compute_rolling_features(games_df, recent_form_window=5, min_games=3)
    games_rolling.to_csv(os.path.join(SAVE_DIR, 'games_rolling.csv'), index=False)
    
    # ---- Step 4: Build matchups ----
    print("\n" + "="*60)
    print("STEP 4: Building matchup rows")
    print("="*60)
    matchups_df = build_matchup_rows(games_rolling, supplements_df)
    matchups_df.to_csv(os.path.join(SAVE_DIR, 'matchups_final.csv'), index=False)
    
    # ---- Step 5: Split ----
    print("\n" + "="*60)
    print("STEP 5: Train/Val/Test split")
    print("="*60)
    train, val, test = split_data(matchups_df)
    
    train.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
    val.to_csv(  os.path.join(SAVE_DIR, 'val.csv'),   index=False)
    test.to_csv( os.path.join(SAVE_DIR, 'test.csv'),  index=False)
    
    print("\n" + "="*60)
    print("DATA PIPELINE COMPLETE")
    print("="*60)
    print(f"Files saved to: {SAVE_DIR}")
    print("\nNext step: run modeling script on train.csv + val.csv")
    

if __name__ == "__main__":
    
    # Load already-cleaned files (skip steps 1-3)
    games_rolling  = pd.read_csv(os.path.join(SAVE_DIR, 'games_rolling.csv'),
                                  parse_dates=['date'])
    supplements_df = pd.read_csv(os.path.join(SAVE_DIR, 'supplements_cleaned.csv'))
    
    # Apply normalization and check remaining mismatches
    games_rolling['team_normalized'] = games_rolling['team'].apply(
        normalize_stathead_name
    )
    
    stathead_teams     = set(games_rolling['team_normalized'].unique())
    teamrankings_teams = set(supplements_df['Team'].unique())
    
    mismatches = stathead_teams - teamrankings_teams
    
    print(f"Remaining mismatches: {len(mismatches)}")
    for t in sorted(mismatches):
        print(f"  '{t}'")
        '''
'''
if __name__ == "__main__":

    # Load already-processed files
    games_rolling  = pd.read_csv(
        os.path.join(SAVE_DIR, 'games_rolling.csv'),
        parse_dates=['date']
    )
    supplements_df = pd.read_csv(
        os.path.join(SAVE_DIR, 'supplements_cleaned.csv')
    )

    # Rebuild matchups with correct name map
    print("="*60)
    print("Rebuilding matchup rows with fixed name map")
    print("="*60)
    matchups_df = build_matchup_rows_v2(games_rolling, supplements_df)
    matchups_df.to_csv(os.path.join(SAVE_DIR, 'matchups_final.csv'), index=False)
    print(f"Saved {len(matchups_df)} matchup rows")

    # Rebuild train/val/test split
    print("\n" + "="*60)
    print("Rebuilding train/val/test split")
    print("="*60)
    train, val, test = split_data(matchups_df)

    train.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
    val.to_csv(  os.path.join(SAVE_DIR, 'val.csv'),   index=False)
    test.to_csv( os.path.join(SAVE_DIR, 'test.csv'),  index=False)

    print(f"\nTrain: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Total: {len(train)+len(val)+len(test)}")
'''

#run this for new tourney data
if __name__ == "__main__":
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ---- Download fresh Stathead data ----
    # Go to the same Stathead URL
    # Change date range to include through Round of 32
    # Download all pages again
    # Save as 'stathead_updated.csv'
    
    stathead_path = os.path.join(SAVE_DIR, 'data1.csv')
    sos_path      = os.path.join(SAVE_DIR, 'teamrankings_SOS_raw.csv')
    pace_path     = os.path.join(SAVE_DIR, 'teamrankings_Pace_raw.csv')
    
    # ---- Re-run ALL steps with updated data ----
    print("="*60)
    print("STEP 1: Cleaning updated Stathead data")
    print("="*60)
    games_df = clean_stathead(stathead_path)
    games_df.to_csv(os.path.join(SAVE_DIR, 'stathead_cleaned.csv'), index=False)
    
    # SOS and Pace don't need re-scraping
    # They are season averages and won't change meaningfully
    supplements_df = clean_teamrankings(sos_path, pace_path)
    
    print("\n" + "="*60)
    print("STEP 3: Recomputing rolling features with tournament games")
    print("="*60)
    games_rolling = compute_rolling_features(
        games_df, 
        recent_form_window=5, 
        min_games=3
    )
    games_rolling.to_csv(os.path.join(SAVE_DIR, 'games_rolling.csv'), index=False)
    
    print("\n" + "="*60)
    print("STEP 4: Rebuilding matchup rows")
    print("="*60)
    matchups_df = build_matchup_rows_v2(games_rolling, supplements_df)
    matchups_df.to_csv(os.path.join(SAVE_DIR, 'matchups_final.csv'), index=False)
    
    print("\n" + "="*60)
    print("STEP 5: Updated train/val split")
    print("="*60)
    train, val, test = split_data(matchups_df)
    
    train.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
    val.to_csv(  os.path.join(SAVE_DIR, 'val.csv'),   index=False)
    
    print(f"\nUpdated training size: {len(train)}")
