import pandas as pd
import time
import os 






df = pd.read_csv(r'C:\Users\Massimo Camuso\Desktop\march madness model\data1.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())


SAVE_DIR = r"C:\Users\Massimo Camuso\Desktop\march madness model"

def scrape_teamrankings_stat(url, stat_name):
    """
    Simple TeamRankings scraper
    Returns DataFrame with Team and stat columns
    """
    try:
        time.sleep(2)  # rate limiting
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        tables = pd.read_html(url, storage_options={"User-Agent": headers['User-Agent']})
        
        if not tables:
            print(f"No tables found for {stat_name}")
            return None
        
        df = tables[0]
        print(f"\n{stat_name} raw columns: {df.columns.tolist()}")
        print(df.head(3))
        
        return df
    
    except Exception as e:
        print(f"Failed {stat_name}: {e}")
        return None


def get_sos_and_pace():
    """
    Scrape only SOS and Pace from TeamRankings NCAAB
    These are the only two stats not in Stathead
    """
    
    urls = {
        "SOS": "https://www.teamrankings.com/ncaa-basketball/ranking/schedule-strength-by-other",
        "Pace": "https://www.teamrankings.com/ncaa-basketball/stat/possessions-per-game"
    }
    
    results = {}
    
    for stat_name, url in urls.items():
        print(f"\nScraping {stat_name}...")
        df = scrape_teamrankings_stat(url, stat_name)
        
        if df is not None:
            results[stat_name] = df
    
    return results


if __name__ == "__main__":
    raw_results = get_sos_and_pace()
    
    # Save raw outputs first
    # Inspect before cleaning - column names vary on TeamRankings
    for stat_name, df in raw_results.items():
        save_path = os.path.join(SAVE_DIR, f'teamrankings_{stat_name}_raw.csv')
        df.to_csv(save_path, index=False)
        print(f"Saved raw {stat_name} to {save_path}")