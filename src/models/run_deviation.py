import pandas as pd
from deviaton_model import compute_deviation, save_deviation

def main():
    IN = "data/interim/data_10min.parquet"
    OUT_SCORES = "data/processed/deviation_scores.parquet"
    OUT_PARAMS = "config/deviation_params.json"
    
    cols = ["p (mbar)", "T (degC)", "rh (%)", "wv (m/s)"]
    windows = {
        '1h': 6,
        '3h': 18,
        '12h': 72
    }
    
    df = pd.read_parquet(IN)
    
    deviation_df, params = compute_deviation(df, cols, windows)
    
    save_deviation(deviation_df, params, OUT_SCORES, OUT_PARAMS)

if __name__ == '__main__':
    main()