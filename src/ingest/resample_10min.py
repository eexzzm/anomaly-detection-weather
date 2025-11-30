import pandas as pd

def resample(df):
    full_index = pd.date_range(
        start=df["Date Time"].min(),
        end=df["Date Time"].max(),
        freq='10min'
    )
    
    # reindex to force every time stamp to exist
    # missing time stamp will set to NaN  
    df = df.set_index("Date Time").reindex(full_index)
    df.index.name = "Date Time"
    
    # non-volatile variables that is safe to be fill with artificial values
    safe_col = [
        "p (mbar)",
        "T (degC)",
        "Tdew (degC)",
        "rh (%)",
        "VPact (mbar)",
        "VPdef (mbar)",
    ]
    
    # imputation flag detects NaN
    for col in safe_col:
        df[col + '_filled'] = df[col].isna()
        
    # forward fill only on short gaps
    df[safe_col] = df[safe_col].ffill(limit=3)
    
    return df

if __name__ == '__main__':
    IN_PATH = "data/interim/cleaned_data.parquet"
    OUT_PATH = "data/interim/data_10min.parquet"
    
    df = pd.read_parquet(IN_PATH)
    df = resample(df)
    df.to_parquet(OUT_PATH)