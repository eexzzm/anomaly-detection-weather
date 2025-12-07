import pandas as pd
import numpy as np
import json
from pathlib import Path

'''
belajar yang belum ngerti jir
new api and func
'''


# avoid zero mad instabilty
def compute_mad_floor(mad_series):
    valid = mad_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) == 0:
        return 1e-6
    floor = max(valid.quantile(0.05), 1e-9)
    
    return float(floor)

def compute_deviation(df, cols, windows, eps=1e-9):
    out = pd.DataFrame(index=df.index)
    mad_floors = {}
    
    for col in cols:
        series = df[col]
        
        for w_name, w in windows.items():
            
            # rolling baseline median
            median_t = series.rolling(window=w, min_periods=w).median()
            
            # rolling MAD
            abs_dev = (series - median_t).abs()
            mad_t = abs_dev.rolling(window=w, min_periods=w).median()
            
            # compute and apply MAD floor
            mad_floor = compute_mad_floor(mad_t)
            mad_floors[f"{col}_{w_name}"] = mad_floor
            mad_t = mad_t.clip(lower=mad_floor)
            
            # deviation score
            score = (series - median_t).abs() / (mad_t + eps) # raw
            score_sm = score.rolling(window=3, min_periods=1).mean() # 30 minutes smoothing
            
            # store results
            out[f"{col}_median_{w_name}"] = median_t
            out[f"{col}_mad_{w_name}"] = mad_t
            out[f"{col}_score_{w_name}"] = score
            out[f"{col}_score_sm_{w_name}"] = score_sm
    
    params = {
        'windows': windows,
        'eps': eps,
        'mad_floors_per_var': mad_floors,
    }    
    
    return out, params

# save deviation into parquet and params into json
def save_deviation(df, params, out_path_scores, out_path_params):
    Path(out_path_scores).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path_params).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path_scores)
    
    with open(out_path_params, 'w') as f:
        json.dump(params, f, indent=2)
    