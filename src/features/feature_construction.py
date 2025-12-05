import pandas as pd
import numpy as np
import json


# extract core signal from the resampled weather table
def base_feature(df):
    df = df.copy()
    
    # difference features
    df["dT_10"]  = df["T (degC)"].diff(1)
    df["dT_30"]  = df["T (degC)"].diff(3)
    df["dT_60"]  = df["T (degC)"].diff(6)

    df["dp_10"]  = df["p (mbar)"].diff(1)
    df["dp_30"]  = df["p (mbar)"].diff(3)
    df["dp_60"]  = df["p (mbar)"].diff(6)

    df["drh_10"] = df["rh (%)"].diff(1)
    df["drh_30"] = df["rh (%)"].diff(3)
    df["drh_60"] = df["rh (%)"].diff(6)
    
    # rolling mean and std
    windows = {
        '1h' : 6,
        '3h' : 18,
        '12h' : 72,
    }
    
    # rolling process
    for name, win in windows.items():
        df[f"T_mean_{name}"] = df["T (degC)"].rolling(win).mean()
        df[f"T_std_{name}"] = df["T (degC)"].rolling(win).std()
        df[f"p_mean_{name}"] = df["p (mbar)"].rolling(win).mean()
        df[f"p_std_{name}"] = df["p (mbar)"].rolling(win).std()
        df[f"rh_mean_{name}"] = df["rh (%)"].rolling(win).mean()
        df[f"rh_std_{name}"] = df["rh (%)"].rolling(win).std()
    
    # physical consistency dew spread
    df["dew_spread"] = df["T (degC)"] - df["Tdew (degC)"]
    
    return df

# convert cyclic time pattern into usable numeric format for the model to understand
def seasonal_features(df):
    df = df.copy()
    
    hour = df.index.hour
    dayofyear = df.index.dayofyear
    
    # hour-of-day cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # day-of-year cyclic encoding
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365)
    
    return df

# give stable numeric scale features to remove distortion from outliers
def scale_features(df, save_path):
    df = df.copy()
    
    # exclude boolean columns
    df_num = df.select_dtypes(include=[np.number])
    
    # robust stats
    median = df_num.median()
    iqrs = df_num.quantile(0.75) - df_num.quantile(0.25)
    iqrs = iqrs.replace(0, 1e-9) # avoid zero division
    
    scaled = (df_num - median) / iqrs
    
    final = pd.concat([scaled, df.select_dtypes(include=['bool'])], axis=1)
    
    params = {
        'medians' : median.to_dict(),
        'iqrs' : iqrs.to_dict()
    }
    
    with open(save_path, "w") as f:
        json.dump(params, f, indent=2)
        
    return final