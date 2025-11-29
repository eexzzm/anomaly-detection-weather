import pandas as pd
import numpy as np
error_col = [
    "wv (m/s)", "max. wv (m/s)"
]

drop_col = [
    'Tpot (K)', 'VPmax (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)'
]

def clean_error_values(df):
    # mark the error values as missing data
    for col in error_col:
        df[col] = df[col].replace(-9999, np.nan)
    
    # fill the gap with interpolation
    df[error_col] = df[error_col].interpolate(method='linear')
    
    return df
    
def clean_data(df):
    df = df.copy()
    
    df = df.drop_duplicates()
    df["Date Time"] = pd.to_datetime(df["Date Time"], dayfirst=True)
    df = df.drop(columns=drop_col, axis=1) 
    df = clean_error_values(df)
    
    return df