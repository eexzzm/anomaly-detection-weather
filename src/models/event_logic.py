import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/processed/anomaly_scores.parquet"
OUTPUT = "data/processed/events.parquet"

# to convert anomaly scores into range 0 - 1
def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())

# construct event flag, severity, intepretation per timestamp
def build_event(df):
    
    WIND_SPIKE_THRESHOLD = (5.0*(30/10))
    
    # physical event detection
    df["pressure_drop"] = (df["dp_60"] < -3.0).astype(int)
    df["wind_10"] = df["wv (m/s)"].diff()
    
    df["wind_spike"] = (df["wind_10"] > WIND_SPIKE_THRESHOLD).astype(int)
    df["temp_humd_conflict"] = ((df["T (degC)"] > 30.0) & (df["rh (%)"] > 90.0) ).astype(int)    
    
    # anomaly gate
    anomaly_threshold = df["anomaly_iso"].quantile(0.99) # top 1% most anomalous
    df["ml_anomaly"] = (df["anomaly_iso"] > anomaly_threshold).astype(int)
    
    # unified event trigger
    df["event_trigger"] = (
        (   
         (df["pressure_drop"] == 1) |
         (df["wind_spike"] == 1) |
         (df["wind_spike"] == 1)
        ) & (
            df["ml_anomaly"] == 1
        )
    ).astype(int)
    
    # severity components
    '''
    assign a group for each continous segment
    and detect how many event inside each group.
    track how many physical indicators fired on each timestamp
    '''
    df["severity_magnitude"] = normalize(df["anomaly_iso"])
    
    df["event_group"] = (
        df["event_trigger"] != df["event_trigger"].shift()
    ).cumsum()
    
    df["event_duration"] = (
        df.groupby("event_group")["event_trigger"]
        .transform("sum")
    )
    
    df["agreement_count"] = (
        df["pressure_drop"] +
        df["wind_spike"] +
        df["temp_humd_conflict"]
    )
    
    # final severity score
    df["severity"] = (
        0.5 * df["severity_magnitude"] +
        0.3 * (df["event_duration"] / df["event_duration"].max()) +
        0.2 * (df["agreement_count"] / 3.0)
    )
    
    # give human-readable interpretation
    df["interpretation"] = np.select(
        [
            df["severity"] >= 0.8,
            df["severity"] >= 0.5,
            df["severity"] >= 0.3
        ],
        [
            "Severe weather anomaly",
            "Moderate anomaly",
            "Mild anomaly"
        ],
        default="Normal"
    )
    
    return df

def main():
    df = pd.read_parquet(DATA_PATH)
    
    df_events = build_event(df)
    
    output_cols = [
        "event_trigger",
        "severity",
        "interpretation",
        "pressure_drop",
        "wind_spike",
        "temp_humd_conflict",
        "event_duration",
        "agreement_count"
    ]

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    
    df_events[output_cols].to_parquet(OUTPUT, index=True)
    
if __name__ == "__main__":
    main()