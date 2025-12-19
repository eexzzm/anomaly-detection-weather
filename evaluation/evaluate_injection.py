import pandas as pd
from pathlib import Path
from src.models.event_logic import build_event

INPUT_PATH = "data/processed/anomaly_scores.parquet"
OUTPUT = "evaluation/report.md"

# injection window
START = "2010-06-01 12:00:00"
END = "2010-06-01 14:00:00"

def load_data(input):
    df = pd.read_parquet(input)
    df = df.sort_index()
    
    return df

def precision_evaluation(detected: pd.Series, injected: pd.Series):
    # need to fix, this is not eqyal
    # detected is a unified anomalous and the injected is only 1 anomalous event
    tp = ((detected == 1) & (injected == 1)).sum()
    fp = ((detected == 1) & (injected == 0)).sum()

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def inject_pressure(df):
    df_injected = df.copy()
    mask = (df_injected.index >= START) & (df_injected.index <= END)

    df_injected["injected"] = 0
    df_injected.loc[mask, "p (mbar)"] -= 15.0
    df_injected.loc[mask, "injected"] = 1

    return df_injected, mask

def evaluate(df, mask):
    df_events = build_event(df)

    precision = precision_evaluation(
        df_events["event_trigger"],
        df["injected"]
    )

    inject_start = df.index[mask][0]

    detected_times = df_events[
        (df_events["event_trigger"] == 1) & mask
    ].index

    if detected_times.empty:
        lag_minutes = None
    else:
        lag_minutes = (
            detected_times.min() - inject_start
        ).total_seconds() / 60

    return precision, lag_minutes

def main():
    df = load_data(INPUT_PATH)

    df_injected, mask = inject_pressure(df)

    precision, lag_minutes = evaluate(df_injected, mask)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        f.write("# Synthetic Evaluation Report — Pressure Injection\n\n")
        f.write(f"Precision: {precision:.2f}\n")

        if lag_minutes is None:
            f.write("Detection lag: no detection\n")
        else:
            f.write(f"Detection lag: {lag_minutes:.1f} minutes\n")
        
if __name__ == "__main__":
    main()