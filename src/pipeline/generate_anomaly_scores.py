from pathlib import Path
import pandas as pd
from src.pipeline.unsupervised import UnsupervisedModel

DATA_PATH = Path("data/interim/features.parquet")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("data/processed/anomaly_scores.parquet")

def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

def main():
    df = load_data()
    df = df.dropna()
    X = df.select_dtypes(include=["float64", "int64"])
    
    models = UnsupervisedModel.load(MODEL_DIR)
    
    df["anomaly_iso"] = -models.iso.score_samples(X)
    df["anomaly_lof"] = -models.lof.score_samples(X)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()