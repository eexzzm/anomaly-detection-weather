from src.features.feature_construction import base_feature, scale_features, seasonal_features
import pandas as pd
    
INPUT = "data/interim/data_10min.parquet"
OUTPUT = "data/interim/features.parquet"

def main():
    df = pd.read_parquet(INPUT)
    
    df = base_feature(df)
    df = seasonal_features(df, )
    df = scale_features(df, "config/scaler_params.json")
    
    df.to_parquet(OUTPUT, index=True)
    
if __name__ == "__main__":
    main()