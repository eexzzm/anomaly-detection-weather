from src.data.clean_data import clean_data
from src.data.load_data import load_data

RAW_PATH = 'data/raw/jena_climate_2009_2016.csv'
OUT_PATH = 'data/interim/cleaned_data.parquet'

def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    
    df.to_parquet(OUT_PATH)

if __name__ == "__main__":
    main()