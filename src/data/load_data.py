import pandas as pd

def load_data():
    df = pd.read_csv('data/raw/jena_climate_2009_2016.csv')
    print('load_data success')
    return df