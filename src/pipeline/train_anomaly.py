import pandas as pd
from pathlib import Path
from datetime import datetime
from src.pipeline.unsupervised import UnsupervisedModel

DATA_PATH = "data/interim/features.parquet"
MODEL_DIR = Path("model")
DOC_PATH = Path("docs/model_notes.MD")

def load_data():
    df = pd.read_parquet(DATA_PATH)
    X = df.select_dtypes(include=["float64", "int64"])
    X = X.dropna()
    
    return X

def write_summary(model, X):
    text = []
    text.append(f"# Training Summary — {datetime.now()}\n")
    text.append(f"- Data shape: {X.shape}\n")
    text.append("- IsolationForest: 200 trees, auto contamination\n")
    text.append("- LOF: 20 neighbors, novelty=True\n")
    text.append("- One-Class SVM: RBF kernel, nu=0.05\n")

    DOC_PATH.write_text("\n".join(text))
    
def main():
    X = load_data()
    m = UnsupervisedModel()
    m.fit(X)
    m.save(MODEL_DIR)
    write_summary(m, X)
    
if __name__ == "__main__":
    main()