from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import joblib

class UnsupervisedModel:
    def __init__(self):
        self.iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto")
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        
    def fit(self, X):
        self.iso.fit(X)
        self.lof.fit(X)
        self.ocsvm.fit(X)

    def save(self, path):
        joblib.dump(self.iso, f"{path}/isoforest.pkl")
        joblib.dump(self.lof, f"{path}/lof.pkl")
        joblib.dump(self.ocsvm, f"{path}/ocsvm.pkl")