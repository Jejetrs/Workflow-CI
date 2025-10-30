import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ========================
# ARGUMENT PARSER
# ========================
parser = argparse.ArgumentParser(description="Train Diabetes Prediction Model")
parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
parser.add_argument("--scaler_path", type=str, default="scaler.pkl")
parser.add_argument("--n_iter", type=int, default=30)
parser.add_argument("--cv_folds", type=int, default=5)
args = parser.parse_args()

# ========================
# LOAD DATA
# ========================
data = pd.read_csv(args.data_path)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# IDENTIFIKASI KOLOM
# ========================
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns

# ========================
# PREPROCESSOR
# ========================
# Load scaler dari file, tapi jangan gunakan transform di luar pipeline
scaler = joblib.load(args.scaler_path)

numeric_transformer = Pipeline([
    ("scaler", scaler),
    ("poly", PolynomialFeatures(include_bias=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# ========================
# PIPELINE MODEL
# ========================
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(max_iter=5000, random_state=42))
])

# ========================
# RANDOMIZED SEARCH
# ========================
param_dist = {
    "preproc__num__poly__degree": [1, 2],
    "clf__penalty": ["elasticnet", "l2"],
    "clf__solver": ["saga"],
    "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "clf__C": uniform(0.001, 10),
    "clf__class_weight": [None, "balanced"],
    "clf__tol": [1e-4, 1e-3]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=args.n_iter,
    scoring="f1",
    cv=args.cv_folds,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# fit langsung DataFrame, biar pipeline bisa akses columns
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_
print("\nBest Hyperparameters:", best_params)
print("="*60)

# ========================
# EVALUASI MODEL
# ========================
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluasi Tuning Model Logistic Regression")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("="*60)

# ========================
# LOGGING MLflow
# ========================
# Log parameters
mlflow.log_params(best_params)
mlflow.log_param("n_iter", args.n_iter)
mlflow.log_param("cv_folds", args.cv_folds)

# Log metrics
mlflow.log_metrics({
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1
})

# Log model
mlflow.sklearn.log_model(best_model, "model")

# Log scaler artifact
mlflow.log_artifact(args.scaler_path, artifact_path="preprocessing")

# Get run info
active_run = mlflow.active_run()
if active_run:
    run_id = active_run.info.run_id
    experiment_id = active_run.info.experiment_id
    print(f"✅ MLflow Run ID: {run_id}")
    print(f"✅ Experiment ID: {experiment_id}")

# ========================
# SAVE MODEL LOCAL
# ========================
local_model_path = "diabetes_model_ci.pkl"
joblib.dump(best_model, local_model_path)
print(f"\nModel saved locally: {local_model_path}")

print("\n" + "="*60)
print("✅ Training completed successfully!")
print(f"Final Metrics - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
print("="*60)
