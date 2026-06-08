
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json

# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
from imblearn.over_sampling import SMOTE

# Authenticate with HuggingFace (required for hf:// paths in pandas)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")
login(token=hf_token)

# MLflow tracking configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///content/mlruns"))
mlflow.set_experiment("Vehicle_PredictiveMaintenance")

# Helper function to evaluate model performance
def model_performance_classification_sklearn(model, predictors, target):
    pred = model.predict(predictors)
    pred_proba = model.predict_proba(predictors)[:, 1] if hasattr(model, "predict_proba") else pred
    return pd.DataFrame({
        "Accuracy":  [accuracy_score(target, pred)],
        "Precision": [precision_score(target, pred, zero_division=0)],
        "Recall":    [recall_score(target, pred, zero_division=0)],
        "F1-Score":  [f1_score(target, pred, zero_division=0)],
        "AUC-ROC":   [roc_auc_score(target, pred_proba)],
    })

# Load data from Hugging Face datasets
Xtrain_path = "hf://datasets/arss25/VehiclePreditiveMaintanence/Xtrain.csv"
Xtest_path  = "hf://datasets/arss25/VehiclePreditiveMaintanence/Xtest.csv"
ytrain_path = "hf://datasets/arss25/VehiclePreditiveMaintanence/ytrain.csv"
ytest_path  = "hf://datasets/arss25/VehiclePreditiveMaintanence/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

target = 'Engine Condition'
numeric_features = Xtrain.columns.tolist()

# Apply SMOTE oversampling to the training data
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=42)
Xtrain_smote, ytrain_smote = sm.fit_resample(Xtrain, ytrain)
print("Training data resampled with SMOTE.")

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 5, 10],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__max_features': ['sqrt', 'log2'],
    'randomforestclassifier__class_weight': [None],
}

model_pipeline = make_pipeline(preprocessor, rf_model)

with mlflow.start_run() as run:
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain_smote, ytrain_smote)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    best_model_name = "Tuned Random Forest"

    y_pred_train = best_model.predict(Xtrain_smote)
    y_pred_test  = best_model.predict(Xtest)

    train_report = classification_report(ytrain_smote, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        'train_accuracy':  train_report['accuracy'],
        'train_precision': train_report['1']['precision'],
        'train_recall':    train_report['1']['recall'],
        'train_f1-score':  train_report['1']['f1-score'],
        'test_accuracy':   test_report['accuracy'],
        'test_precision':  test_report['1']['precision'],
        'test_recall':     test_report['1']['recall'],
        'test_f1-score':   test_report['1']['f1-score']
    })

    final_test_metrics = model_performance_classification_sklearn(best_model, Xtest, ytest)

    model_path = "predictive_maintenance/model_building/best_engine_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    meta = {
        "model_name": best_model_name,
        "features": numeric_features,
        "alert_threshold": 0.50,
        "test_f1":       round(float(final_test_metrics["F1-Score"][0]), 4),
        "test_recall":   round(float(final_test_metrics["Recall"][0]), 4),
        "test_accuracy": round(float(final_test_metrics["Accuracy"][0]), 4),
    }
    with open("predictive_maintenance/model_building/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Metadata:", json.dumps(meta, indent=2))

    api = HfApi(token=hf_token)
    model_repo_id = "arss25/VehiclePredictiveMaintenance-Model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
        print(f"Model repo already exists: {model_repo_id}")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)
        print(f"Created model repo: {model_repo_id}")

    for upload_file in [model_path, "predictive_maintenance/model_building/model_metadata.json"]:
        api.upload_file(
            path_or_fileobj=upload_file,
            path_in_repo=upload_file.split("/")[-1],
            repo_id=model_repo_id,
            repo_type="model",
        )
        print(f"Uploaded: {upload_file}")

    print(f"\nModel registered at: https://huggingface.co/{model_repo_id}")
