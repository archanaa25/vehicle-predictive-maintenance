# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Authenticate with HuggingFace (required for hf:// paths in pandas)
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Define constants for the dataset and output paths
api = HfApi(token=hf_token)
DATASET_PATH = "hf://datasets/arss25/VehiclePreditiveMaintanence/engine.csv"
vehicle_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Initial shape: {vehicle_dataset.shape}")

# --------------------------------------------------------------------------
# Explicit data cleaning and unnecessary-column removal
# --------------------------------------------------------------------------
# Remove obvious unnecessary columns if they exist in source data
candidate_unnecessary_cols = ["Unnamed: 0", "id", "index"]
unnecessary_columns = [c for c in candidate_unnecessary_cols if c in vehicle_dataset.columns]

if unnecessary_columns:
    vehicle_dataset = vehicle_dataset.drop(columns=unnecessary_columns)
    print(f"Removed unnecessary columns: {unnecessary_columns}")
else:
    print("No unnecessary columns found to remove.")

# Remove duplicate rows if any
duplicate_rows = int(vehicle_dataset.duplicated().sum())
if duplicate_rows > 0:
    vehicle_dataset = vehicle_dataset.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicate rows: {duplicate_rows}")
else:
    print("No duplicate rows found.")

# Handle missing values (drop rows only if missing values exist)
missing_cells = int(vehicle_dataset.isnull().sum().sum())
if missing_cells > 0:
    vehicle_dataset = vehicle_dataset.dropna().reset_index(drop=True)
    print(f"Missing values found ({missing_cells}) and removed using dropna().")
else:
    print("No missing values found.")

print(f"Final shape after cleaning: {vehicle_dataset.shape}")

# Define the target variable for the classification task
target = "Engine Condition"

# Define numeric features
numeric_features = vehicle_dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove target feature from numeric features
numeric_features.remove(target)

# Define predictor matrix (X) and target variable (y)
X = vehicle_dataset[numeric_features]
y = vehicle_dataset[target]

# Split the cleaned dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Save split datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Saved cleaned train/test files locally.")

# Upload cleaned train/test datasets to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="arss25/VehiclePreditiveMaintanence",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face dataset repo.")
