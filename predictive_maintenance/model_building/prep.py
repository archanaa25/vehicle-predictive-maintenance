# for data manipulation
import pandas as pd
import sklearn
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
DATASET_PATH = "hf://datasets/arss25/VehiclePreditiveMaintanence/engine_data.csv"
vehicle_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Engine Condition'


# Define numeric and categorical features
numeric_features = vehicle_dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
#remove target feature from numeric_feature
numeric_features.remove(target)

# Define predictor matrix (X) using selected numeric and categorical features
X = vehicle_dataset[numeric_features]

# Define target variable
y = vehicle_dataset[target]


# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="arss25/VehiclePreditiveMaintanence",
        repo_type="dataset",
    )
