import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. SECURE CONFIGURATION
def setup_mlflow():
    # Use dbutils to get the token you just uploaded via CLI
    try:
        # These will work natively on Databricks Serverless
        token = dbutils.secrets.get(scope="mlops_scope", key="databricks_token")
        
        # Public configuration (Safe to keep in code)
        os.environ["DATABRICKS_HOST"] = "https://community.cloud.databricks.com"
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"
        
        # Your specific experiment ID
        mlflow.set_experiment(experiment_id="1048423805993394")
        print("Successfully configured MLflow with Secrets.")
    except Exception as e:
        print(f"Error accessing secrets: {e}")
        raise e

# 2. DATA PREPARATION
def load_and_prep_data():
    print("Loading diamonds dataset...")
    # Using Databricks built-in dataset
    df = pd.read_csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv").drop("Unnamed: 0", axis=1)
    df = pd.get_dummies(df)
    
    X = df.drop("price", axis=1)
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING & REGISTRATION
def train_and_register():
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_and_prep_data()

    with mlflow.start_run(run_name="Secure_Serverless_Run") as run:
        # Define model parameters
        params = {"n_estimators": 100, "max_depth": 6}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Log to Experiment
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        
        # Register to Unity Catalog
        # This requires the 3-part name: catalog.schema.model
        model_name = "workspace.default.test_model"
        signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )
        
        print(f"Model Registered: {model_name}")
        print(f"Final MSE: {mse:.2f}")

if __name__ == "__main__":
    train_and_register()