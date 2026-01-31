import os
import mlflow
import pandas as pd
from datetime import datetime

# 1. SECURE CONFIGURATION
def setup_inference():
    try:
        # Pulling token from the same secret scope we set up
        token = dbutils.secrets.get(scope="mlops_scope", key="databricks_token")
        os.environ["DATABRICKS_HOST"] = "https://community.cloud.databricks.com"
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"
    except Exception as e:
        print(f"Auth error: {e}")
        raise e

# 2. DATA PREP (The data you want to predict on)
def get_inference_data():
    # In a real scenario, this would be a new Delta table
    df = pd.read_csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv").drop("Unnamed: 0", axis=1)
    df_features = pd.get_dummies(df).drop("price", axis=1)
    return df_features

# 3. RUN INFERENCE
def run_batch_inference():
    setup_inference()
    
    model_uri = 'runs:/1b419844ef014423945cced3a40a972a/model'
    
    print(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    print("Running predictions...")
    data = get_inference_data()
    predictions = model.predict(data)
    
    # Create output dataframe
    results_df = data.copy()
    results_df["predicted_price"] = predictions
    results_df["inference_date"] = datetime.now()

    # IMPROVED FIX: Convert ALL incompatible integer types to Spark-compatible types
    # Spark doesn't support: int8, int16, uint8, uint16, uint32, uint64
    
    # Convert small integers to int32
    for dtype in ['int8', 'int16', 'uint8', 'uint16']:
        cols = results_df.select_dtypes(include=[dtype]).columns
        if len(cols) > 0:
            results_df[cols] = results_df[cols].astype('int32')
    
    # Convert large unsigned integers to int64
    for dtype in ['uint32', 'uint64']:
        cols = results_df.select_dtypes(include=[dtype]).columns
        if len(cols) > 0:
            results_df[cols] = results_df[cols].astype('int64')
    
    # Clean column names for Delta table compatibility
    # Delta doesn't allow spaces and special characters: ' ,;{}()\n\t='
    results_df.columns = results_df.columns.str.replace(' ', '_', regex=False)
    results_df.columns = results_df.columns.str.replace(',', '_', regex=False)
    results_df.columns = results_df.columns.str.replace(';', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('{', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('}', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('(', '_', regex=False)
    results_df.columns = results_df.columns.str.replace(')', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('\n', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('\t', '_', regex=False)
    results_df.columns = results_df.columns.str.replace('=', '_', regex=False)

    # 4. SAVE TO DELTA TABLE
    output_table = "workspace.default.test_model_predictions"
    spark_df = spark.createDataFrame(results_df)
    spark_df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(output_table)
    
    print(f"Success! Predictions saved to {output_table}")

if __name__ == "__main__":
    run_batch_inference()