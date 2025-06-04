from zenml import step, pipeline
import pandas as pd
import numpy as np
from typing import Tuple, List, Any
import mlflow
from mlflow.models import infer_signature
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy.sparse import spmatrix
from loguru import logger


@step
def load_data(path: str) -> None:
    """Load data from a CSV file."""
    df = pd.read_parquet(path)
    logger.info(
        f"Data loaded from {path} with {len(df)} rows and {len(df.columns)} columns."
    )


@step
def read_dataframe(filename: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical_values = ["PULocationID", "DOLocationID"]
    df[categorical_values] = df[categorical_values].astype(str)
    logger.info(
        f"Data read from {filename} with {len(df)} rows and {len(df.columns)} columns."
    )

    return df, categorical_values


@step
def instantiate_dict_vectorizer() -> DictVectorizer:
    """Instantiate a DictVectorizer."""
    dv = DictVectorizer()
    print("DictVectorizer instantiated.")
    return dv


@step
def get_target_variable(df: pd.DataFrame) -> np.ndarray:
    """Extract the target variable from the DataFrame."""
    return df["duration"].values  # type: ignore


@step
def preprocess(
    df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False
) -> Tuple[np.ndarray[Any, Any] | spmatrix, DictVectorizer]:
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)  # type: ignore
    return X, dv  # type: ignore


@step
def train_model(X_train, y_train) -> None:
    """Train the model and track the run with MLflow."""
    with mlflow.start_run():
        model = LinearRegression()
        mlflow.log_params(model.get_params())
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="linear_regression_model",
        )
        # Print the intercept
        logger.info(f"Model intercept: {model.intercept_:.2f}")


@pipeline
def main():
    load_data(
        "/Users/thobelasixpence/Documents/mlops-zoomcamp-course-2025/mlops-zoomcamp-hw-2025/homeworks/hw-03/data/yellow_tripdata_2023-03.parquet"
    )
    # Read and preprocess the dataframe
    df, categorical_values = read_dataframe(
        "/Users/thobelasixpence/Documents/mlops-zoomcamp-course-2025/mlops-zoomcamp-hw-2025/homeworks/hw-03/data/yellow_tripdata_2023-03.parquet"
    )

    # Initialize DictVectorizer
    dv = instantiate_dict_vectorizer()

    # Define target variable
    y = get_target_variable(df)
    # Preprocess the data
    X, dv = preprocess(df, dv, fit_dv=True)

    # Train the model
    train_model(X, y)


if __name__ == "__main__":
    main()
