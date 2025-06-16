import pandas as pd
import numpy as np
import pickle

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


df = read_data(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
)


dicts = df[categorical].to_dict(orient="records")
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(np.std(y_pred))


# lets make the year and month to be passed through the command line
import sys

if len(sys.argv) != 3:
    print("Usage: python starter.py <year> <month>")
    sys.exit(1)
year = int(sys.argv[1])
month = int(sys.argv[2])
df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
df["prediction"] = y_pred

print(f"Mean prediction: {np.mean(y_pred):.2f} minutes")
df[["ride_id", "prediction"]].to_parquet(
    f"predictions_{year:04d}_{month:02d}_v2.parquet",
    engine="pyarrow",
    index=False,
)
