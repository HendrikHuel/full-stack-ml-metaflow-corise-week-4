
from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, retry, catch, timeout
from metaflow.cards import Markdown, Table, Image, Artifact

import pandas as pd

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    obviously_bad_data_filters = [
        "fare_amount > 0",  # fare_amount in US Dollars
        "trip_distance <= 100",  # trip_distance in miles
        "trip_distance > 0",
        "passenger_count > 0",
        "tpep_pickup_datetime < tpep_dropoff_datetime",
        "tip_amount >= 0",
        "tolls_amount >= 0",
        "improvement_surcharge >= 0",
        "total_amount >= 0",
        "congestion_surcharge >= 0",
        "airport_fee >= 0",
        # TODO: add some logic to filter out what you decide is bad data!
        # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
    ]

    df = df.query(" & ".join(obviously_bad_data_filters))

    if len(df) == 0:
        raise ValueError("No entries remain after filtering.")

    return df

@project(name="taxi_fare_prediction")
@trigger(events=["s3"])
@conda_base(
libraries={
    "pandas": "2.1.2",  # bump version
    "pyarrow": "13.0.0", # bump version
    #"numpy": "1.21.2",  # omit defining numpy since pandas comes with it
    "scikit-learn": "1.3.2", # bump version
}
)
class TaxiFarePrediction(FlowSpec):
data_url = Parameter("data_url", default=URL)

@retry(times=3, minutes_between_retries=1)
@step
def start(self):
    """Read data seperately to allow retries."""
    import pandas as pd

    self.df = pd.read_parquet(self.data_url)

    self.next(self.transform_features)

@step
def transform_features(self):
    """Clean data."""

    self.df = clean_data(self.df)

    self.X = self.df["trip_distance"].values.reshape(-1, 1)
    self.y = self.df["total_amount"].values

    self.next(self.train_linear_model)

@timeout(minutes=5)
@step
def train_linear_model(self):
    "Train linear model."
    from sklearn.linear_model import LinearRegression

    self.model = LinearRegression()

    self.model.fit(self.X, self.y)

    self.next(self.predict)
    
def predict(self):
    "Do insample prediction."
    from sklearn.metrics import mean_absolute_error

    self.y_hat = self.model.predict(self.X)
    self.score = mean_absolute_error(self.y, self.y_hat)

@step
def end(self):
    """
    End of flow!
    """
    print('Scores:')
    print(f"The insample MAE of the linear model is {self.score:.2f}.")


if __name__ == "__main__":
    TaxiFarePrediction()
