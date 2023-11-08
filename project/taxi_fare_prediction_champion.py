
from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, retry, timeout, project

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        from utils.preprocessing import clean_data

        self.df = clean_data(self.df)

        self.X = self.df["trip_distance"].to_numpy().reshape(-1, 1)
        self.y = self.df["total_amount"].to_numpy()

        self.next(self.train_linear_model)

    @timeout(minutes=5)
    @step
    def train_linear_model(self):
        "Train linear model."
        from sklearn.linear_model import LinearRegression

        self.model_name = "Linear Regression"

        self.model = LinearRegression()

        self.model.fit(self.X, self.y)

        self.next(self.validate)
    
    @timeout(minutes=5)
    @step
    def validate(self):
        "Do CV for linear model."
        import numpy as np
        from sklearn.model_selection import cross_val_score

        self.score = np.mean(cross_val_score(self.model, self.X, self.y, cv=5))

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print('Scores:')
        print(f"The CV R^2 of the linear model is {self.score:.2f}.")


if __name__ == "__main__":
    TaxiFarePrediction()
