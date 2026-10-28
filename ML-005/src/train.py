import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor


class DataTrainer:
    def __init__(self, target, x_df, y_df):
        self.target_column_name = target
        self.X: pd.DataFrame = x_df
        self.Y: pd.DataFrame = y_df

    def get_train_and_test_df(self, random_state: int = 42):
        return train_test_split(
            self.X, self.Y, test_size=0.2, random_state=random_state
        )

    def train_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)

        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)

        baseline = DummyRegressor(strategy="median")
        baseline.fit(x_train, y_train)
        baseline_predictions = baseline.predict(x_test)
        baseline_mae = mean_absolute_error(y_test, baseline_predictions)

        comparison = pd.DataFrame(
            {"actual_price": y_test, "predicted_price": test_predictions}
        )

        return (
            train_predictions,
            test_predictions,
            train_mae,
            test_mae,
            baseline_mae,
            comparison,
            model,
        )
