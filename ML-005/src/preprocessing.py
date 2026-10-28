import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.rules = {}
        self.scale_rules = {}

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.category_columns = {}

    def fit_encoder(self, data: pd.DataFrame) -> None:
        self.category_columns = data.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        if self.category_columns:
            self.encoder.fit(data[self.category_columns])

    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        logger.debug("Выполняется удаление повторяющихся значений.")
        df.drop_duplicates(inplace=True)

        return df

    def fit(self, data: pd.DataFrame) -> None:
        for column in data.columns:
            logger.debug("Вычисляется правило для колонки: %s", column)
            if data[column].isna().mean() > 0.9:
                logger.debug(
                    "В колонке %s слишком много пропусков, она удаляется", column
                )
                self.rules[column] = {"action": "drop"}
                continue

            if pd.api.types.is_numeric_dtype(data[column]):
                logger.debug(
                    "Для колонки %s подсчитывается среднее значение и дисперсия", column
                )
                mean_value = data[column].mean()
                dispersion_value = data[column].std()
                self.rules[column] = {
                    "action": "generate_average",
                    "mean": mean_value,
                    "dispersion": dispersion_value,
                }
                continue

            if pd.api.types.is_object_dtype(data[column]):
                logger.debug(
                    "Для колонки %s подсчитываются вероятности появления значений.",
                    column,
                )
                probabilities = data[column].value_counts(normalize=True)

                self.rules[column] = {
                    "action": "generate_random",
                    "classes": probabilities.index.tolist(),
                    "probabilities": probabilities.values.tolist(),
                }
                continue

    @staticmethod
    def transform_date(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["Last_Launch"] = pd.to_datetime(df["Latest_Launch"], errors="coerce")
        logger.debug("Дата последнего обслуживания преобразована в формат pandas")

        df["Launch_Year"] = df["Last_Launch"].dt.year
        df["Launch_Month"] = df["Last_Launch"].dt.month

        df = df.drop(columns=["Last_Launch", "Latest_Launch"])

        logger.debug(
            "Дата последнего обслуживания разбита на отдельные колонки год и месяц."
        )

        return df

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        output_df = data.copy()

        for column, rule in self.rules.items():
            if rule["action"] == "drop":
                output_df = output_df.drop(columns=[column], errors="ignore")
                continue

            missing_mask = output_df[column].isna()
            missing_count = missing_mask.sum()

            if missing_count == 0:
                continue

            if rule["action"] == "generate_average":
                random_fills = np.random.normal(
                    loc=rule["mean"], scale=rule["dispersion"], size=missing_count
                )
                output_df.loc[missing_mask, column] = random_fills

            elif rule["action"] == "generate_random":
                random_fills = np.random.choice(
                    rule["classes"], p=rule["probabilities"], size=missing_count
                )
                output_df.loc[missing_mask, column] = random_fills

        return output_df

    def category_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.category_columns:
            return data.copy()

        encoded = self.encoder.transform(data[self.category_columns])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(self.category_columns),
            index=data.index,
        )

        remaining_df = data.drop(columns=self.category_columns)

        return pd.concat(
            [remaining_df, encoded_df],
            axis=1,
        )

    def _calculate_scale_rules(self, data: pd.DataFrame) -> None:
        numeric_columns = data.select_dtypes(include="number").columns

        for column in numeric_columns:
            self.scale_rules[column] = {
                "mean": data[column].mean(),
                "dispersion": data[column].std(),
            }

    def fit_scale(self, data: pd.DataFrame) -> None:
        self.scale_rules.clear()
        self._calculate_scale_rules(data)

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        output_df = data.copy()

        for column, rule in self.scale_rules.items():
            if column in output_df.columns:
                mean = rule["mean"]
                dispersion = rule["dispersion"]

                if dispersion == 0:
                    output_df[column] = 0
                else:
                    output_df[column] = (output_df[column] - mean) / dispersion

        return output_df
