import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.rules = {}
        self.scale_rules = {}

    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> int:
        duplicates_count = data.duplicated().sum()

        logger.debug("Выполняется удаление повторяющихся значений.")
        data.drop_duplicates(inplace=True)
        return duplicates_count

    @staticmethod
    def remove_pass(data: pd.DataFrame) -> int:
        none_values_count = data.isna().sum().sum()
        return none_values_count

    def fill_pass(self, data: pd.DataFrame) -> pd.DataFrame:
        output_df = data.copy()
        self._calculate_rules(data)

        for column, rule in self.rules.items():
            if rule == 'prop':
                output_df = output_df.drop(column=[column], errors="ignore")
                continue

            missing_mask = output_df[column].isna()
            missing_count = missing_mask.sum()

            if missing_count == 0:
                continue

            if (rule == "generate_average"):
                random_fills = np.random.normal(
                    loc = rule["mean"],
                    scale = rule["dispersion"],
                    size = missing_count
                )
                output_df.loc[missing_mask, column] = random_fills

            elif(rule == "generate_random"):
                random_fills = np.random.choice(
                    rule["classes"],
                    p=rule["probabilities"],
                    size = missing_count
                )
                output_df.loc[missing_mask, column] = random_fills

        return output_df

    def _calculate_rules(self, data: pd.DataFrame) -> None:
        for column in data.columns:
            logger.debug("Вычисляется правило для колонки: %s", column)
            if data[column].isna().mean() > 0.9:
                logger.debug("В колонке %s слишком много пропусков, она удаляется", column)
                self.rules[column] = {"action": "drop"}
                continue

            if pd.api.types.is_numeric_dtype(data[column]):
                logger.debug("Для колонки %s подсчитывается среднее значение и дисперсия", column)
                mean_value = data[column].mean()
                dispersion_value = data[column].std()
                self.rules[column] = {
                    "action": "generate_average",
                    "mean": mean_value,
                    "dispersion": dispersion_value
                }
                continue

            if pd.api.types.is_object_dtype(data[column]):
                logger.debug("Для колонки %s подсчитываются вероятности появления значений.", column)
                probabilities = data[column].value_counts(normalize=True)

                self.rules[column] = {
                    "action": "generate_random",
                    "classes": probabilities.index.tolist(),
                    "probabilities": probabilities.values.tolist()
                }
                continue

    @staticmethod
    def category_encoding(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["Last_Launch"] = pd.to_datetime(df["Latest_Launch"], errors="coerce")
        logger.debug("Дата последнего обслуживания преобразована в формат pandas")

        df["Launch_Year"] = df["Last_Launch"].dt.year
        df["Launch_Month"] = df["Last_Launch"].dt.month
        df = df.drop(columns=["Last_Launch"])
        logger.debug("Дата последнего обслуживания разбита на отдельные колонки год и месяц.")

        category_columns = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=category_columns, drop_first=True)
        logger.debug("Категорийные признаки разбиты на множество конлонок и преобразованы в числовые значения.")

        return df

    def _calculate_scale_rules(self, data: pd.DataFrame) -> None:
        numeric_columns = data.select_dtypes(include="number").columns

        for column in numeric_columns:
            self.scale_rules[column] = {
                "mean": data[column].mean(),
                "dispersion" : data[column].std()
            }

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        output_df = data.copy()

        for column, rule in self.scale_rules.items():
            if column in output_df.columns:
                mean = rules["mean"]
                dispersion = rules["dispersion"]

                if dispersion == 0:
                    output_df[column] = 0
                else:
                    output_df[column] = (output_df[column] - mean) / dispersion

        return output_df