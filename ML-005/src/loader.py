import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd

from utils import validator as vld

logger = logging.getLogger(__name__)


class DataLoader:
    VALIDATORS: ClassVar[dict[str, tuple[vld.Validator, ...]]] = {
        "load_data": (
            vld.not_empty,
            vld.check_required_columns(
                "Manufacturer",
                "Model",
                "Sales_in_thousands",
                "__year_resale_value",
                "Vehicle_type",
                "Price_in_thousands",
                "Engine_size",
                "Horsepower",
                "Wheelbase",
                "Width",
                "Length",
                "Curb_weight",
                "Fuel_capacity",
                "Fuel_efficiency",
                "Latest_Launch",
                "Power_perf_factor",
            ),
        ),
    }

    @classmethod
    def _validate(cls, method_name: str, data: pd.DataFrame) -> None:
        for validator in cls.VALIDATORS.get(method_name, ()):
            validator(data)

    @classmethod
    def load_data(cls, data_path: Path) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=data_path)
        logger.debug("Данные загружены в память. Выполняется валидация...")
        cls._validate("load_data", data)
        return data

    @staticmethod
    def save_data(
        data: pd.DataFrame, path: str = "./data/car_sales_normilized.csv"
    ) -> None:
        data.to_csv(path_or_buf=path, index=False)
        logger.debug("Данные сохранены по пути: %s", path)
