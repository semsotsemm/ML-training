import pandas as pd
from typing import ClassVar
from pathlib import Path

from utils import validator as vld


class DataLoader:
    VALIDATORS: ClassVar[dict[str, tuple[vld.Validator, ...]]] = {
        "load_data": (
            vld.not_empty,
            vld.check_required_columns("Manufacturer","Model","Sales_in_thousands","__year_resale_value","Vehicle_type","Price_in_thousands", "Engine_size","Horsepower","Wheelbase","Width","Length","Curb_weight","Fuel_capacity","Fuel_efficiency","Latest_Launch","Power_perf_factor")),
    }


    @classmethod
    def _validate(cls, method_name: str, data: pd.DataFrame) -> None:
        for validator in cls.VALIDATORS.get(method_name, ()):
            validator(data)


    @classmethod
    def load_data(cls, data_path: Path) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=data_path)
        cls._validate("load_data", data)
        return data