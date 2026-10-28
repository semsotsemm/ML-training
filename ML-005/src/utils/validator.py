import pandas as pd
from collections.abc import Callable


class ValidationError(ValueError):
    pass

Validator = Callable[[pd.DataFrame], None]


def not_empty(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValidationError("Ошибка 21: Датасет пуст: набор данных не содержит записей.")


def check_required_columns(*required_columns: str) -> Validator:
    def validate(data: pd.DataFrame) -> None:
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValidationError(f"Ошибка 22: Отсутствуют обязательные признаки: {missing_columns}.")

    return validate
