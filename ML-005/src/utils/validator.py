import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    pass


Validator = Callable[[pd.DataFrame], None]


def not_empty(data: pd.DataFrame) -> None:
    logger.debug("Датасет проверяется на отсутствие записей")
    if data.empty:
        raise ValidationError(
            "Ошибка 21: Датасет пуст: набор данных не содержит записей."
        )


def check_required_columns(*required_columns: str) -> Validator:
    def validate(data: pd.DataFrame) -> None:
        logger.debug("Датасет проверяется на наличие обязательных признаков.")
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValidationError(
                f"Ошибка 22: Отсутствуют обязательные признаки: {missing_columns}."
            )

    return validate
