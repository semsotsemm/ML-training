import logging

from loader import DataLoader
from utils.logger import configure_logger
from utils.validator import ValidationError
from preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)
data_path = "./data/Car_sales.csv"

def main():
    configure_logger(level=logging.INFO)
    loader = DataLoader()
    preprocessor = DataPreprocessor()

    try:
        data = loader.load_data(data_path)
        logger.info("Данные загружены в память. Датафрейм имеет форму: %s", data.shape)

        data = preprocessor.fill_pass(data)
        logger.info("Датафрейм очищен от пустых значений.")

        removed_duplicates = preprocessor.remove_duplicates(data)
        logger.info("Из датафрейма было удалено %s дублирующихся строк", removed_duplicates)

        data = preprocessor.category_encoding(data)
        logger.info("Категории разбиты на числовые признаки.")

        data = preprocessor.scale_data(data)
        logger.info("Признаки нормализованы.")
    except ValidationError as e:
        logger.error(e)

if __name__ == "__main__":
    main()