import logging

from loader import DataLoader
from utils.logger import configure_logger
from utils.validator import ValidationError

logger = logging.getLogger(__name__)
data_path = "./data/Car_sales.csv"

def main():
    configure_logger(level=logging.DEBUG)
    loader = DataLoader()
    try:
        data = loader.load_data(data_path)
        logger.debug("Данные загружены в память")
    except ValidationError as e:
        logger.error(e)

if __name__ == "__main__":
    main()