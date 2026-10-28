import logging

from sympy import lowergamma

from loader import DataLoader
from preprocessing import DataPreprocessor
from utils.logger import configure_logger
from utils.validator import ValidationError
from train import DataTrainer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
data_path = "./data/Car_sales.csv"

TARGET = "Price_in_thousands"


def main():
    configure_logger(level=logging.INFO)
    loader = DataLoader()
    preprocessor = DataPreprocessor()

    try:
        data = loader.load_data(data_path)
        logger.info("Данные загружены в память. Датафрейм имеет форму: %s", data.shape)

        data = preprocessor.remove_duplicates(data)
        logger.info("Из датафрейма было удалены дублирующиеся строки")

        data = data.dropna(subset=[TARGET])
        logger.info("Из датафрейма удалены строки с не указанной ценой.")

        data = preprocessor.transform_date(data)
        logger.info("Дата последнего обслуживания преобразована в формат pandas")

        y = data[TARGET]
        x = data.drop(columns=[TARGET])
        trainer = DataTrainer(TARGET, x, y)
        x_train, x_test, y_train, y_test = trainer.get_train_and_test_df(
            random_state=42
        )

        preprocessor.fit(x_train)
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)
        logger.info(
            "Датафреймы для тренировки и тестирования очищены от пустых значений."
        )

        preprocessor.fit_encoder(x_train)
        x_train = preprocessor.category_encoding(x_train)
        x_test = preprocessor.category_encoding(x_test)
        logger.info(
            "Датафреймы для тренировки и тестирования разбиты на числовые признаки."
        )

        preprocessor.fit_scale(x_train)
        x_train = preprocessor.scale_data(x_train)
        x_test = preprocessor.scale_data(x_test)
        logger.info(
            "В датафреймах для тренировки и тестирования нормализованы признаки."
        )

        if logger.isEnabledFor(logging.DEBUG):
            loader.save_data(x_train)
            logger.info("Полученные данные сохранены.")

        (
            train_predictions,
            test_predictions,
            train_mae,
            test_mae,
            baseline_mae,
            comparison,
            model,
        ) = trainer.train_model(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
        logger.info("Обучение завершено.")
        logger.info("train_mae: %s", train_mae)
        logger.info("test_mae: %s", test_mae)
        # logger.info("train_prediction: %s", train_predictions)
        # logger.info("test_prediction: %s", test_predictions)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(["Train", "Test"], [train_mae, test_mae])
        axes[0].set_title("Средняя абсолютная ошибка")
        axes[0].set_ylabel("MAE, тыс. долларов")

        axes[1].scatter(y_test, test_predictions, alpha=0.7)

        low = min(y_test.min(), test_predictions.min())
        high = max(y_test.max(), test_predictions.max())

        axes[1].plot([low, high], [low, high], "--", color="black")
        axes[1].set_xlabel("Фактическая цена")
        axes[1].set_ylabel("Предсказанная цена")
        axes[1].set_title("Предсказание на test")

        plt.tight_layout()
        plt.show()

    except ValidationError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
