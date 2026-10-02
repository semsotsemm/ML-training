import numpy as np


weights = np.array([0.12, 0.6, -0.02], dtype=np.float32)
bias = 0.5


def predict_price(x: np.ndarray) -> np.ndarray:
    prediction = x @ weights + bias
    return prediction


if __name__ == '__main__':
    x = np.array([
        [40, 1, 30],
        [60, 2, 10],
        [75, 3, 5],
        [50, 2, 20]],
        dtype=np.float32)
    y = np.array([5.0, 8.2, 11, 6.6], dtype=np.float32)
    new_flat = np.array([55, 2, 15], dtype=np.float32)

    print("Предсказание модели: %s" % str(predict_price(x)))
    print("Настоящие цены: %s" % str(y))
    print("Матрица ошибок: %s" % str(y - predict_price(x)))
    print("Предсказание модели для новой квартиры: $" % str(predict_price(new_flat)))
