import pandas as pd
import random


class DataGenerator:
    def __init__(self):
        self._email_chars = "qwertyuiopasdfghjklzxcvbnm"
        self._available_names = ["Алексей", "Иван", "Дмитрий", "Михаил", "Кирилл", "Егор", "Артем"]
        self._available_cities = ["Минск", "Гродно", "Новогрудок", "Алматы", "Питер", "Москва"]
        self._products_list = [
            {
                "product_name" : "Iphone 15 PRO",
                "product_price" : 3500
            },
            {
                "product_name" : "Iphone 14",
                "product_price" : 2000
            }
        ]


    def _generate_email(self, min_length:int = 14, max_length:int = 20) -> str:
        email = ""
        email_length = random.randint(min_length, max_length)

        for _ in range(email_length):
            email += random.choice(self._email_chars)

        return email + "@gmail.com"


    def generate_data(self) -> pd.DataFrame:
        data = []

        for _ in range(10):
            new_user = {
                "Имя пользователя" : random.choice(self._available_names),
                "Город проживания" : random.choice(self._available_cities),
                "Email" : self._generate_email(),
                "Список покупок":[random.choice(self._products_list) for _ in range(5)]
            }
            data["Пользователи"].append(new_user)

        return pd.DataFrame(data)


def main():
    data_generator = DataGenerator()
    data = data_generator.generate_data()
    print(data.head(2))
    data.to_csv("data.csv", index = False)
    new_data = [{"id": 1, "Имя пользователя": "11", "Возраст": 2},{"id": 1, "Имя пользователя": "11", "Возраст": 2}]
    df = pd.DataFrame(new_data)


if __name__ == '__main__':
    main()