import pandas as pd
import io

def load_data() -> pd.DataFrame:
    csv_data = """id,name,age,department,salary,is_manager,notes
    1,Иван,34,IT,150000.0,False,Отличный работник
    2,Мария,,HR,120000.0,True,
    3,Петр,28,IT,,False,В отпуске
    4,Анна,45,Finance,200000.0,True,
    5,Алексей,31,,130000.0,False,Испытательный срок"""

    virtual_data = io.StringIO(csv_data) # Создание проекции данных в оперативной памяти
    return pd.read_csv(filepath_or_buffer=virtual_data)


def get_null_rows_proportion(df: pd.DataFrame) -> float:
    is_null_rows = df.isnull() # Получаем true/false матрицу null-ов
    return is_null_rows.any(axis=1).mean()

def get_columns_type_proportion(df: pd.DataFrame) -> dict:
    count_of_value = df.dtypes.value_counts()
    total_sum = count_of_value.sum()
    proportion = {}
    for dtype_name, count in count_of_value.items():
        proportion[dtype_name] = count / total_sum
    return proportion

def main():
    data = load_data()
    print(f"{'Доля каждого типа данных в исходном датафрейме:':=^100}")
    proportion_of_datatype = get_columns_type_proportion(data)
    for datatype_name, count in proportion_of_datatype.items():
        print(f"{datatype_name}: {count:.1%}")
    print(f"Доля строк, в которых хотя бы 1 значение NaN: {get_null_rows_proportion(data):.1%}")
    print(get_columns_type_proportion(data))
    print(f"{' Типы данных всех колонок (при нескольких типах данных в одной колонке obj): ':=^100}\n{data.dtypes}\n\n")
    print(f"{' Количество NaN значений для каждой колонки: ':=^100}\n{data.isnull().sum()}\n\n")
    print(f"{' Общая информация по таблице: ':=^100}\n")
    data.info()

if __name__ == "__main__":
    main()