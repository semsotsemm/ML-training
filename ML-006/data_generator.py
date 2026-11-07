import pandas as pd
import numpy as np


np.random.seed(42)
count_of_wells = 800
file_path = "./data/raw/"


def generate_wells_info():
    industrial_zone_x = np.random.normal(loc=1000, scale=300, size=500)
    industrial_zone_y = np.random.normal(loc=1000, scale=300, size=500)

    water_fence_x = np.random.normal(loc=4000, scale=400, size=200)
    water_fence_y = np.random.normal(loc=3500, scale=400, size=200)

    random_wells_x = np.random.uniform(low=0, high=5000, size=100)
    random_wells_y = np.random.uniform(low=0, high=5000, size=100)

    x = np.concatenate([industrial_zone_x, water_fence_x, random_wells_x])
    y = np.concatenate([industrial_zone_y, water_fence_y, random_wells_y])
    z = np.random.uniform(low=-150, high=-5, size=800)

    wells_index = [f"Well_{i:04d}" for i in range(1, count_of_wells + 1)]

    df = pd.DataFrame(
        {"well_id": wells_index, "x": x.round(2), "y": y.round(2), "z": z.round(2)}
    )

    error_x_index = np.random.choice(df.index, size=30, replace=False)
    df.loc[error_x_index, "x"] = -999.0

    error_z_index = np.random.choice(df.index, size=50, replace=False)
    df.loc[error_z_index, "z"] = np.random.uniform(low=5, high=150, size=50)

    nan_y_index = np.random.choice(df.index, size=10, replace=False)
    df.loc[nan_y_index, "y"] = np.nan

    nan_z_index = np.random.choice(df.index, size=5, replace=False)
    df.loc[nan_z_index, "z"] = np.nan

    duplicate_rows_ids = np.random.choice(df.index, size=30, replace=False)
    duplicate_df = pd.DataFrame(
        {
            "well_id": (f"Well_{i:04d}" for i in range(801, 831)),
            "x": df.loc[duplicate_rows_ids, "x"],
            "y": df.loc[duplicate_rows_ids, "y"],
            "z": df.loc[duplicate_rows_ids, "z"],
        }
    )
    df = pd.concat([df, duplicate_df], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(f"{file_path}/wells.csv", index=False)
    print("Wells.csv successfully generated!")


def generate_concentration_info():
    count_of_all_wells = 830
    count_of_measurement = 2400
    file_name = "concentration.csv"
    count_of_random_wells = count_of_measurement - count_of_all_wells

    base_wells = [f"Well_{i:04d}" for i in range(1, count_of_all_wells + 1)]
    extra_wells = np.random.choice(base_wells, size=count_of_random_wells, replace=True)
    all_wells = np.concatenate([base_wells, extra_wells])

    all_possible_date = pd.date_range(start="2020-01-01", end="2025-12-31")
    random_dates = np.random.choice(
        all_possible_date.strftime("%d.%m.%Y"), size=count_of_measurement, replace=True
    )

    df = pd.DataFrame({"well_id": all_wells, "t": random_dates, "concentration": 0.0})

    df["well_num"] = df["well_id"].str.replace("Well_", "").astype(int)

    is_industrial = df["well_num"] <= 500
    is_water_fence = (df["well_num"] > 500) & (df["well_num"] <= 700)
    is_forest = df["well_num"] > 700

    df.loc[is_industrial, "concentration"] = np.random.lognormal(
        mean=4.0, sigma=1.0, size=is_industrial.sum()
    )
    df.loc[is_water_fence, "concentration"] = np.random.lognormal(
        mean=1.0, sigma=0.5, size=is_water_fence.sum()
    )
    forest_spikes = np.random.exponential(scale=2.0, size=is_forest.sum())
    forest_zeros = np.random.choice([0, 1], p=[0.95, 0.05], size=is_forest.sum())
    df.loc[is_forest, "concentration"] = forest_spikes * forest_zeros

    df["concentration"] = df["concentration"].round(2)
    df["concentration"] = df["concentration"].astype(object)

    text_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[text_idx[15:], "concentration"] = "Error"
    df.loc[text_idx[:15], "concentration"] = "<0.05"

    valid_idx = df.index.difference(text_idx)
    negative_idx = np.random.choice(valid_idx, size=45, replace=False)
    df.loc[negative_idx, "concentration"] = np.random.randint(
        low=-100.0, high=-1.0, size=45
    )

    valid_idx = df.index.difference(negative_idx)
    nan_concentration_idx = np.random.choice(valid_idx, size=30, replace=False)
    df.loc[nan_concentration_idx, "concentration"] = np.nan

    nan_date_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[nan_date_idx, "t"] = np.nan

    dup_source = df.sample(n=40, random_state=42)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    dup_source.loc[dup_source.index[:20], "concentration"] = 999.99
    df = pd.concat([df, dup_source], ignore_index=True)

    df = df.drop(columns=["well_num"])

    df.to_csv(file_path + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_head():
    file_name = "head.csv"
    list_available_date = pd.date_range(start="2020-01-01", end="2025-12-31")
    random_date = np.random.choice(
        list_available_date.strftime("%d.%m.%Y"), size=830, replace=True
    )

    head = np.random.normal(loc=30, scale=5, size=830)

    df = pd.DataFrame(
        {
            "well_id": [f"Well_{i:04d}" for i in range(1, 831)],
            "t": random_date,
            "head": head.round(2),
        }
    )

    duplicates = df.sample(n=20, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    nan_idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "t"] = np.nan

    nan_idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "head"] = np.nan

    incorrect_value_idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[incorrect_value_idx, "head"] = np.random.randint(
        low=-100.0, high=-1.0, size=10
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(file_path + file_name, index=False)


def generate_cores():
    file_name = "cores.csv"
    available_lithology_list = ["Sand", "Clay", "Gravel", "Loam"]
    lithology_list = []
    z_top_list = []
    z_bottom_list = []
    list_well = []

    for i in range(1, 831):
        z_height = 0
        count_of_lithologies = int(np.random.randint(low=3, high=7))

        for _ in range(count_of_lithologies):
            lithology_list.append(np.random.choice(available_lithology_list))

            weight_lithology = np.random.randint(low=-40, high=-10)
            z_top_list.append(z_height)
            z_bottom_list.append(z_height + weight_lithology)

            z_height += weight_lithology

            list_well.append(f"Well_{i:04d}")

    df = pd.DataFrame(
        {
            "well_id": list_well,
            "z_top": z_top_list,
            "z_bottom": z_bottom_list,
            "lithology": lithology_list,
        }
    )

    error_values_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[error_values_idx, "z_top"] = np.random.randint(low=10, high=40, size=30)

    nan_values_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[nan_values_idx, "z_bottom"] = np.nan

    duplicates = df.sample(n=30, random_state=42)
    pd.concat([df, duplicates], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(file_path + file_name, index=False)


def main():
    generate_cores()


if __name__ == "__main__":
    main()
