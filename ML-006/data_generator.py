import pandas as pd
import numpy as np

np.random.seed(42)
count_of_wells = 800

def generate_data():
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
        {
            "well_id": wells_index,
            "x": x.round(2),
            "y": y.round(2),
            "z": z.round(2)
        }
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
            "x" : df.loc[duplicate_rows_ids, "x"],
            "y" : df.loc[duplicate_rows_ids, "y"],
            "z" : df.loc[duplicate_rows_ids, "z"]
        }
    )
    df = pd.concat([df, duplicate_df], ignore_index=True)


    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    df.to_csv("data/interim/wells.csv", index=False)
    print("Wells.csv successfully generated!")

if __name__ == "__main__":
    generate_data()