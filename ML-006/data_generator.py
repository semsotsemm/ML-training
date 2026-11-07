import os
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter

np.random.seed(42)

COUNT_OF_WELLS = 800
COUNT_OF_DUPLICATES = 40
TOTAL_WELLS = COUNT_OF_WELLS + COUNT_OF_DUPLICATES  # 840
FILE_PATH = "./data/raw/"

os.makedirs(FILE_PATH, exist_ok=True)


def _true_concentration_field(x, y, z, days_since_2020):
    dx_total, dy_total = 3000.0, 2500.0
    dist_total = np.hypot(dx_total, dy_total)
    ux, uy = dx_total / dist_total, dy_total / dist_total  # Единичный вектор потока

    v_front = 1.15
    plume_center_x = 1000.0 + ux * v_front * days_since_2020
    plume_center_y = 1000.0 + uy * v_front * days_since_2020

    rel_x = x - plume_center_x
    rel_y = y - plume_center_y
    longitudinal = rel_x * ux + rel_y * uy
    transverse = -rel_x * uy + rel_y * ux

    sigma_long = 350.0 + 0.18 * days_since_2020
    sigma_trans = 180.0 + 0.06 * days_since_2020
    sigma_z = 25.0

    z_core = -45.0

    plume_moving = 650.0 * np.exp(
        -0.5 * (
            (longitudinal / sigma_long) ** 2
            + (transverse / sigma_trans) ** 2
            + ((z - z_core) / sigma_z) ** 2
        )
    )
    source_halo = 1200.0 * np.exp(
        -0.5 * (
            ((x - 1000.0) / 280.0) ** 2
            + ((y - 1000.0) / 280.0) ** 2
            + ((z + 35.0) / 20.0) ** 2
        )
    )

    return plume_moving + source_halo


def generate_wells_info():
    file_name = "wells.csv"

    industrial_x = np.random.normal(loc=1000, scale=300, size=500)
    industrial_y = np.random.normal(loc=1000, scale=300, size=500)

    water_fence_x = np.random.normal(loc=4000, scale=400, size=200)
    water_fence_y = np.random.normal(loc=3500, scale=400, size=200)

    forest_x = np.random.uniform(low=0, high=5000, size=100)
    forest_y = np.random.uniform(low=0, high=5000, size=100)

    x = np.concatenate([industrial_x, water_fence_x, forest_x])
    y = np.concatenate([industrial_y, water_fence_y, forest_y])
    z = np.random.uniform(low=-140, high=-10, size=COUNT_OF_WELLS)

    wells_index = [f"Well_{i:04d}" for i in range(1, COUNT_OF_WELLS + 1)]

    dup_indices = np.random.choice(COUNT_OF_WELLS, size=COUNT_OF_DUPLICATES, replace=False)
    dup_wells_index = [f"Well_{i:04d}" for i in range(COUNT_OF_WELLS + 1, TOTAL_WELLS + 1)]
    dup_x = x[dup_indices] + np.random.normal(0, 0.02, size=COUNT_OF_DUPLICATES)
    dup_y = y[dup_indices] + np.random.normal(0, 0.02, size=COUNT_OF_DUPLICATES)
    dup_z = z[dup_indices]

    all_wells = wells_index + dup_wells_index
    all_x = np.concatenate([x, dup_x])
    all_y = np.concatenate([y, dup_y])
    all_z = np.concatenate([z, dup_z])

    clean_coords = pd.DataFrame({
        "well_id": all_wells,
        "x": all_x,
        "y": all_y,
        "z": all_z
    }).set_index("well_id")

    df = pd.DataFrame({
        "well_id": all_wells,
        "x": all_x.round(2),
        "y": all_y.round(2),
        "z": all_z.round(2),
    })

    error_x_idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[error_x_idx, "x"] = -999.0

    error_z_idx = np.random.choice(df.index, size=45, replace=False)
    df.loc[error_z_idx, "z"] = df.loc[error_z_idx, "z"].abs()

    nan_y_idx = np.random.choice(df.index, size=12, replace=False)
    df.loc[nan_y_idx, "y"] = np.nan

    nan_z_idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[nan_z_idx, "z"] = np.nan

    exact_dups = df.sample(n=15, random_state=42)
    df = pd.concat([df, exact_dups], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated! (Total unique IDs: {TOTAL_WELLS})")
    return clean_coords


def generate_sources():
    file_name = "sources.csv"
    df = pd.DataFrame({
        "source_id": ["SRC_01", "SRC_02", "SRC_03", "SRC_04", "SRC_05"],
        "name": [
            "Цех обезжиривания №1",
            "Старый шламонакопитель",
            "Склад растворителей",
            "Дренажный коллектор Юг",
            "Аварийный сброс 2021",
        ],
        "x": [950.0, 1120.5, 1030.0, 890.0, 1450.0],
        "y": [1020.0, 940.0, 1100.0, 850.0, 1300.0],
        "z_top": [-15.0, -5.0, -10.0, -20.0, -8.0],
        "z_bottom": [-45.0, -30.0, -25.0, -55.0, -20.0],
        "start_date": ["01.01.2020", "15.03.2020", "10.06.2020", "2020-02-01", "12.08.2021"],
        "end_date": [np.nan, "31.12.2023", np.nan, np.nan, "15.09.2022"],
        "estimated_release_rate_g_day": [450.0, 320.0, 180.0, -999.0, 850.0],  # -999.0 как пропуск
    })
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_permeability():
    file_name = "permeability.tif"
    grid_size = 500

    raw_noise = np.random.normal(loc=0.0, scale=1.8, size=(grid_size, grid_size))
    smoothed_noise = gaussian_filter(raw_noise, sigma=18)

    base_permeability = 4.5
    perm_matrix = base_permeability * np.exp(smoothed_noise)

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    channel_dist = np.abs((xx - 100) * 250 - (yy - 100) * 300) / np.hypot(250, 300)
    perm_matrix += 12.0 * np.exp(-0.5 * (channel_dist / 25.0) ** 2)

    rock_mask = gaussian_filter(np.random.normal(size=(grid_size, grid_size)), sigma=12)
    perm_matrix[rock_mask < -1.4] = 0.005

    perm_matrix[0:5, 0:5] = -9999.0
    perm_matrix[490:493, 490:493] = np.nan

    perm_matrix = perm_matrix.astype(np.float32)
    tifffile.imwrite(FILE_PATH + file_name, perm_matrix)
    print(f"{file_name} successfully generated!")


def generate_concentration_info(clean_coords):
    file_name = "concentration.csv"
    count_of_measurement = 2800

    base_wells = [f"Well_{i:04d}" for i in range(1, TOTAL_WELLS + 1)]
    extra_wells = np.random.choice(base_wells, size=count_of_measurement - TOTAL_WELLS, replace=True)
    all_wells = np.concatenate([base_wells, extra_wells])

    all_dates = pd.date_range(start="2020-01-01", end="2025-10-31")
    chosen_dates = np.random.choice(all_dates, size=count_of_measurement, replace=True)
    days_since_2020 = (pd.to_datetime(chosen_dates) - pd.Timestamp("2020-01-01")).days.values

    wx = clean_coords.loc[all_wells, "x"].values
    wy = clean_coords.loc[all_wells, "y"].values
    wz = clean_coords.loc[all_wells, "z"].values

    true_conc = _true_concentration_field(wx, wy, wz, days_since_2020)
    noisy_conc = true_conc * np.random.lognormal(mean=0.0, sigma=0.35, size=count_of_measurement)
    bg_noise = np.random.exponential(scale=0.15, size=count_of_measurement)
    final_conc = (noisy_conc + bg_noise).round(2)

    formatted_dates = pd.to_datetime(chosen_dates).strftime("%d.%m.%Y").values

    iso_idx = np.random.choice(count_of_measurement, size=40, replace=False)
    formatted_dates[iso_idx] = pd.to_datetime(chosen_dates[iso_idx]).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "well_id": all_wells,
        "t": formatted_dates,
        "concentration": final_conc.astype(object),
    })

    low_conc_mask = df.index[final_conc < 0.08]
    if len(low_conc_mask) > 0:
        censor_idx = np.random.choice(low_conc_mask, size=min(60, len(low_conc_mask)), replace=False)
        df.loc[censor_idx[:35], "concentration"] = "<0.05"
        df.loc[censor_idx[35:], "concentration"] = "ND"

    text_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[text_idx, "concentration"] = "Error"

    valid_idx = df.index.difference(text_idx)
    neg_idx = np.random.choice(valid_idx, size=35, replace=False)
    df.loc[neg_idx, "concentration"] = np.random.uniform(-50.0, -0.5, size=35).round(2)

    valid_idx = valid_idx.difference(neg_idx)
    nan_c_idx = np.random.choice(valid_idx, size=30, replace=False)
    df.loc[nan_c_idx, "concentration"] = np.nan

    nan_t_idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[nan_t_idx, "t"] = np.nan

    dup_source = df.sample(n=45, random_state=42).copy()
    dup_source.iloc[:20, dup_source.columns.get_loc("concentration")] = 999.99
    df = pd.concat([df, dup_source], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_head(clean_coords):
    file_name = "head.csv"
    count_records = 1600

    base_wells = [f"Well_{i:04d}" for i in range(1, TOTAL_WELLS + 1)]
    sampled_wells = np.random.choice(base_wells, size=count_records, replace=True)

    dates = pd.date_range(start="2020-01-01", end="2025-10-31")
    random_dates = np.random.choice(dates.strftime("%d.%m.%Y"), size=count_records, replace=True)

    wx = clean_coords.loc[sampled_wells, "x"].values
    wy = clean_coords.loc[sampled_wells, "y"].values

    dist_to_intake = np.hypot(wx - 4000.0, wy - 3500.0)
    cone_of_depression = -6.5 * np.exp(-dist_to_intake / 600.0)
    head = 42.0 - 0.0035 * wx - 0.0025 * wy + cone_of_depression + np.random.normal(0, 0.8, size=count_records)

    df = pd.DataFrame({
        "well_id": sampled_wells,
        "t": random_dates,
        "head": head.round(2),
    })

    duplicates = df.sample(n=25, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    df.loc[np.random.choice(df.index, size=15, replace=False), "t"] = np.nan
    df.loc[np.random.choice(df.index, size=15, replace=False), "head"] = np.nan
    df.loc[np.random.choice(df.index, size=12, replace=False), "head"] = np.random.uniform(-100.0, -5.0, size=12).round(2)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_cores():
    file_name = "cores.csv"
    available_lithology = ["Sand", "Clay", "Gravel", "Loam"]
    lithology_list, z_top_list, z_bottom_list, list_well = [], [], [], []

    for i in range(1, TOTAL_WELLS + 1):
        z_height = 0.0
        count_layers = int(np.random.randint(low=3, high=7))
        for _ in range(count_layers):
            lithology_list.append(np.random.choice(available_lithology, p=[0.4, 0.25, 0.15, 0.2]))
            thickness = int(np.random.randint(low=-40, high=-10))
            z_top_list.append(z_height)
            z_bottom_list.append(z_height + thickness)
            z_height += thickness
            list_well.append(f"Well_{i:04d}")

    df = pd.DataFrame({
        "well_id": list_well,
        "z_top": z_top_list,
        "z_bottom": z_bottom_list,
        "lithology": lithology_list,
    })

    typo_idx = np.random.choice(df.index, size=40, replace=False)
    df.loc[typo_idx[:15], "lithology"] = "Caly"
    df.loc[typo_idx[15:30], "lithology"] = "sand "
    df.loc[typo_idx[30:], "lithology"] = "SAND"

    err_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[err_idx, "z_top"] = np.random.randint(low=10, high=40, size=30)

    df.loc[np.random.choice(df.index, size=20, replace=False), "z_bottom"] = np.nan
    df.loc[np.random.choice(df.index, size=15, replace=False), "lithology"] = np.nan

    duplicates = df.sample(n=30, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_pumping_and_abstraction():
    pumping_wells = [f"Well_{i:04d}" for i in range(501, 531)] + [f"Well_{i:04d}" for i in range(1, 16)]
    months = pd.date_range(start="2020-01-01", end="2025-12-01", freq="MS")

    records = []
    for w in pumping_wells:
        is_intake = int(w.replace("Well_", "")) > 500
        base_rate = 1200.0 if is_intake else 350.0  # м3/сут
        for m in months:
            seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * m.month / 12.0)
            rate = max(0.0, np.random.normal(base_rate * seasonal, 40.0))
            records.append({
                "well_id": w,
                "t": m.strftime("%d.%m.%Y"),
                "pumping_rate": round(rate, 2),
                "units": "m3/day",
            })

    df_pump = pd.DataFrame(records)

    ls_idx = np.random.choice(df_pump.index, size=int(len(df_pump) * 0.05), replace=False)
    df_pump.loc[ls_idx, "pumping_rate"] = (df_pump.loc[ls_idx, "pumping_rate"] / 86.4).round(3)
    df_pump.loc[ls_idx, "units"] = "l/s"

    err_idx = np.random.choice(df_pump.index.difference(ls_idx), size=20, replace=False)
    df_pump.loc[err_idx, "pumping_rate"] = -df_pump.loc[err_idx, "pumping_rate"]
    df_pump.loc[np.random.choice(df_pump.index, size=25, replace=False), "pumping_rate"] = np.nan

    df_pump = df_pump.sample(frac=1, random_state=42).reset_index(drop=True)
    df_pump.to_csv(FILE_PATH + "pumping.csv", index=False)
    print("pumping.csv successfully generated!")

    abs_records = []
    for m in months:
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (m.month - 3) / 12.0)
        total_vol = np.random.normal(36000.0 * seasonal, 1500.0)
        abs_records.append({
            "intake_name": "Заречный водозабор",
            "x_center": 4000.0,
            "y_center": 3500.0,
            "t": m.strftime("%d.%m.%Y"),
            "total_abstraction_m3_day": round(total_vol, 1),
        })
    df_abs = pd.DataFrame(abs_records)
    df_abs.loc[np.random.choice(df_abs.index, size=3, replace=False), "total_abstraction_m3_day"] = np.nan
    df_abs.to_csv(FILE_PATH + "abstraction.csv", index=False)
    print("abstraction.csv successfully generated!")


def generate_recharge():
    file_name = "recharge.csv"
    months = pd.date_range(start="2020-01-01", end="2025-12-01", freq="MS")
    zones = ["Industrial_Zone", "Water_Intake_Zone", "Forest_Zone"]

    records = []
    for m in months:
        month_factor = {1: 2, 2: 3, 3: 15, 4: 55, 5: 40, 6: 20, 7: 15, 8: 18, 9: 25, 10: 30, 11: 15, 12: 5}[m.month]
        for z in zones:
            zone_mult = 1.3 if z == "Forest_Zone" else (0.6 if z == "Industrial_Zone" else 1.0)
            val = max(0.0, np.random.normal(month_factor * zone_mult, 3.0))
            records.append({
                "zone": z,
                "t": m.strftime("%d.%m.%Y"),
                "recharge_mm": round(val, 2),
            })

    df = pd.DataFrame(records)
    df.loc[np.random.choice(df.index, size=8, replace=False), "recharge_mm"] = -999.0
    df.loc[np.random.choice(df.index, size=6, replace=False), "recharge_mm"] = np.nan
    df.to_csv(FILE_PATH + file_name, index=False)
    print(f"{file_name} successfully generated!")


def generate_queries_test():
    n_queries = 600

    alpha = np.random.uniform(0.0, 1.15, size=n_queries)
    qx = np.clip(1000.0 + alpha * 3000.0 + np.random.normal(0, 450, size=n_queries), 0, 5000).round(2)
    qy = np.clip(1000.0 + alpha * 2500.0 + np.random.normal(0, 450, size=n_queries), 0, 5000).round(2)
    qz = np.random.uniform(-120.0, -10.0, size=n_queries).round(2)

    test_dates = pd.date_range(start="2022-01-01", end="2026-06-30")
    chosen_dates = np.random.choice(test_dates, size=n_queries, replace=True)
    days_since_2020 = (pd.to_datetime(chosen_dates) - pd.Timestamp("2020-01-01")).days.values

    df_test = pd.DataFrame({
        "query_id": [f"Q_{i:04d}" for i in range(1, n_queries + 1)],
        "x": qx,
        "y": qy,
        "z": qz,
        "t": pd.to_datetime(chosen_dates).strftime("%d.%m.%Y"),
    })
    df_test.to_csv(FILE_PATH + "queries_test.csv", index=False)
    print("queries_test.csv successfully generated!")

    true_q_conc = _true_concentration_field(qx, qy, qz, days_since_2020)
    df_gt = df_test.copy()
    df_gt["true_concentration"] = (true_q_conc + 0.1).round(2)
    df_gt.to_csv(FILE_PATH + "_ground_truth_answers.csv", index=False)
    print("_ground_truth_answers.csv (hidden answers for self-check) generated!")


def main():
    clean_coords = generate_wells_info()
    generate_sources()
    generate_permeability()
    generate_concentration_info(clean_coords)
    generate_head(clean_coords)
    generate_cores()
    generate_pumping_and_abstraction()
    generate_recharge()
    generate_queries_test()
    print("\nAll olympiad dataset files are ready in:", FILE_PATH)


if __name__ == "__main__":
    main()