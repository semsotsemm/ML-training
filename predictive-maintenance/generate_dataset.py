import numpy as np
import pandas as pd

def generate_telemetry(sample_count: int = 1000, seed: int = 1) -> pd.Dataframe:
    np.random.seed(seed)
    temperature = np.random.uniform(low=30.0, high=105.0, size=sample_count)
    vibration = np.random.uniform(low=0.1, high=6.0, size=sample_count)
    uptime_hours = np.random.uniform(low=0, high=2000, size=sample_count)
    risk_score = (temperature / 100.0) * 1.5 + (vibration / 5.0) * 2.0 + (uptime_hours / 2000.0) * 1.0
    noise = np.random.normal(loc=0.0, scale=0.3, size=sample_count)
    final_risk = risk_score + noise

    target = (final_risk > 2.8).astype(int)

    df = pd.DataFrame(
        {
            "Temperature_C": np.round(temperature, 2),
            "Vibration_mms": np.round(vibration, 2),
            "Uptime_hrs": np.round(uptime_hours, 1),
            "Needs_maintenance": target
        }
    )

if __name__ == '__main__':
    dataset = generate_telemetry(1000)

    file_path = 'telemetry_data.csv'
    dataset.to_csv(file_path, index=False)