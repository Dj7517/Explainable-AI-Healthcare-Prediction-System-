import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

age = np.random.randint(25, 80, n)
gender = np.random.choice([0, 1], n)  # 0=Female, 1=Male
blood_pressure = np.random.randint(80, 200, n)
cholesterol = np.random.choice([0, 1, 2], n)  # 0=Normal, 1=Above Normal, 2=Well Above Normal
bmi = np.round(np.random.uniform(18, 45, n), 1)
glucose = np.random.choice([0, 1, 2], n)  # 0=Normal, 1=Above Normal, 2=Well Above Normal
smoking = np.random.choice([0, 1], n)
alcohol = np.random.choice([0, 1], n)
physical_activity = np.random.choice([0, 1], n)

# Risk score
risk = (
    (age > 50).astype(int) * 0.2 +
    gender * 0.1 +
    (blood_pressure > 140).astype(int) * 0.25 +
    (cholesterol >= 1).astype(int) * 0.2 +
    (bmi > 30).astype(int) * 0.1 +
    (glucose >= 1).astype(int) * 0.15 +
    smoking * 0.15 +
    alcohol * 0.05 +
    (1 - physical_activity) * 0.1 +
    np.random.normal(0, 0.1, n)
)

target = (risk > 0.5).astype(int)

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'blood_pressure': blood_pressure,
    'cholesterol': cholesterol,
    'bmi': bmi,
    'glucose_level': glucose,
    'smoking': smoking,
    'alcohol_intake': alcohol,
    'physical_activity': physical_activity,
    'heart_disease': target
})

df.to_csv('/home/claude/xai_healthcare/data/heart_disease_dataset.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df['heart_disease'].mean():.1%} positive cases")
print(df.head())
