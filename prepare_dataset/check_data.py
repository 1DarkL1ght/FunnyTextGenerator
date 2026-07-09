import pandas as pd

df = pd.read_csv("data/concatenated_anekdot_dataset_v2_1_2M.csv")

print(df.head())
print(df.columns)
print(df[df.isna().any(axis=1)].index)
val = df.loc[391774, 'Text']
print(f"1. Представление (repr): {repr(val)}")
print(f"2. Тип: {type(val)}")
print(f"3. Длина: {len(str(val))}")
print(f"4. Сравнение с None: {val is None}")
print(df.describe())