import pandas as pd

df = pd.read_csv("data/hf_1_raw.csv")
df = df.rename(columns={"text": "Text"})
df = df.dropna()
print(len(df))
df.to_csv("data/cleaned_hf_1.csv")
