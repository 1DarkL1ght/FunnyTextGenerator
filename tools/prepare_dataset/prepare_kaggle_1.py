import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("konstantinalbul/russian-jokes")

df = pd.read_csv("data/kaggle_1.csv")
df = df.drop(columns=["theme", "rating"])
df = df.dropna()
df = df.rename(columns={"text": "Text"})
print(df.head())
print(len(df))
df.to_csv("data/cleaned_kaggle_1.csv")