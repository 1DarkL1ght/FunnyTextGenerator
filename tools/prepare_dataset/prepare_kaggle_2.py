import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("dokster/jokes-in-russian-dataset-500k")

with open("data/kaggle_2.txt", encoding="utf-8") as f:
    lines = f.readlines()

df = pd.DataFrame({"Text": lines})
print(df.head())
print(len(df))
df.to_csv("data/cleaned_kaggle_2.csv")