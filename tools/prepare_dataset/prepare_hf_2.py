import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/samedad/mem-and-russian-jokes-dataset/data/train-00000-of-00001.parquet")

df["Text"] = df["conversations"].apply(lambda x: x[1]["value"])
df = df.dropna()
df = df.drop(columns=["conversations", "source", "score", "__index_level_0__"])
print(len(df))
df.to_csv("data/cleaned_hf_2.csv")