import pandas as pd

df_base = pd.read_csv("data/concatenated_anekdot_dataset.csv")

df1 = pd.read_csv("data/cleaned_kaggle_1.csv")
df2 = pd.read_csv("data/cleaned_kaggle_2.csv")
df3 = pd.read_csv("data/cleaned_hf_1.csv")
df4 = pd.read_csv("data/cleaned_hf_2.csv")

df = pd.concat([df_base, df1, df2, df3, df4], ignore_index=True)
df["Text"] = df["Text"].apply(lambda x: x.strip().replace("\r\n", "\n") if isinstance(x, str) else x)
df = df.drop_duplicates(subset="Text", ignore_index=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(subset=["Text"])
df = df[df['Text'].map(type) == str]
df = df.drop(index=391774) # idk just NaN val cannot be deleted somehow
print(df.head())
print(len(df))
df.to_csv(f"data/concatenated_anekdot_dataset_v2_1_2M.csv", index=False)