import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Настройки путей
file_path = r"/mnt/e/datasets/funny_text_generator/parquet/concatenated_anekdot_dataset_v2_1_2M.parquet"  # Укажите путь к вашему файлу
column_name = "Text"             # Имя целевой колонки

# # 2. Загрузка данных
# # Читаем только нужную колонку, чтобы сэкономить оперативную память
# df = pd.read_parquet(file_path, columns=[column_name])

# # Удаляем пустые значения, если они есть
# df = df.dropna(subset=[column_name])

# # 3. Расчет длин текстов
# # Считаем длину в символах
# df["char_length"] = df[column_name].astype(str).str.len()

# # Считаем длину в словах (разделение по пробелам)
# df["word_length"] = df[column_name].astype(str).str.split().str.len()

# # 4. Визуализация распределения
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# sns.set_theme(style="whitegrid")

# # График распределения по символам
# sns.histplot(df["char_length"], bins=50, kde=True, ax=axes[0], color="skyblue")
# axes[0].set_title("Распределение длин текстов (в символах)")
# axes[0].set_xlabel("Количество символов")
# axes[0].set_ylabel("Количество текстов")

# # График распределения по словам
# sns.histplot(df["word_length"], bins=50, kde=True, ax=axes[1], color="salmon")
# axes[1].set_title("Распределение длин текстов (в словах)")
# axes[1].set_xlabel("Количество слов")
# axes[1].set_ylabel("Количество текстов")

# # Оптимизация расположения и вывод на экран
# plt.tight_layout()
# plt.show()

# # (Опционально) Вывод базовой статистики в консоль
# print("Статистика по символам:")
# print(df["char_length"].describe())
# print("\nСтатистика по словам:")
# print(df["word_length"].describe())
# print(np.quantile(df["char_length"], q=0.99999))

# # 2. Загрузка ВСЕХ данных (чтобы сохранить структуру файла)
# df = pd.read_parquet(file_path)

# # 3. Поиск индекса самой длинной строки
# # idxmax() находит индекс первого вхождения с максимальным значением
# max_len_idx = df[column_name].astype(str).str.len().idxmax()

# # Опционально: выведем информацию об удаляемом тексте
# removed_text = df.loc[max_len_idx, column_name]
# print(f"Удаляем текст (индекс {max_len_idx}, длина {len(str(removed_text))} симв.):")
# print(f"'{str(removed_text)[:100]}...'") 

# # 4. Удаление строки и перезапись файла
# df_cleaned = df.drop(index=max_len_idx)
# df_cleaned.to_parquet(file_path, index=False)

# print(f"\nФайл {file_path} успешно перезаписан. Строк было: {len(df)}, стало: {len(df_cleaned)}.")

q = 0.99999                         # Квантиль (0.95 означает, что мы оставляем 95% коротких текстов, а 5% самых длинных удаляем)

# 2. Загрузка всех данных
df = pd.read_parquet(file_path)
initial_count = len(df)

# 3. Вычисление длин текстов
lengths = df[column_name].astype(str).str.len()

# 4. Расчет порогового значения (квантиля)
# Например, если q=0.95, то N — это длина, выше которой находится только 5% текстов
N_threshold = lengths.quantile(q)
print(f"Порог для {q*100}% квантиля: {int(N_threshold)} символов.")

# 5. Фильтрация строк
# Оставляем только те строки, которые укладываются в заданный квантиль
mask = lengths <= N_threshold
df_cleaned = df[mask]

removed_count = initial_count - len(df_cleaned)

# 6. Перезапись файла
if removed_count > 0:
    df_cleaned.to_parquet(file_path, index=False)
    print(f"Успешно удалено строк (самых длинных): {removed_count}")
    print(f"Файл {file_path} перезаписан. Было строк: {initial_count}, стало: {len(df_cleaned)}.")
else:
    print("Ни один текст не превысил порог. Файл не изменялся.")