import pandas as pd

# Загрузка данных с учетом:
# - разделитель столбцов: запятая
# - десятичный разделитель: точка (по умолчанию)
df = pd.read_csv('C:/Users/melix/OneDrive/Desktop/Диплом/new/balanced.csv', sep=',', encoding='utf-8')

# Проверка первых строк и типов данных
print("Первые 5 строк данных:")
print(df.head())
print("\nТипы данных столбцов:")
print(df.dtypes)

# Обработка числовых столбцов
numeric_columns = ['Hemoglobin', 'PLT_Count', 'WBC_Count', 'MCV', 'MCH', 'MCHC']

for col in numeric_columns:
    if col in df.columns:
        # Преобразуем в float (если данные уже с точкой)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Применяем специфичные преобразования
        if col == 'Hemoglobin':
            df[col] = (df[col] * 10).round().astype(int)
        elif col == 'PLT_Count':
            df[col] = (df[col] / 1000).round().astype(int)
        elif col == 'WBC_Count':
            df[col] = (df[col] / 1000).round(2)
        elif col in ['MCV', 'MCH']:
            df[col] = df[col].round(1)
        elif col == 'MCHC':
            df[col] = (df[col] * 10).round().astype(int)

# Обработка возраста (если есть некорректные значения)
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)

# Сохранение результата
output_file = 'CBC.csv'
df.to_csv(output_file, index=False, encoding='utf-8', sep=',', float_format='%.2f')

print("\nРезультат обработки:")
print(df.head())
print(f"\nДанные успешно сохранены в файл: {output_file}")