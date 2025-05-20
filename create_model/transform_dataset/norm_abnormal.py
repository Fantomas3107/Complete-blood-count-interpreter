import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
def load_dataset(filepath):
    """
    Загружает датасет из CSV файла и оставляет только нужные столбцы
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Датасет успешно загружен. Размер: {df.shape}")
        
        # Оставляем только нужные столбцы
        selected_columns = ['Age', 'Gender', 'Hemoglobin', 'Hematocrit', 'PLT_Count', 'WBC_Count', 
                           'RBC_Count', 'MCV', 'MCH', 'MCHC']
        
        # Проверяем наличие столбцов в датасете
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            print(f"Внимание! В датасете отсутствуют следующие столбцы: {missing_columns}")
            # Можно скорректировать список столбцов в зависимости от фактических названий
        
        df_selected = df[selected_columns]
        print(f"Выбраны столбцы: {selected_columns}")
        
        return df_selected
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return None

# Функция для определения норм в зависимости от возраста и пола
def define_reference_ranges():
    """
    Определяет референсные значения для каждого показателя в зависимости от возраста и пола
    согласно нормам с сайта Гемотест (https://gemotest.ru/info/spravochnik/analizy/obshchiy-analiz-krovi/)
    """
    # Создаем словарь с обновленными нормами
    reference_ranges = {
        'Hemoglobin': {  # Гемоглобин (г/л)
            'male': {
                'young_adult': (13.2, 17.3),    # 18-44 лет
                'middle_adult': (13.1, 17.2),   # 45-64 лет
                'elderly': (12.6, 17.4)         # от 65 лет
            },
            'female': {
                'young_adult': (11.7, 15.5),    # 18-44 лет
                'middle_adult': (11.7, 16.0),   # 45-64 лет
                'elderly': (11.7, 16.1)         # от 65 лет
            }
        },
        'Hematocrit': {  # Гематокрит (%)
            'male': {
                'young_adult': (39, 49),      # 18-44 лет
                'middle_adult': (39, 50),     # 45-64 лет
                'elderly': (37, 51)           # от 65 лет
            },
            'female': {
                'young_adult': (35, 45),      # 18-44 лет
                'middle_adult': (35, 47),     # 45-64 лет
                'elderly': (37, 51)           # от 65 лет
            }
        },
        'RBC_Count': {  # Эритроциты (10^12/л)
            'male': {
                'young_adult': (4.3, 5.7),    # 18-44 лет
                'middle_adult': (4.2, 5.6),   # 45-64 лет
                'elderly': (3.8, 5.8)         # от 65 лет
            },
            'female': {
                'young_adult': (3.8, 5.1),    # 18-44 лет
                'middle_adult': (3.8, 5.3),   # 45-64 лет
                'elderly': (3.8, 5.2)         # от 65 лет
            }
        },
        'WBC_Count': {  # Лейкоциты (10^9/л)
            'all': {
                'young_adult': (4500, 11300),    # 18-44 лет
                'middle_adult': (4500, 11300),   # 45-64 лет
                'elderly': (4500, 11300)         # от 65 лет
            }
        },
        'PLT_Count': {  # Тромбоциты (10^9/л)
            'all': {
                'young_adult': (180000, 320000),    # 18-44 лет
                'middle_adult': (180000, 320000),   # 45-64 лет
                'elderly': (180000, 320000)         # от 65 лет
            }
        },
        'MCV': {  # Средний объем эритроцита (фл)
            'all': {
                'young_adult': (80, 100),     # 18-44 лет
                'middle_adult': (80, 100),    # 45-64 лет
                'elderly': (80, 100)          # от 65 лет
            }
        },
        'MCH': {  # Эритроциты (10^12/л)
            'male': {
                'young_adult': (27, 34),    # 18-44 лет
                'middle_adult': (27, 35),   # 45-64 лет
                'elderly': (27, 34)         # от 65 лет
            },
            'female': {
                'young_adult': (27, 34),    # 18-44 лет
                'middle_adult': (27, 34),   # 45-64 лет
                'elderly': (27, 35)         # от 65 лет
            }
        },
        'MCHC': {  # Средняя концентрация гемоглобина в эритроците (г/дл)
            'all': {
                'young_adult': (32, 36),      # 18-44 лет
                'middle_adult': (32, 36),     # 45-64 лет
                'elderly': (32, 36)           # от 65 лет
            }
        }
    }
    
    return reference_ranges

# Функция для определения возрастной категории
def get_age_category(age):
    """
    Определяет возрастную категорию на основе возраста
    """
    if age < 18:
        return 'young_adult'  # Возвращаем young_adult для всех людей младше 18 лет
    elif age < 45:
        return 'young_adult'  # 18-44 лет
    elif age < 65:
        return 'middle_adult' # 45-64 лет
    else:
        return 'elderly'      # от 65 лет

# Функция для определения, находится ли показатель в норме
def is_normal(value, parameter, gender, age, reference_ranges):
    """
    Проверяет, находится ли показатель в пределах нормы
    """
    age_category = get_age_category(age)
    
    # Определяем, есть ли гендерные различия для данного параметра
    if gender.lower() in reference_ranges[parameter]:
        gender_key = gender.lower()
    else:
        gender_key = 'all'
    
    # Получаем диапазон нормы
    min_value, max_value = reference_ranges[parameter][gender_key][age_category]
    
    # Проверяем, находится ли значение в пределах нормы
    return min_value <= value <= max_value

# Функция для создания бинарных меток (норма/не норма)
def create_binary_labels(df, reference_ranges):
    """
    Создает бинарные метки для каждой записи: 0 - все показатели в норме, 1 - есть отклонения
    """
    # Создаем столбцы для отметки нормальности каждого показателя
    parameters = ['Hemoglobin', 'Hematocrit', 'RBC_Count', 'WBC_Count', 'PLT_Count', 'MCV', 'MCH', 'MCHC']
    
    for param in parameters:
        column_name = f"{param}_normal"
        df[column_name] = df.apply(
            lambda row: is_normal(row[param], param, row['Gender'], row['Age'], reference_ranges),
            axis=1
        )
    
    # Создаем целевую переменную: 0 - все в норме, 1 - есть отклонения
    df['all_normal'] = df[[f"{param}_normal" for param in parameters]].all(axis=1)
    df['target'] = (~df['all_normal']).astype(int)
    
    # Подсчитываем количество отклонений от нормы
    df['abnormal_count'] = len(parameters) - df[[f"{param}_normal" for param in parameters]].sum(axis=1)
    
    return df

# Функция для анализа датасета
def analyze_dataset(df):
    """
    Выполняет базовый анализ датасета
    """
    print("\nОсновная информация о датасете:")
    print(df.info())
    
    print("\nСтатистика по числовым параметрам:")
    print(df.describe())
    
    print("\nРаспределение по полу:")
    print(df['Gender'].value_counts())
    
    print("\nРаспределение возрастных категорий:")
    age_categories = df['Age'].apply(get_age_category).value_counts()
    print(age_categories)
    
    if 'target' in df.columns:
        print("\nРаспределение целевой переменной:")
        print(df['target'].value_counts())
        print(f"Процент записей с отклонениями: {df['target'].mean() * 100:.2f}%")
    
    return df

# Функция для сохранения подготовленного датасета
def save_processed_dataset(df, output_path):
    """
    Сохраняет обработанный датасет в CSV файл
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"\nОбработанный датасет сохранен в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении датасета: {e}")

# Визуализация распределения показателей
def visualize_distributions(df):
    """
    Создает графики распределения всех показателей и сохраняет их
    """
    parameters = ['Hemoglobin', 'Hematocrit', 'RBC_Count', 'WBC_Count', 'PLT_Count', 'MCV', 'MCH', 'MCHC']
    
    plt.figure(figsize=(18, 15))
    for i, param in enumerate(parameters):
        plt.subplot(3, 3, i+1)
        sns.histplot(data=df, x=param, hue='Gender', kde=True, bins=30)
        plt.title(f'Распределение {param} по полу')
    
    plt.tight_layout()
    plt.savefig('parameter_distributions.png')
    print("\nГрафики распределения параметров сохранены в parameter_distributions.png")
    
    # Визуализация возрастного распределения
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', bins=30, kde=True)
    plt.title('Распределение по возрасту')
    plt.savefig('age_distribution.png')
    print("График распределения по возрасту сохранен в age_distribution.png")
    
    # Визуализация соотношения нормы/не нормы
    if 'target' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='target')
        plt.title('Распределение целевой переменной')
        plt.xticks([0, 1], ['Норма', 'Отклонение'])
        plt.savefig('target_distribution.png')
        print("График распределения целевой переменной сохранен в target_distribution.png")

# Основная функция для обработки датасета
def process_dataset(input_path, output_path):
    """
    Полная обработка датасета от загрузки до сохранения
    """
    # Загружаем датасет
    df = load_dataset(input_path)
    if df is None:
        return
    
    # Обработка возможных пропущенных значений
    print(f"\nПроверка пропущенных значений:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    if missing_values.sum() > 0:
        print("\nЗаполнение пропущенных значений медианными значениями...")
        df = df.fillna(df.median())
    
    # Приведение столбца Gender к стандартному виду (Male/Female)
    if 'Gender' in df.columns:
        # Проверяем уникальные значения
        unique_genders = df['Gender'].unique()
        print(f"\nУникальные значения в столбце Gender: {unique_genders}")
        
        # Стандартизируем формат
        df['Gender'] = df['Gender'].str.strip().str.capitalize()
        df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female', '0': 'Male', '1': 'Female'})
    
    # Определяем референсные диапазоны
    reference_ranges = define_reference_ranges()
    
    # Создаем бинарные метки
    df = create_binary_labels(df, reference_ranges)
    
    # Анализируем датасет
    df = analyze_dataset(df)
    
    # Визуализируем распределения
    visualize_distributions(df)
    
    # Сохраняем обработанный датасет
    save_processed_dataset(df, output_path)
    
    return df

# Пример использования
if __name__ == "__main__":
    input_file = "C:/Users/melix/OneDrive/Desktop/Диплом/new/cbc.csv"
    output_file = "C:/Users/melix/OneDrive/Desktop/Диплом/new/new.csv"
    
    processed_df = process_dataset(input_file, output_file)
    
    if processed_df is not None:
        print("\nПервые 5 строк обработанного датасета:")
        print(processed_df.head())