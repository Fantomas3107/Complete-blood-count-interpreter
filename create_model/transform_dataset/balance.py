import pandas as pd
import numpy as np
from sklearn.utils import resample

def balance_dataset(df, output_path):
    """
    Балансирует датасет путем генерации синтетических данных "нормы"
    """
    print("\nНачинаем балансировку датасета...")
    
    # Разделение на нормальные и аномальные записи
    df_normal = df[df['target'] == 0]
    df_abnormal = df[df['target'] == 1]
    
    print(f"Исходный датасет: {len(df)} записей")
    print(f"Нормальные записи: {len(df_normal)} ({len(df_normal)/len(df)*100:.2f}%)")
    print(f"Аномальные записи: {len(df_abnormal)} ({len(df_abnormal)/len(df)*100:.2f}%)")
    
    # Получение референсных диапазонов для генерации данных
    reference_ranges = define_reference_ranges()
    
    # Количество синтетических записей, которые нужно создать
    n_synthetic = len(df_abnormal) - len(df_normal)
    print(f"Требуется создать {n_synthetic} синтетических записей 'нормы'")
    
    # Создаем синтетические данные в пределах нормы
    synthetic_records = []
    
    for _ in range(n_synthetic):
        # Копируем случайную запись из оригинальных нормальных данных для сохранения распределения
        base_record = df_normal.sample(1).iloc[0].copy()
        
        # Генерируем возраст в пределах тех же распределений
        age = np.random.randint(20, 80)
        gender = np.random.choice(['Male', 'Female'])
        
        # Определяем возрастную категорию
        age_category = get_age_category(age)
        
        # Генерируем значения параметров в пределах нормы
        new_record = {
            'Age': age,
            'Gender': gender
        }
        
        # Генерируем значения для каждого параметра в пределах нормы
        parameters = ['Hemoglobin', 'Hematocrit', 'RBC_Count', 'WBC_Count', 'PLT_Count', 'MCV', 'MCH', 'MCHC']
        
        for param in parameters:
            # Определяем, есть ли гендерные различия для данного параметра
            if gender.lower() in reference_ranges[param]:
                gender_key = gender.lower()
            else:
                gender_key = 'all'
            
            # Определяем, есть ли возрастные различия для данного параметра и пола
            if age_category in reference_ranges[param][gender_key]:
                age_key = age_category
            else:
                age_key = 'all'
            
            # Получаем диапазон нормы
            min_value, max_value = reference_ranges[param][gender_key][age_key]
            
            # Генерируем значение с небольшой вариацией вокруг среднего значения диапазона
            mean_value = (min_value + max_value) / 2
            std_dev = (max_value - min_value) / 6  # 99.7% значений будут в пределах диапазона
            
            # Генерируем значение с нормальным распределением в пределах диапазона
            value = np.random.normal(mean_value, std_dev)
            
            # Ограничиваем значение диапазоном нормы
            value = max(min_value, min(max_value, value))
            
            # Округляем значения для сохранения формата
            if param == 'PLT_Count' or param == 'WBC_Count':
                value = int(value)
            elif param in ['Hemoglobin', 'Hematocrit' 'RBC_Count', 'MCV', 'MCH', 'MCHC']:
                value = round(value, 2)
                
            new_record[param] = value
        
        # Добавляем метки нормальности
        for param in parameters:
            new_record[f"{param}_normal"] = True
        
        new_record['all_normal'] = True
        new_record['target'] = 0
        new_record['abnormal_count'] = 0
        
        synthetic_records.append(new_record)
    
    # Создаем DataFrame из синтетических записей
    df_synthetic = pd.DataFrame(synthetic_records)
    
    # Объединяем с оригинальным датасетом
    df_balanced = pd.concat([df, df_synthetic], ignore_index=True)
    
    print(f"Созданы синтетические записи: {len(df_synthetic)}")
    print(f"Сбалансированный датасет: {len(df_balanced)} записей")
    print(f"Нормальные записи в итоге: {len(df_balanced[df_balanced['target'] == 0])} ({len(df_balanced[df_balanced['target'] == 0])/len(df_balanced)*100:.2f}%)")
    print(f"Аномальные записи в итоге: {len(df_balanced[df_balanced['target'] == 1])} ({len(df_balanced[df_balanced['target'] == 1])/len(df_balanced)*100:.2f}%)")
    
    # Перемешиваем датасет
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Сохраняем сбалансированный датасет
    df_balanced.to_csv(output_path, index=False)
    print(f"\nСбалансированный датасет сохранен в {output_path}")
    
    # Визуализация распределения целевой переменной
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_balanced, x='target')
    plt.title('Распределение целевой переменной в сбалансированном датасете')
    plt.xticks([0, 1], ['Норма', 'Отклонение'])
    plt.savefig('balanced_target_distribution.png')
    print("График распределения целевой переменной сохранен в balanced_target_distribution.png")
    
    return df_balanced

# Функция для определения норм в зависимости от возраста и пола
def define_reference_ranges():
    """
    Определяет референсные значения для каждого показателя в зависимости от возраста и пола
    согласно нормам с сайта Гемотест
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
    if age < 18:
        return 'young_adult'  # Возвращаем young_adult для всех людей младше 18 лет
    elif age < 45:
        return 'young_adult'  # 18-44 лет
    elif age < 65:
        return 'middle_adult' # 45-64 лет
    else:
        return 'elderly'      # от 65 лет

# Обновленная основная функция для обработки датасета
def process_and_balance_dataset(input_path, output_path_processed, output_path_balanced):
    """
    Полная обработка датасета от загрузки до сохранения,
    включая балансировку данных
    """
    # Здесь идет ваш оригинальный код из функции process_dataset
    # ...
    
    # Предположим, что processed_df - это результат выполнения process_dataset
    processed_df = pd.read_csv(output_path_processed)
    
    # Балансируем датасет
    balanced_df = balance_dataset(processed_df, output_path_balanced)
    
    return balanced_df

# Пример использования
if __name__ == "__main__":
    output_file_processed = "C:/Users/melix/OneDrive/Desktop/Диплом/new/new.csv"
    output_file_balanced = "C:/Users/melix/OneDrive/Desktop/Диплом/new/balanced.csv"
    
    # Если у вас уже есть обработанный файл, можно сразу его загрузить и сбалансировать
    processed_df = pd.read_csv(output_file_processed)
    balanced_df = balance_dataset(processed_df, output_file_balanced)
    
    print("\nПервые 5 строк сбалансированного датасета:")
    print(balanced_df.head())