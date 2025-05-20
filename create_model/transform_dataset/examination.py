import csv

def find_min_max_in_first_column(file_path):
    min_val = None
    max_val = None
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:  # Пропускаем пустые строки
                continue
            try:
                current_value = float(row[0])  # Преобразуем первое значение в число
                if min_val is None or current_value < min_val:
                    min_val = current_value
                if max_val is None or current_value > max_val:
                    max_val = current_value
            except ValueError:
                print(f"Пропущено нечисловое значение: {row[0]}")
    
    return min_val, max_val

# Пример использования
file_path = 'balanced.csv'  # Замените на путь к вашему файлу
min_value, max_value = find_min_max_in_first_column(file_path)

print(f"Минимальное значение в первом столбце: {min_value}")
print(f"Максимальное значение в первом столбце: {max_value}")