def get_normal_ranges(age, gender):
    """
    Возвращает нормальные диапазоны показателей крови для взрослых (от 18 лет и старше)
    в зависимости от возраста и пола. Данные на основе информации с сайта https://gemotest.ru/
    
    Args:
        age (int): Возраст пациента в годах (только от 18 лет и старше)
        gender (str): Пол пациента ('М' или 'Ж')
    
    Returns:
        dict: Словарь с нормальными диапазонами для каждого показателя
    
    Raises:
        ValueError: Если возраст меньше 18 лет
    """
    if age < 18:
        raise ValueError("Эта функция предназначена только для взрослых пациентов (18 лет и старше)")
    
    gender = gender.upper()
    
    # Определение возрастной группы для взрослых
    if age < 45:
        age_group = "young_adult"    # 18-44 лет
    elif age < 65:
        age_group = "middle_adult"   # 45-64 лет
    else:
        age_group = "elderly"        # от 65 лет
    
    # Нормы гемоглобина (г/л)
    hemoglobin_ranges = {
        "young_adult": {
            "М": (132, 173),
            "Ж": (117, 155)
        },
        "middle_adult": {
            "М": (131, 172),
            "Ж": (117, 160)
        },
        "elderly": {
            "М": (126, 174),
            "Ж": (117, 161)
        }
    }
    
    # Нормы гематокрита (%)
    hematocrit_ranges = {
        "young_adult": {
            "М": (39, 49),
            "Ж": (35, 45)
        },
        "middle_adult": {
            "М": (39, 50),
            "Ж": (35, 47)
        },
        "elderly": {
            "М": (37, 51),
            "Ж": (37, 51)
        }
    }
    
    # Нормы эритроцитов (×10¹²/л)
    rbc_ranges = {
        "young_adult": {
            "М": (4.3, 5.7),
            "Ж": (3.8, 5.1)
        },
        "middle_adult": {
            "М": (4.2, 5.6),
            "Ж": (3.8, 5.3)
        },
        "elderly": {
            "М": (3.8, 5.8),
            "Ж": (3.8, 5.2)
        }
    }
    
    # Нормы лейкоцитов (×10⁹/л)
    wbc_ranges = {
        "young_adult": (4.5, 11.3),
        "middle_adult": (4.5, 11.3),
        "elderly": (4.5, 11.3)
    }
    
    # Нормы тромбоцитов (×10⁹/л)
    plt_ranges = {
        "young_adult": (180, 320),
        "middle_adult": (180, 320),
        "elderly": (180, 320)
    }
    
    # Нормы MCV (фл)
    mcv_ranges = {
        "young_adult": (80, 100),
        "middle_adult": (80, 100),
        "elderly": (80, 100)
    }
    
    # Нормы MCH (пг)
    mch_ranges = {
        "young_adult": {
            "М": (27, 34),
            "Ж": (27, 34)
        },
        "middle_adult": {
            "М": (27, 35),
            "Ж": (27, 34)
        },
        "elderly": {
            "М": (27, 34),
            "Ж": (27, 35)
        }
    }
    
    # Нормы MCHC (г/л)
    mchc_ranges = {
        "young_adult": (320, 360),
        "middle_adult": (320, 360),
        "elderly": (320, 360)
    }
    
    # Функция для получения значения в зависимости от пола и возрастной группы
    def get_value(ranges_dict):
        if isinstance(ranges_dict[age_group], dict):
            # Для показателей, зависящих от пола
            return ranges_dict[age_group][gender]
        else:
            # Для показателей, не зависящих от пола
            return ranges_dict[age_group]
    
    # Сформировать итоговый словарь с нормальными диапазонами
    return {
        "hemoglobin": get_value(hemoglobin_ranges),
        "hematocrit": get_value(hematocrit_ranges),
        "rbc_count": get_value(rbc_ranges),
        "wbc_count": get_value(wbc_ranges),
        "plt_count": get_value(plt_ranges),
        "mcv": get_value(mcv_ranges),
        "mch": get_value(mch_ranges),
        "mchc": get_value(mchc_ranges)
    }