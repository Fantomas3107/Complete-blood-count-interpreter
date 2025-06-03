def get_normal_ranges(age, gender):

    if age < 18:
        raise ValueError("Эта функция предназначена только для взрослых пациентов (18 лет и старше)")
    
    gender = gender.upper()
    
    if age < 45:
        age_group = "young_adult"    
    elif age < 65:
        age_group = "middle_adult" 
    else:
        age_group = "elderly"        
    
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
    
    wbc_ranges = {
        "young_adult": (4.5, 11.3),
        "middle_adult": (4.5, 11.3),
        "elderly": (4.5, 11.3)
    }
    
    plt_ranges = {
        "young_adult": (180, 320),
        "middle_adult": (180, 320),
        "elderly": (180, 320)
    }
    
    mcv_ranges = {
        "young_adult": (80, 100),
        "middle_adult": (80, 100),
        "elderly": (80, 100)
    }

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

    mchc_ranges = {
        "young_adult": (320, 360),
        "middle_adult": (320, 360),
        "elderly": (320, 360)
    }
    
    def get_value(ranges_dict):
        if isinstance(ranges_dict[age_group], dict):
            return ranges_dict[age_group][gender]
        else:
            return ranges_dict[age_group]
    
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