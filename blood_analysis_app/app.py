import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from reference_values import get_normal_ranges

app = Flask(__name__)

# Загрузка модели и компонентов
def load_model_components():
    model_path = os.path.join('model', 'blood_analysis_model.h5')
    scaler_path = os.path.join('model', 'scaler.pkl')
    
    # Проверка наличия модели и scaler
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    # Загрузка модели
    model = tf.keras.models.load_model(model_path)
    
    # Загрузка scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# Глобальные переменные для модели и scaler
model, scaler = load_model_components()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Получение данных из формы
        data = request.form
        
        age = int(data.get('age', 0))
        gender = data.get('gender', 'М')
        
        # Получение показателей анализа крови
        hemoglobin = float(data.get('hemoglobin', 0))
        hematocrit = float(data.get('hematocrit', 0))
        plt_count = float(data.get('plt_count', 0))
        wbc_count = float(data.get('wbc_count', 0))
        rbc_count = float(data.get('rbc_count', 0))
        mcv = float(data.get('mcv', 0))
        mch = float(data.get('mch', 0))
        mchc = float(data.get('mchc', 0))
        
        # Получение норм для данного возраста и пола
        normal_ranges = get_normal_ranges(age, gender)
        
        # Формирование входных данных для модели
        gender_male = 1 if gender.lower() in ['м', 'муж', 'мужской', 'male', 'm'] else 0
        
        # Создание словаря с показателями анализа и их нормальными значениями
        results = {
            'Гемоглобин (г/л)': {
                'value': hemoglobin,
                'min': normal_ranges['hemoglobin'][0],
                'max': normal_ranges['hemoglobin'][1],
                'is_normal': normal_ranges['hemoglobin'][0] <= hemoglobin <= normal_ranges['hemoglobin'][1]
            },
            'Гематокрит (%)': {
                'value': hematocrit,
                'min': normal_ranges['hematocrit'][0],
                'max': normal_ranges['hematocrit'][1],
                'is_normal': normal_ranges['hematocrit'][0] <= hematocrit <= normal_ranges['hematocrit'][1]
            },
            'Тромбоциты (×10⁹/л)': {
                'value': plt_count,
                'min': normal_ranges['plt_count'][0],
                'max': normal_ranges['plt_count'][1],
                'is_normal': normal_ranges['plt_count'][0] <= plt_count <= normal_ranges['plt_count'][1]
            },
            'Лейкоциты (×10⁹/л)': {
                'value': wbc_count,
                'min': normal_ranges['wbc_count'][0],
                'max': normal_ranges['wbc_count'][1],
                'is_normal': normal_ranges['wbc_count'][0] <= wbc_count <= normal_ranges['wbc_count'][1]
            },
            'Эритроциты (×10¹²/л)': {
                'value': rbc_count,
                'min': normal_ranges['rbc_count'][0],
                'max': normal_ranges['rbc_count'][1],
                'is_normal': normal_ranges['rbc_count'][0] <= rbc_count <= normal_ranges['rbc_count'][1]
            },
            'MCV (фл)': {
                'value': mcv,
                'min': normal_ranges['mcv'][0],
                'max': normal_ranges['mcv'][1],
                'is_normal': normal_ranges['mcv'][0] <= mcv <= normal_ranges['mcv'][1]
            },
            'MCH (пг)': {
                'value': mch,
                'min': normal_ranges['mch'][0],
                'max': normal_ranges['mch'][1],
                'is_normal': normal_ranges['mch'][0] <= mch <= normal_ranges['mch'][1]
            },
            'MCHC (г/л)': {
                'value': mchc,
                'min': normal_ranges['mchc'][0],
                'max': normal_ranges['mchc'][1],
                'is_normal': normal_ranges['mchc'][0] <= mchc <= normal_ranges['mchc'][1]
            }
        }
        
        # Подсчет количества отклонений
        deviations_count = sum(1 for item in results.values() if not item['is_normal'])
        
        # Если модель загружена, выполняем прогноз
        prediction_result = None
        prediction_probability = None
        
        if model is not None and scaler is not None:
            # Формирование входного вектора (порядок должен соответствовать обучению модели)
            input_data = np.array([[
                age, hemoglobin, hematocrit, plt_count, wbc_count, 
                rbc_count, mcv, mch, mchc, gender_male
            ]])
            
            # Стандартизация данных
            input_scaled = scaler.transform(input_data)
            
            # Прогноз
            prediction_probability = float(model.predict(input_scaled)[0][0])
            prediction_result = 1 if prediction_probability > 0.5 else 0
        else:
            # Если модель не загружена, делаем простую эвристику на основе количества отклонений
            prediction_result = 1 if deviations_count > 0 else 0
            prediction_probability = min(deviations_count / len(results), 1.0)
        
        return render_template(
            'result.html',
            age=age,
            gender=gender,
            results=results,
            prediction=prediction_result,
            probability=prediction_probability,
            deviations_count=deviations_count
        )
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)