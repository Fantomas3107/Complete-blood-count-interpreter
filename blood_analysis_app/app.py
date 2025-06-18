from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime

app = Flask(__name__)

class BloodAnalysisAI:
    def __init__(self, model_path='create_model/model/result/'):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.load_model(model_path)
        
        self.disease_info = {
            'Normal': {
                'description': 'Показатели в пределах нормы',
                'recommendations': [
                    'Поддерживайте здоровый образ жизни',
                    'Регулярно проходите профилактические осмотры',
                    'Сбалансированное питание и физическая активность',
                    'Достаточный сон и управление стрессом'
                ],
                'severity': 'low'
            },
            'Rheumatoid Arthritis': {
                'description': 'Ревматоидный артрит - хроническое аутоиммунное воспалительное заболевание',
                'recommendations': [
                    'Обязательная консультация ревматолога',
                    'Рассмотреть дополнительные анализы: РФ, АЦЦП, СОЭ',
                    'Регулярное наблюдение у специалиста',
                    'Противовоспалительная терапия по назначению врача',
                    'Лечебная физкультура и физиотерапия'
                ],
                'severity': 'high'
            },
            'Systemic Lupus Erythematosus': {
                'description': 'Системная красная волчанка - аутоиммунное заболевание соединительной ткани',
                'recommendations': [
                    'Срочная консультация ревматолога',
                    'Анализы на АНА, анти-dsDNA, комплемент',
                    'Защита от солнечного излучения',
                    'Регулярный мониторинг функции почек',
                    'Иммуносупрессивная терапия по показаниям'
                ],
                'severity': 'high'
            },
            'Sjögren\'s Syndrome': {
                'description': 'Синдром Шегрена - аутоиммунное поражение слюнных и слезных желез',
                'recommendations': [
                    'Консультация ревматолога и офтальмолога',
                    'Анализы на SSA/SSB антитела',
                    'Увлажнение глаз и полости рта',
                    'Регулярная стоматологическая помощь',
                    'Мониторинг функции желез'
                ],
                'severity': 'medium'
            },
            'Psoriatic Arthritis': {
                'description': 'Псориатический артрит - воспалительное заболевание суставов при псориазе',
                'recommendations': [
                    'Консультация ревматолога и дерматолога',
                    'МРТ суставов для оценки воспаления',
                    'Лечение кожных проявлений псориаза',
                    'Биологическая терапия при тяжелых формах',
                    'Регулярная физическая активность'
                ],
                'severity': 'medium'
            },
            'Graves disease': {
                'description': 'Воспалительные заболевания кишечника (болезнь Крона, язвенный колит)',
                'recommendations': [
                    'Консультация гастроэнтеролога',
                    'Колоноскопия с биопсией',
                    'Диетотерапия и исключение триггеров',
                    'Противовоспалительная терапия',
                    'Мониторинг нутритивного статуса'
                ],
                'severity': 'high'
            }
        }
    
    def load_model(self, model_path):
        try:
            model_file = os.path.join(model_path, 'model.h5')
            if os.path.exists(model_file):
                self.model = tf.keras.models.load_model(model_file, compile=False)
                
            scaler_file = os.path.join(model_path, 'scaler.pkl')
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                
            le_file = os.path.join(model_path, 'label_encoder.pkl')
            if os.path.exists(le_file):
                self.label_encoder = joblib.load(le_file)
                
            metadata_file = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    
            print("Модель успешно загружена")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def get_normal_ranges(self, age, gender):
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
        
        neutrophils_ranges = {
            "young_adult": (2.0, 7.0),
            "middle_adult": (2.0, 7.0),
            "elderly": (2.0, 7.0)
        }
        
        lymphocytes_ranges = {
            "young_adult": (1.2, 3.4),
            "middle_adult": (1.2, 3.4),
            "elderly": (1.2, 3.4)
        }
        
        monocytes_ranges = {
            "young_adult": (0.1, 0.9),
            "middle_adult": (0.1, 0.9),
            "elderly": (0.1, 0.9)
        }
        
        eosinophils_ranges = {
            "young_adult": (0.02, 0.5),
            "middle_adult": (0.02, 0.5),
            "elderly": (0.02, 0.5)
        }
        
        esr_ranges = {
            "young_adult": {
                "М": (2, 15),
                "Ж": (2, 20)
            },
            "middle_adult": {
                "М": (2, 20),
                "Ж": (2, 25)
            },
            "elderly": {
                "М": (2, 30),
                "Ж": (2, 35)
            }
        }
        
        crp_ranges = {
            "young_adult": (0, 3),
            "middle_adult": (0, 5),
            "elderly": (0, 10)
        }
        
        def get_value(ranges_dict):
            if isinstance(ranges_dict[age_group], dict):
                return ranges_dict[age_group][gender]
            else:
                return ranges_dict[age_group]
        
        return {
            "WBC_Count": get_value(wbc_ranges),
            "RBC_Count": get_value(rbc_ranges),
            "Hemoglobin": get_value(hemoglobin_ranges),
            "Hematocrit": get_value(hematocrit_ranges),
            "PLT_Count": get_value(plt_ranges),
            "Neutrophils": get_value(neutrophils_ranges),
            "Lymphocytes": get_value(lymphocytes_ranges),
            "Monocytes": get_value(monocytes_ranges),
            "Eosinophils": get_value(eosinophils_ranges),
            "ESR": get_value(esr_ranges),
            "CRP": get_value(crp_ranges),
            "MCV": get_value(mcv_ranges),
            "MCH": get_value(mch_ranges),
            "MCHC": get_value(mchc_ranges)
        }
    
    def analyze_blood_test(self, blood_data, age, gender):
        try:
            features = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'Hematocrit', 'PLT_Count',
                       'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'ESR',
                       'Age', 'CRP']
            
            feature_vector = []
            for feature in features:
                if feature == 'Age':
                    feature_vector.append(age)
                elif feature == 'Sickness_Duration_Months':
                    feature_vector.append(0)  
                elif feature in blood_data:
                    feature_vector.append(float(blood_data[feature]))
                else:
                    default_values = {
                        'CRP': 1.0, 'ESR': 10.0, 'Neutrophils': 4.0,
                        'Lymphocytes': 2.0, 'Monocytes': 0.5, 'Eosinophils': 0.2
                    }
                    feature_vector.append(default_values.get(feature, 0))
            
            if self.scaler and self.model:
                feature_vector = np.array(feature_vector).reshape(1, -1)
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                predictions = self.model.predict(feature_vector_scaled)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                if self.label_encoder:
                    predicted_class = self.label_encoder.classes_[predicted_class_idx]
                else:
                    predicted_class = f"Class_{predicted_class_idx}"
                
                return {
                    'predicted_disease': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        self.label_encoder.classes_[i]: float(predictions[0][i])
                        for i in range(len(predictions[0]))
                    } if self.label_encoder else {}
                }
            else:
                return {'predicted_disease': 'Normal', 'confidence': 0.5, 'probabilities': {}}
                
        except Exception as e:
            print(f"Ошибка анализа: {e}")
            return {'predicted_disease': 'Normal', 'confidence': 0.5, 'probabilities': {}}
    
    def check_deviations(self, blood_data, age, gender):
        deviations = {}
        
        reference_ranges = self.get_normal_ranges(age, gender)
        
        for parameter, value in blood_data.items():
            if parameter in reference_ranges:
                min_val, max_val = reference_ranges[parameter]
                
                is_normal = min_val <= float(value) <= max_val
                deviation_type = None
                
                if not is_normal:
                    if float(value) < min_val:
                        deviation_type = 'below'
                    else:
                        deviation_type = 'above'
                
                deviations[parameter] = {
                    'value': float(value),
                    'min': min_val,
                    'max': max_val,
                    'is_normal': is_normal,
                    'deviation_type': deviation_type
                }
        
        return deviations

ai_analyzer = BloodAnalysisAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        
        blood_data = {
            'WBC_Count': float(request.form['wbc_count']),
            'RBC_Count': float(request.form['rbc_count']),
            'Hemoglobin': float(request.form['hemoglobin']),
            'Hematocrit': float(request.form['hematocrit']),
            'PLT_Count': float(request.form['plt_count']),
        }
        optional_params = ['neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'esr', 'crp']
        for param in optional_params:
            if param in request.form and request.form[param]:
                blood_data[param.title()] = float(request.form[param])
        
        ai_result = ai_analyzer.analyze_blood_test(blood_data, age, gender)
        
        deviations = ai_analyzer.check_deviations(blood_data, age, gender)
        
        deviations_count = sum(1 for dev in deviations.values() if not dev['is_normal'])
        
        predicted_disease = ai_result['predicted_disease']
        disease_info = ai_analyzer.disease_info.get(predicted_disease, {
            'description': 'Неизвестное состояние',
            'recommendations': ['Обратитесь к врачу для дополнительной консультации'],
            'severity': 'medium'
        })
        
        return render_template('result.html',
                             age=age,
                             gender=gender,
                             results=deviations,
                             deviations_count=deviations_count,
                             ai_prediction=ai_result,
                             disease_info=disease_info,
                             analysis_date=datetime.now().strftime('%d.%m.%Y %H:%M'))
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.get_json()
        
        age = data['age']
        gender = data['gender']
        blood_data = data['blood_data']
        
        ai_result = ai_analyzer.analyze_blood_test(blood_data, age, gender)
        
        deviations = ai_analyzer.check_deviations(blood_data, age, gender)
        
        return jsonify({
            'success': True,
            'ai_prediction': ai_result,
            'deviations': deviations,
            'disease_info': ai_analyzer.disease_info.get(ai_result['predicted_disease'], {})
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)