import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_data(file_path):

    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)

    print(f"Размер датасета: {df.shape}")
    print(f"Распределение целевой переменной: \n{df['target'].value_counts(normalize=True)}")

    features = ['Age', 'Hemoglobin', 'Hematocrit', 'PLT_Count', 'WBC_Count', 'RBC_Count', 'MCV', 'MCH', 'MCHC']

    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
    features.append('Gender_Male')

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Размер тренировочной выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def build_and_train_model(X_train, y_train, X_test, y_test, input_dim):

    print("\nСоздание модели нейронной сети...")

    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ]

    print("\nНачало обучения модели...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    print("\nОценка модели на тестовой выборке...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Точность положительного класса (Precision): {precision:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"F1-мера: {f1:.4f}")

    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=['Норма', 'Требуется консультация']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Норма', 'Отклонение'],
                yticklabels=['Норма', 'Отклонение'])
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.title('Матрица ошибок')
    plt.savefig('confusion_matrix.png')
    print("Матрица ошибок сохранена в файл confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("ROC-кривая сохранена в файл roc_curve.png")

    return accuracy, precision, recall, f1, roc_auc

def plot_training_history(history):

    print("\nПостроение графиков обучения...")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Валидационная выборка')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Валидационная выборка')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Графики обучения сохранены в файл training_history.png")

def create_prediction_model(model, scaler, features):

    def predict_need_doctor(age, gender, hemoglobin, hematocrit, plt_count, wbc_count, rbc_count, mcv, mch, mchc):

        gender_male = 1 if gender.lower() in ['м', 'муж', 'мужской', 'male', 'm'] else 0

        input_data = np.array([[age, hemoglobin, hematocrit, plt_count, wbc_count, rbc_count, mcv, mch, mchc, gender_male]])

        input_scaled = scaler.transform(input_data)

        probability = model.predict(input_scaled)[0][0]
        prediction = 1 if probability > 0.5 else 0

        return prediction, probability

    return predict_need_doctor

def save_model_and_components(model, scaler, features):

    print("\nСохранение модели и компонентов...")

    if not os.path.exists('model'):
        os.makedirs('model')

    model.save('model/blood_analysis_model.h5')

    import joblib
    joblib.dump(scaler, 'model/scaler.pkl')

    with open('model/features.txt', 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")

    print("Модель и компоненты сохранены в директорию 'model'")

def main():

    data_path = "/content/CBC_dataset.csv"
    X_train, X_test, y_train, y_test, scaler, features = load_and_prepare_data(data_path)
    model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_train.shape[1])
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
    plot_training_history(history)
    predict_function = create_prediction_model(model, scaler, features)
    save_model_and_components(model, scaler, features)

if __name__ == "__main__":
    main()