import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                      ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import label_binarize
from itertools import cycle

np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('ggplot')

def load_data(file_path):
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
    print(f"{'='*60}")

    df = pd.read_csv(file_path)

    if 'Diagnosis' not in df.columns:
        raise ValueError("Столбец 'Diagnosis' не найден в данных")

    initial_size = len(df)
    df = df[df['Diagnosis'] != 'Other']
    removed_count = initial_size - len(df)
    if removed_count > 0:
        print(f"Удалено {removed_count} записей с Diagnosis='Other'")

    print(f"Итоговое количество строк: {len(df)}")

    target_col = 'Diagnosis'
    class_dist = df[target_col].value_counts(normalize=True)
    n_classes = len(class_dist)
    is_binary = n_classes == 2

    print("\nРаспределение классов:")
    for class_name, proportion in class_dist.items():
        count = df[target_col].value_counts()[class_name]
        print(f"{class_name}: {count} ({proportion:.3f})")

    print(f"Количество классов: {n_classes}")

    min_class_size = df[target_col].value_counts().min()
    max_class_size = df[target_col].value_counts().max()
    imbalance_ratio = max_class_size / min_class_size
    print(f"Коэффициент дисбаланса: {imbalance_ratio:.2f}")

    if imbalance_ratio > 3:
        print("Обнаружен значительный дисбаланс классов!")

    return df, target_col, is_binary, n_classes

def preprocess_data_improved(df, target_col, validation_size=0.2):
    print(f"\n{'='*60}")
    print("УЛУЧШЕННАЯ ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА")
    print(f"{'='*60}")

    features = ['WBC_Count', 'RBC_Count', 'Hemoglobin', 'Hematocrit', 'PLT_Count',
               'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'ESR',
               'Age', 'Reticulocyte_Count', 'Basophils', 'Gender']

    available_features = [f for f in features if f in df.columns]
    print(f"Используется {len(available_features)} признаков: {available_features}")

    gender_encoder = None
    if 'Gender' in df.columns:
        gender_encoder = LabelEncoder()
        df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    X = df[available_features]
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=validation_size, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    n_classes = len(class_names)
    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\nРазмеры после разделения:")
    print(f"Train: {X_train_scaled.shape}")
    print(f"Validation: {X_val_scaled.shape}")
    print(f"Веса классов: {class_weight_dict}")

    return (X_train_scaled, X_val_scaled,
            y_train_cat, y_val_cat,
            y_train, y_val,
            scaler, available_features, le, class_names, class_weight_dict, gender_encoder)

def build_regularized_model(input_shape, n_classes):
    print(f"\n{'='*60}")
    print("ПОСТРОЕНИЕ РЕГУЛЯРИЗОВАННОЙ МОДЕЛИ")
    print(f"{'='*60}")

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(input_shape,),
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(n_classes, activation='softmax',
              kernel_initializer='glorot_uniform'))

    model.compile(
        optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Общее количество параметров: {model.count_params():,}")
    model.summary()
    return model

def create_callbacks():
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-8,
            verbose=1,
            cooldown=3
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print(f"{'='*60}")

    callbacks = create_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history

def cross_validate_model(df, target_col, features, class_weights, n_classes):
    print(f"\n{'='*60}")
    print("КРОСС-ВАЛИДАЦИЯ")
    print(f"{'='*60}")

    X = df[features].values
    y = df[target_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        y_train_fold_cat = to_categorical(y_train_fold, n_classes)
        y_val_fold_cat = to_categorical(y_val_fold, n_classes)

        model = build_regularized_model(X_train_fold.shape[1], n_classes)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train_fold, y_train_fold_cat,
                 validation_data=(X_val_fold, y_val_fold_cat),
                 epochs=50, batch_size=32,
                 callbacks=[early_stop],
                 class_weight=class_weights,
                 verbose=0)

        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold_cat, verbose=0)
        cv_scores.append(val_acc)

    return cv_scores

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, class_names):
    print(f"\n{'='*60}")
    print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
    print(f"{'='*60}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    try:
        if len(class_names) == 2:
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            print(f"AUC-ROC:   {auc_score:.4f}")
        else:
            auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            print(f"AUC-ROC:   {auc_score:.4f}")
    except Exception as e:
        print(f"Ошибка при вычислении AUC: {e}")
        auc_score = None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score
    }

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок', fontsize=16)
    plt.xlabel('Предсказание', fontsize=14)
    plt.ylabel('Истенные классы', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_score, class_names):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-class Classification', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history_with_log(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.semilogy(history.history['loss'], label='Обучающая', linewidth=2, color='blue')
    plt.semilogy(history.history['val_loss'], label='Валидационная', linewidth=2, color='red')
    plt.title('Динамика функции потерь (логарифм)', fontsize=14)
    plt.xlabel('Эпохи', fontsize=12)
    plt.ylabel('Потери (log scale)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Обучающая', linewidth=2, color='blue')
    plt.plot(history.history['val_accuracy'], label='Валидационная', linewidth=2, color='red')
    plt.title('Динамика точности', fontsize=14)
    plt.xlabel('Эпохи', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
    plt.plot(loss_diff, linewidth=2, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Разность потерь (Train - Val)', fontsize=14)
    plt.xlabel('Эпохи', fontsize=12)
    plt.ylabel('Разность потерь', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_with_log.png', dpi=300, bbox_inches='tight')
    plt.show()

def main_improved():
    try:
        data_path = "/content/Complete_Updated_Autoimmune_Disorder_Dataset2.csv"

        print("ЗАПУСК СИСТЕМЫ КЛАССИФИКАЦИИ С ИТОГОВЫМИ МЕТРИКАМИ")
        print("="*80)

        df, target_col, is_binary, n_classes = load_data(data_path)

        (X_train_scaled, X_val_scaled,
         y_train_cat, y_val_cat,
         y_train, y_val,
         scaler, features, le, class_names, class_weight_dict, gender_encoder) = preprocess_data_improved(
            df, target_col)

        cv_scores = cross_validate_model(df, target_col, features,
                                       class_weight_dict, n_classes)

        model = build_regularized_model(X_train_scaled.shape[1], n_classes)
        history = train_model(
            model, X_train_scaled, y_train_cat, X_val_scaled, y_val_cat, class_weight_dict)

        plot_training_history_with_log(history)

        y_pred_proba = model.predict(X_val_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        metrics = calculate_comprehensive_metrics(y_val, y_pred, y_pred_proba, class_names)

        plot_confusion_matrix(y_val, y_pred, class_names)
        plot_roc_curve(y_val, y_pred_proba, class_names)

        print(f"\n{'='*60}")
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print(f"{'='*60}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"CV Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        if metrics['auc_score'] is not None:
            print(f"AUC-ROC: {metrics['auc_score']:.4f}")

        return model, history, metrics, cv_scores

    except Exception as e:
        print(f"\n Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model, history, metrics, cv_scores = main_improved()