import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def train_evaluate_models(feature_matrix, y_train, output_dir):
    """
    Entrena y evalúa diferentes modelos de clasificación para la detección de cáncer

    Args:
        feature_matrix: Matriz de características extraídas de las imágenes
        y_train: Vector de etiquetas (0: no cáncer, 1: cáncer)
        output_dir: Directorio para guardar los resultados

    Returns:
        best_model: El modelo con mejor desempeño
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        accuracy: Precisión del mejor modelo
    """
    print("Preparando datos para entrenamiento...")

    # Aplanar las etiquetas si es necesario
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train_split, y_test = train_test_split(
        feature_matrix, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

    # Crear directorio para resultados si no existe
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Entrenar diferentes modelos
    print("Entrenando modelos...")

    # Pipeline con escalado de características
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', None)  # Placeholder para el clasificador
    ])

    # 1. Random Forest
    pipeline.set_params(classifier=RandomForestClassifier(random_state=42))

    # Parámetros para búsqueda de hiperparámetros
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_split)

    # Obtener el mejor modelo
    best_rf = grid_search.best_estimator_
    rf_accuracy = best_rf.score(X_test, y_test)
    print(f"Mejor Random Forest - Accuracy: {rf_accuracy:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")

    # 2. SVM
    pipeline.set_params(classifier=SVC(probability=True, random_state=42))

    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_split)

    # Obtener el mejor modelo
    best_svm = grid_search.best_estimator_
    svm_accuracy = best_svm.score(X_test, y_test)
    print(f"Mejor SVM - Accuracy: {svm_accuracy:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")

    # Seleccionar el mejor modelo global
    if rf_accuracy > svm_accuracy:
        best_model = best_rf
        accuracy = rf_accuracy
        print("Random Forest seleccionado como mejor modelo.")
    else:
        best_model = best_svm
        accuracy = svm_accuracy
        print("SVM seleccionado como mejor modelo.")

    # Evaluar el modelo final
    y_pred = best_model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Resultados de Evaluación del Mejor Modelo ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Crear gráficas de evaluación
    visualize_model_results(best_model, X_test, y_test, models_dir)

    return best_model, X_test, y_test, accuracy


def visualize_model_results(model, X_test, y_test, output_dir):
    """
    Crea visualizaciones para evaluar el rendimiento del modelo

    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        output_dir: Directorio para guardar visualizaciones
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()

    classes = ['No Cancer (0)', 'Cancer (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Añadir texto con los valores
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # 3. Visualización de importancia de características (si es RandomForest)
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Importancia de Características')
        plt.bar(range(len(importances)), importances[indices],
                align='center', alpha=0.7)
        plt.xticks(range(len(importances)), indices, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    print(f"Visualizaciones guardadas en: {output_dir}")


def predict_cancer(model, image_features):
    """
    Predice si una imagen contiene células cancerosas

    Args:
        model: Modelo entrenado
        image_features: Características de la imagen a clasificar

    Returns:
        prediction: 0 (no cáncer) o 1 (cáncer)
        probability: Probabilidad de cáncer
    """
    # Asegurarse de que las características están en el formato correcto
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)

    # Obtener la predicción y la probabilidad
    prediction = model.predict(image_features)[0]
    probability = model.predict_proba(image_features)[0][1]  # Probabilidad de la clase 1 (cáncer)

    return prediction, probability


def main_with_model(feature_matrix_path=None, y_train_path=None):
    """
    Función principal que carga datos procesados y entrena un modelo

    Args:
        feature_matrix_path: Ruta a la matriz de características guardada
        y_train_path: Ruta a las etiquetas guardadas

    Returns:
        model: Modelo entrenado
        accuracy: Precisión del modelo
    """
    output_dir = "cancerDetection"

    # Cargar datos procesados si se proporcionan rutas, de lo contrario usar datos del script principal
    if feature_matrix_path and y_train_path and os.path.exists(feature_matrix_path) and os.path.exists(y_train_path):
        print("Cargando datos procesados...")
        feature_matrix = np.load(feature_matrix_path)
        y_train = np.load(y_train_path)
    else:
        # Si no se especifican rutas, intentar cargar desde la ubicación predeterminada
        try:
            print("Intentando cargar datos desde la ubicación predeterminada...")
            feature_matrix = np.load(os.path.join(output_dir, 'feature_matrix.npy'))
            y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
        except FileNotFoundError:
            print("Error: No se encontraron los archivos de datos procesados.")
            print("Ejecuta primero el script de procesamiento o proporciona las rutas correctas.")
            return None, 0

    print(f"Datos cargados: {feature_matrix.shape} características, {y_train.shape} etiquetas")

    # Entrenar y evaluar modelos
    model, X_test, y_test, accuracy = train_evaluate_models(feature_matrix, y_train, output_dir)

    # Guardar el modelo entrenado
    from joblib import dump
    dump(model, os.path.join(output_dir, 'cancer_detection_model.joblib'))
    print(f"Modelo guardado en: {os.path.join(output_dir, 'cancer_detection_model.joblib')}")

    print(f"\n¡Modelo completado! Accuracy final: {accuracy * 100:.2f}%")

    return model, accuracy



# if __name__ == "__main__":
#    model, accuracy = main_with_model()