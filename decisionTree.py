import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import seaborn as sns


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return images, labels


def load_cifar10_dataset(folder_path):
    train_images, train_labels = [], []
    for i in range(1, 6):  # Cargar los 5 batches de entrenamiento
        file_path = os.path.join(folder_path, f'data_batch_{i}')
        images, labels = load_cifar10_batch(file_path)
        train_images.append(images)
        train_labels.append(labels)
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    test_images, test_labels = load_cifar10_batch(os.path.join(folder_path, 'test_batch'))

    # Convertir a arrays de numpy en caso de que no lo sean
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

# Crear el directorio de salida si no existe
output_dir = "decision_tree"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_folder = "./cifar-10-batches-py"

# Cargar los datos (una sola vez)
(X_train, y_train), (X_test, y_test) = load_cifar10_dataset(data_folder)

# Mostrar las formas de los datos
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_test:", y_test.shape)

# Preprocesar los datos:
# 1. Convertir las imágenes a vectores unidimensionales (32x32x3 -> 3072)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 2. Normalizar los valores de píxeles al rango [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 3. Aplanar las etiquetas (por si estuvieran en formato 2D)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Dividir los datos en entrenamiento y validación (útil para ajustar hiperparámetros)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Crear el modelo de Árbol de Decisión
tree = DecisionTreeClassifier(random_state=42)

# Ajustar hiperparámetros usando GridSearchCV
param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)

# Entrenar el modelo con los mejores hiperparámetros
best_tree = grid_search.best_estimator_
best_tree.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de validación
y_val_pred = best_tree.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy en el conjunto de validación: {val_accuracy * 100:.2f}%")

# Evaluar el modelo en el conjunto de prueba
y_test_pred = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en el conjunto de prueba: {test_accuracy * 100:.2f}%")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_test_pred, target_names=[
    "Aviones", "Coches", "Pájaros", "Gatos", "Ciervos",
    "Perros", "Ranas", "Caballos", "Barcos", "Camiones"
]))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[
                "Aviones", "Coches", "Pájaros", "Gatos", "Ciervos",
                "Perros", "Ranas", "Caballos", "Barcos", "Camiones"
            ],
            yticklabels=[
                "Aviones", "Coches", "Pájaros", "Gatos", "Ciervos",
                "Perros", "Ranas", "Caballos", "Barcos", "Camiones"
            ])
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
# plt.show()

base_name = f'Decision Tree.png'
output_path = os.path.join(output_dir, base_name)
plt.savefig(output_path)
plt.close()