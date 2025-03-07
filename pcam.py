import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from skimage.measure import shannon_entropy
from scipy import stats
from scipy import ndimage
from skimage.transform import resize

from tqdm import tqdm  # Para mostrar progreso
import h5py

# Definición de funciones
def load_dataset(folder_path):  # Carga el dataset PCam desde los archivos h5
    # Rutas para los archivos h5
    train_x_path = os.path.join(folder_path, 'camelyonpatch_level_2_split_train_x.h5')
    train_y_path = os.path.join(folder_path, 'camelyonpatch_level_2_split_train_y.h5')

    # Cargar datos
    print("Cargando datos de entrenamiento...")
    with h5py.File(train_x_path, 'r') as h5f:
        train_images = h5f['x'][:]  # Esto carga todas las imágenes

    with h5py.File(train_y_path, 'r') as h5f:
        train_labels = h5f['y'][:]  # Esto carga todas las etiquetas

    # Para limitar el tamaño para pruebas, si lo comentas no tiene límite
    train_images, train_labels = train_images[:1000], train_labels[:1000]

    print(f"Datos cargados: {train_images.shape} imágenes, {train_labels.shape} etiquetas")
    return train_images, train_labels


def convert_to_grayscale(images):  # Se convierte el dataset a escala de grises
    gray_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]))
    for i in tqdm(range(len(images))):
        gray_images[i] = rgb2gray(images[i])
    return gray_images

def resize_images(images, target_size=(37, 37)):    # Toma los cachos de 37x37
    resized_images = np.zeros((images.shape[0], target_size[0], target_size[1]))
    for i in tqdm(range(len(images))):
        resized_images[i] = resize(images[i], target_size, anti_aliasing=True)
    return resized_images

def vectorize_images(images):   # Transformación de 2D a 1D
    return images.reshape(images.shape[0], -1)


def calculate_features(gray_images):    # Se calculan los estadísticos
    n_samples = gray_images.shape[0]
    # Inicializar matrices para almacenar características
    features = {
        'promedio': np.zeros(n_samples),
        'varianza': np.zeros(n_samples),
        'curtosis': np.zeros(n_samples),
        'simetria': np.zeros(n_samples),
        'fractal': np.zeros(n_samples),
        'entropia': np.zeros(n_samples),
        'gradiente_x': np.zeros(n_samples),
        'gradiente_y': np.zeros(n_samples),
        'energia': np.zeros(n_samples)
    }
    print("Calculando características estadísticas...")
    for i in tqdm(range(n_samples)):
        img = gray_images[i]

        # Características básicas
        features['promedio'][i] = np.mean(img)
        features['varianza'][i] = np.var(img)

        # Curtosis y simetría (usando scipy.stats)
        features['curtosis'][i] = stats.kurtosis(img.flatten())
        features['simetria'][i] = stats.skew(img.flatten())

        # Dimensión fractal (usando método de conteo de cajas)
        # Implementación simplificada
        threshold = np.mean(img)
        binary_img = img > threshold
        # Aproximación de dimensión fractal usando el método de conteo de cajas
        s = np.sum(binary_img)
        if s > 0:
            features['fractal'][i] = np.log(s) / np.log(img.shape[0])
        else:
            features['fractal'][i] = 0

        # Entropía (medida de aleatoriedad)
        features['entropia'][i] = shannon_entropy(img)

        # Gradiente (cambios de intensidad en x e y)
        grad_x = ndimage.sobel(img, axis=0)
        grad_y = ndimage.sobel(img, axis=1)
        features['gradiente_x'][i] = np.mean(np.abs(grad_x))
        features['gradiente_y'][i] = np.mean(np.abs(grad_y))

        # Energía (suma de los cuadrados de los valores de intensidad)
        features['energia'][i] = np.sum(img ** 2)
    return features


def plot_features(features, output_dir):    # Grafica histogramas
    print("Creando gráficas...")
    # Crear un directorio para las gráficas si no existe
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Matriz para mostrar estadísticas de características
    stats_matrix = np.zeros((len(features), 4))

    # Para cada característica
    for i, (feature_name, feature_values) in enumerate(features.items()):
        # Calcular estadísticas
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)

        # Guardar en la matriz
        stats_matrix[i] = [mean_val, std_val, min_val, max_val]

        # Crear histograma
        plt.figure(figsize=(10, 6))
        plt.hist(feature_values, bins=50, alpha=0.7, color='blue')
        plt.title(f'Histograma de {feature_name}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"{feature_name}_histogram.png"))
        plt.close()

    # Mostrar la matriz de estadísticas
    feature_names = list(features.keys())
    stats_names = ['Media', 'Desv. Estándar', 'Mínimo', 'Máximo']

    print("\nEstadísticas de las características:")
    header = f"{'Característica':<15} | " + " | ".join(f"{stat:<15}" for stat in stats_names)
    print(header)
    print("-" * len(header))

    for i, feature in enumerate(feature_names):
        row = f"{feature:<15} | " + " | ".join(f"{stats_matrix[i, j]:<15.5f}" for j in range(4))
        print(row)

    return stats_matrix


def create_feature_matrix(images, features):
    """Crea una matriz de características combinando los vectores de imágenes y las características calculadas"""
    print("Creando matriz de características...")

    # Obtener los valores de las características
    feature_values = np.column_stack([features[feature] for feature in features])

    # Imprimir información sobre la matriz de características
    print(f"Dimensiones de la matriz de características: {feature_values.shape}")

    return feature_values


# Confirma la existencia de un directorio de salida
output_dir = "cancerDetection"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def visualize_and_save_data(X_resized, X_vectorized, feature_matrix, features, y_train, output_dir):
    vis_dir = os.path.join(output_dir, "visualizaciones")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # Visualizar las imágenes de ejemplo
    plt.figure(figsize=(15, 8))
    for i in range(10):  # Aumentamos a 10 ejemplos
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_resized[i], cmap='gray')
        plt.title(f"Etiqueta: {y_train[i][0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ejemplos_imagenes.png"))
    plt.close()

    # Visualizar la matriz de características como una imagen
    plt.figure(figsize=(12, 10))
    plt.imshow(feature_matrix[:100], aspect='auto', cmap='viridis')
    plt.colorbar(label='Valor de la característica')
    plt.title('Visualización de la matriz de características (primeras 100 muestras)')
    plt.xlabel('Índice de característica')
    plt.ylabel('Índice de muestra')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_matrix_visualization.png"))
    plt.close()

    # Visualizar algunos vectores de imágenes
    plt.figure(figsize=(15, 8))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        # Reshape vector back to image for visualization
        vector_as_img = X_vectorized[i].reshape(37, 37)
        plt.imshow(vector_as_img, cmap='gray')
        plt.title(f"Vector {i} como imagen")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "vectores_como_imagenes.png"))
    plt.close()

    # Visualizar correlaciones entre características
    feature_names = list(features.keys())
    feature_values = np.column_stack([features[feature] for feature in features])

    correlation_matrix = np.corrcoef(feature_values, rowvar=False)
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Coeficiente de correlación')
    plt.title('Matriz de correlación entre características')

    # Etiquetado
    tick_marks = np.arange(len(feature_names))
    plt.xticks(tick_marks, feature_names, rotation=45)
    plt.yticks(tick_marks, feature_names)

    # Añadir valores de correlación en el gráfico
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            plt.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.7 else "white")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "correlation_matrix.png"))
    plt.close()

    # Visualizar distribución de características por clase
    plt.figure(figsize=(20, 15))
    for i, feature_name in enumerate(feature_names):
        plt.subplot(3, 3, i + 1)

        # Separar por clase (0 y 1)
        class_0 = feature_values[y_train.flatten() == 0, i]
        class_1 = feature_values[y_train.flatten() == 1, i]

        plt.hist(class_0, bins=30, alpha=0.5, label='Clase 0 (No cáncer)', color='blue')
        plt.hist(class_1, bins=30, alpha=0.5, label='Clase 1 (Cáncer)', color='red')

        plt.title(f'Distribución de {feature_name} por clase')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "distribucion_por_clase.png"))
    plt.close()

    # Visualiza los datos en 3D usando PCA para reducir dimensionalidad


    # Reducir dimensionalidad a 3 componentes
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(feature_values)

    # Crear gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separar por clase
    mask_0 = y_train.flatten() == 0
    mask_1 = y_train.flatten() == 1

    # Graficar puntos
    ax.scatter(reduced_features[mask_0, 0], reduced_features[mask_0, 1], reduced_features[mask_0, 2],
               c='blue', marker='o', alpha=0.3, label='Clase 0 (No cáncer)')
    ax.scatter(reduced_features[mask_1, 0], reduced_features[mask_1, 1], reduced_features[mask_1, 2],
               c='red', marker='^', alpha=0.3, label='Clase 1 (Cáncer)')

    ax.set_title('Visualización 3D de características usando PCA')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "pca_3d_visualization.png"))
    plt.close()

    print(f"Todas las visualizaciones han sido guardadas en: {vis_dir}")


def main():
    data_folder = "./camelyon-patch-py"  # Ajusta esta ruta a donde tengas los archivos h5

    # Cargamos los datos
    X_train, y_train = load_dataset(data_folder)

    print("Forma de X_train: ", X_train.shape)
    print("Forma de y_train: ", y_train.shape)

    # 1. Convertir a escala de grises
    X_gray = convert_to_grayscale(X_train)
    print("Forma después de convertir a escala de grises:", X_gray.shape)

    # 2. Redimensionar a 37x37
    X_resized = resize_images(X_gray, target_size=(37, 37))
    print("Forma después de redimensionar:", X_resized.shape)

    # 3. Calcular características
    features = calculate_features(X_resized)

    # 4. Graficar características y mostrar estadísticas
    stats_matrix = plot_features(features, output_dir)

    # 5. Vectorizar imágenes
    X_vectorized = vectorize_images(X_resized)
    print("Forma después de vectorizar:", X_vectorized.shape)

    # 6. Crear matriz de características
    feature_matrix = create_feature_matrix(X_vectorized, features)

    # 7. Guardar resultados procesados
    print("Guardando resultados procesados...")
    np.save(os.path.join(output_dir, 'X_vectorized.npy'), X_vectorized)
    np.save(os.path.join(output_dir, 'feature_matrix.npy'), feature_matrix)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)

    # 8. Visualizar y guardar datos como imágenes
    visualize_and_save_data(X_resized, X_vectorized, feature_matrix, features, y_train, output_dir)

    print("Procesamiento completado. Los resultados se guardaron en el directorio:", output_dir)

    return X_vectorized, feature_matrix, y_train

if __name__ == "__main__":
    X_vectorized, feature_matrix, y_train = main()