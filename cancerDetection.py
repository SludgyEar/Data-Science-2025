import numpy as np
import pickle
import os



# Definición de funciones
def load_dataset_batch(file_path):
    """ Separa el dataset en las imagenes y sus labels """
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return images, labels

def load_dataset(folder_path):
    train_images, train_labels = [], []
    file_path = os.path.join(folder_path, f'camelyonpatch_level_2_split_train_x')
    images, labels = load_dataset_batch(file_path)
    train_images.append(images)
    train_labels.append(labels)
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    # test_images, test_labels = x # En vez de "x" se debe de tomar una parte del dataset para los test_images y test_labels

    # Después se deben de convertir a arrays de numpy en caso de que no lo sean
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Y al final retornamos el conjunto de entrenamiento y el conjunto de test
    return train_images, train_labels #, (test_images, test_labels)

# En caso de que no exista, se crea un directorio para guadar la salida
#output_dir = "cancerDetection"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Running...
data_folder = "./camelyon-patch-py"

# Cargamos los datos
(X_train, y_train) = load_dataset(data_folder)
# Mostramos las formas de los datos
print("Forma de X_train: ", X_train.shape)
print("Forma de y_train: ", y_train.shape)

# Procesar los datos:
