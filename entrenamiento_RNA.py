# -*- coding: utf-8 -*-
"""carcasas_pytorch_efficientnet-TFM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15wS8yoa2uydlVPJcPxkgQqr8feRcVQhi
"""

import zipfile
import os
from PIL import Image
import pandas as pd
# Crear carpeta del dataset
if 'dataset' not in os.listdir('/content'):
  os.makedirs('/content/dataset/')

# Montar drive
from google.colab import drive
drive.mount('/content/drive')

# Descomprimir el archivo zip que contiene el dataset
data_dir = '/content/dataset/d_va_3.zip'
with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset-aug')

# Instalar modulo efficientnet_pytorch
!pip install -q efficientnet_pytorch

"""
Crear un nuevo dataset a partir del existente aplicando métodos
de data augmentation
"""

from torchvision.transforms import v2 as T

def im_aug_v2(im, grad: int):
    """
    Realiza una serie de aumentos de datos (data augmentation) sobre una imagen proporcionada.
    Los aumentos incluyen rotación, transformación afín, ajustes de color, desenfoque y transformación elástica.

    Parámetros:
        im: Imagen de entrada que será procesada.
        grad (int): Número de gradientes o iteraciones para las transformaciones.

    Retorna:
        Una lista de imágenes transformadas (im_list_tr_2) con diferentes aumentos aplicados.
    """

    # Lista para almacenar las imágenes transformadas en la primera etapa
    im_list_tr = []

    # Se divide el círculo completo (360 grados) entre el número de iteraciones grad
    degrees_per_it = 360.0 / grad

    # Primera etapa de transformaciones: Rotaciones y transformaciones afines
    for it in range(0, grad):
        # Se calcula el ángulo de rotación para la iteración actual
        degrees = degrees_per_it * it

        # Rotación aleatoria en el rango exacto de grados calculado
        transform = T.RandomRotation([degrees, degrees])
        im_tr = transform(im)  # Se aplica la transformación a la imagen
        im_list_tr.append(im_tr)  # Se guarda la imagen transformada

        # Transformación afín con rotación, traslación, escala y cizalladura (shear)
        transform = T.RandomAffine(
            degrees=degrees,        # Rotación de la imagen
            translate=(0.01, 0.12), # Rango de traslación
            shear=(0.01, 0.03),     # Rango de cizalladura
            scale=(0.7, 0.9)        # Rango de escalado
        )
        im_tr = transform(im)  # Se aplica la transformación a la imagen
        im_list_tr.append(im_tr)  # Se guarda la imagen transformada

    # Lista para almacenar las imágenes transformadas en la segunda etapa
    im_list_tr_2 = []

    # Segunda etapa de transformaciones: Ajustes de color, desenfoque y transformación elástica
    for im in im_list_tr:
        # Ajuste de color (brillo, contraste, saturación y tono)
        transform = T.ColorJitter(
            brightness=0.6,         # Variación de brillo
            contrast=0.7,           # Variación de contraste
            saturation=0.5,         # Variación de saturación
            hue=(-0.5, 0.5)         # Variación de tono
        )
        im_tr = transform(im)  # Se aplica la transformación a la imagen
        im_list_tr_2.append(im_tr)  # Se guarda la imagen transformada

        # Aplicación de desenfoque gaussiano
        transform = T.GaussianBlur(
            kernel_size=(9, 17),    # Tamaño del kernel para el desenfoque
            sigma=(3, 16)           # Rango de desviación estándar para el desenfoque
        )
        im_tr = transform(im)  # Se aplica la transformación a la imagen
        im_list_tr_2.append(im_tr)  # Se guarda la imagen transformada

        # Transformación elástica para distorsionar la imagen
        transform = T.ElasticTransform(
            alpha=(100),            # Intensidad de la distorsión
            sigma=(4, 10)           # Rango de suavizado en la transformación
        )
        im_tr = transform(im)  # Se aplica la transformación a la imagen
        im_list_tr_2.append(im_tr)  # Se guarda la imagen transformada

    # Retorna la lista final de imágenes con todas las transformaciones aplicadas
    return im_list_tr_2

def data_augmentation(path_dataset_original, path_dataset_final, grad):
    """
    Función para realizar un aumento de datos (data augmentation) a un conjunto de datos de imágenes.
    Crea nuevas imágenes transformadas y actualiza los conjuntos de entrenamiento y prueba con las
    imágenes generadas, además de guardar las etiquetas correspondientes en archivos CSV.

    Parámetros:
        path_dataset_original: Ruta del conjunto de datos original (directorio raíz).
        path_dataset_final: Ruta donde se almacenará el conjunto de datos aumentado.
        grad: Número de gradientes o transformaciones a aplicar por imagen.
    """

    # Crear una carpeta temporal para el conjunto de datos aumentado si no existe
    if 'dataset-aug' not in os.listdir('/content'):
        os.makedirs('/content/dataset-aug/')

    # Leer los archivos CSV que contienen las etiquetas del conjunto de datos original
    og_csv_train = pd.read_csv(path_dataset_original + '/train.csv')  # CSV de entrenamiento
    og_csv_test = pd.read_csv(path_dataset_original + '/test.csv')    # CSV de prueba

    # Definir las rutas de las imágenes originales
    og_im_list_train = path_dataset_original + '/images/train'
    og_im_list_test = path_dataset_original + '/images/test'

    # Crear nuevos DataFrames para almacenar la información de las imágenes aumentadas
    train_df = pd.DataFrame(columns=['file', 'classCode', 'classDescription'])
    test_df = pd.DataFrame(columns=['file', 'classCode', 'classDescription'])
    num = {'train': 0, 'test': 0}  # Contadores para nombrar las imágenes nuevas

    # Crear subcarpetas en el directorio final donde se guardarán las imágenes
    if 'images' not in os.listdir(path_dataset_final):
        os.makedirs(path_dataset_final + '/images')
    if 'train' not in os.listdir(path_dataset_final + '/images'):
        os.makedirs(path_dataset_final + '/images/train')
    if 'test' not in os.listdir(path_dataset_final + '/images'):
        os.makedirs(path_dataset_final + '/images/test')

    # Iterar sobre las particiones del conjunto de datos: entrenamiento y prueba
    for part in ['train', 'test']:
        # Seleccionar el CSV correspondiente (entrenamiento o prueba)
        csv = og_csv_train if part == 'train' else og_csv_test

        # Iterar sobre cada fila del CSV para procesar cada imagen
        for row in csv.iterrows():
            classCode = row[1]['classCode']                  # Código de clase de la imagen
            classDescription = row[1]['classDescription']    # Descripción de la clase
            im_file = row[1]['file']                         # Nombre del archivo de la imagen
            # Abrir la imagen desde el directorio original
            im = Image.open(f'/content/dataset/images/{part}/' + im_file)

            # Aplicar las transformaciones (data augmentation) a la imagen
            im_list_tr = im_aug_v2(im, grad)

            # Guardar cada imagen transformada en la carpeta final
            for im_tr in im_list_tr:
                # Generar un índice único para el nombre del archivo
                new_im_index = ("%05d" % (num[part]))
                file_name = f"{new_im_index}.jpg"            # Nombre del nuevo archivo

                # Crear una nueva fila con los datos de la imagen transformada
                df_row = pd.DataFrame({'file': [file_name],
                                       'classCode': classCode,
                                       'classDescription': classDescription})

                # Agregar la fila al DataFrame correspondiente (entrenamiento o prueba)
                if part == 'train':
                    train_df = pd.concat([train_df, df_row], ignore_index=True)
                else:
                    test_df = pd.concat([test_df, df_row], ignore_index=True)

                # Guardar la imagen transformada en el directorio final
                im_tr.save(path_dataset_final + "/" + 'images' +
                           "/" + part + "/" + file_name)

                # Incrementar el contador para la siguiente imagen
                num[part] += 1

        # Guardar los nuevos DataFrames como archivos CSV actualizados
        train_df.to_csv(path_dataset_final + '//train.csv', index=False)
        test_df.to_csv(path_dataset_final + '//test.csv', index=False)

# Directorio del dataset original y del nuevo
path_dataset_original = '/content/dataset'
path_dataset_final = '/content/dataset-aug'

# Ejecutar el código de aumento de datos
data_augmentation(path_dataset_original,path_dataset_final,2)

# Comprobación del número de imágenes en cada conjunto de datos
n_im_og_train = len(os.listdir('/content/dataset/images/train'))
n_im_og_test = len(os.listdir('/content/dataset/images/test'))

n_im_tr_train = len(os.listdir('/content/dataset-aug/images/train'))
n_im_tr_test = len(os.listdir('/content/dataset-aug/images/test'))

print("Train: Im og - Im tr  ", n_im_og_train,' - ',n_im_tr_train)
print("Test: Im og - Im tr  ", n_im_og_test,' - ',n_im_tr_test)

# Commented out IPython magic to ensure Python compatibility.
import os
import copy
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torchsummary import summary

from PIL import Image
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn

import torchvision
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid

# Modelos preentrenados
from efficientnet_pytorch import EfficientNet

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
pd.pandas.set_option('display.max_columns', 20)

# Función para guardar los gráficos generados en el cuaderno
# %matplotlib inline

THRESHOLD = 0.4

# Definir el directorio del conjunto de datos que se quiera usar
#DATA_DIR = '/content/dataset'
DATA_DIR = '/content/dataset-aug'

# Definir el directorio de los conjuntos de entrenamiento y test
TRAIN_DIR = DATA_DIR + '/images/train'
TEST_DIR = DATA_DIR + '/images/test'

"""
Definir directorio del archivo csv que contiene las etiquetas de los datos
de entrenamiento
"""
TRAIN_CSV = DATA_DIR + '/train.csv'

"""
Definir directorio del archivo csv que contiene las etiquetas de los datos
de test
"""
TEST_CSV = DATA_DIR + '/test.csv'

data_df = pd.read_csv(TRAIN_CSV)

# Relación de las etiquetas con un código
labels = {
    0: 'no_presencia',
    1: 'presencia'
}

def encode_label(label: list) -> list:
    """
    Función para convertir etiquetas en códigos (one-hot encoding)
    """

    target = torch.zeros(2)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target: list, text_labels: bool = False, threshold: float = THRESHOLD) -> str:
    """
    Función para convertir etiquetas numéricas obtenidas con las probabilidades
    de salida de la red neuronal en el correspondiente nombre de la etiqueta
    """

    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)

class GeneralDataset(Dataset):
    """
    Clase GeneralDataset para manejar conjuntos de datos personalizados para PyTorch.

    Este dataset está diseñado para trabajar con un DataFrame que contiene información
    sobre imágenes (nombres de archivo y etiquetas de clase). Permite aplicar
    transformaciones opcionales a las imágenes, y está diseñado para ser compatible
    con DataLoader de PyTorch.

    Atributos:
        df (DataFrame): DataFrame con información de las imágenes (e.g., nombre del archivo y clase).
        root_dir (str): Directorio raíz donde se encuentran las imágenes.
        transform (callable, opcional): Transformación que se aplicará a las imágenes (por defecto, None).
    """
    def __init__(self, df, root_dir, transform=None):
        """
        Inicializa el dataset.

        Parámetros:
            df (DataFrame): DataFrame con columnas que incluyen 'file' (nombre del archivo de imagen)
                            y 'classCode' (etiqueta de clase).
            root_dir (str): Ruta del directorio raíz donde se almacenan las imágenes.
            transform (callable, opcional): Transformación a aplicar a cada imagen (por defecto, None).
        """
        self.df = df  # Asignar el DataFrame al atributo de la clase
        self.transform = transform  # Asignar las transformaciones opcionales
        self.root_dir = root_dir  # Asignar el directorio raíz

    def __len__(self):
        """
        Devuelve el número total de elementos en el dataset.

        Retorna:
            int: Número total de filas en el DataFrame.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Recupera un elemento del dataset basado en su índice.

        Parámetros:
            idx (int): Índice del elemento que se desea obtener.

        Retorna:
            tuple: Una tupla que contiene:
                - img (PIL.Image): La imagen procesada.
                - label (int): La etiqueta codificada asociada a la imagen.
        """
        # Obtener la fila correspondiente del DataFrame según el índice
        row = self.df.loc[idx]

        # Extraer el nombre del archivo de imagen y su etiqueta de clase
        img_id, img_label = row['file'], row['classCode']

        # Construir la ruta completa del archivo de imagen
        img_fname = self.root_dir + "/" + str(img_id)

        # Cargar la imagen desde el archivo
        img = Image.open(img_fname)

        # Si se especificaron transformaciones, aplicarlas a la imagen
        if self.transform:
            img = self.transform(img)

        # Retornar la imagen y su etiqueta codificada (se asume que encode_label está definido)
        return img, encode_label(img_label)

# Visualizar el número de datos para cada clase
from itertools import chain
from collections import Counter

image_df = pd.read_csv(TRAIN_CSV)


all_labels = list(chain.from_iterable([i.strip().split(" ")
                                       for i in image_df['classDescription'].values]))

c_val = []
c_val = Counter(all_labels)

n_keys = c_val.keys()
max_idx = max(n_keys)

counts = pd.DataFrame({
    "Labels": [key for key in c_val.keys()],
    "Count": [val for val in c_val.values()]
})

counts.plot(x="Labels", y='Count', kind='barh', title='Class Imbalance')

counts.sort_values(by="Count", ascending=False).style.background_gradient(cmap='Reds')

# Función para calcular la media (mean) y la desviación estándar (std) del conjunto de datos
def mean_std(dirnames):
    """
    Calcula la media (mean) y la desviación estándar (std) de las imágenes RGB en un conjunto de directorios.

    Parámetros:
        dirnames (list): Lista de rutas a directorios que contienen las imágenes.

    Retorna:
        tuple: Una tupla que contiene:
            - mu_rgb (ndarray): Vector con la media (mean) de cada canal RGB.
            - std_rgb (ndarray): Vector con la desviación estándar (std) de cada canal RGB.
    """
    # Lista para almacenar las rutas de todas las imágenes
    imgs_path = []
    for dir in dirnames:
        # Obtener la lista de nombres de archivos en el directorio actual
        filenames = os.listdir(dir)
        # Crear las rutas completas para cada archivo y agregarlas a imgs_path
        for filename in filenames:
            imgs_path.append(os.path.join(dir, filename))

    # Cálculo de la media (mean)
    means = []
    # Procesar imágenes en intervalos (cada 100 imágenes para ahorrar tiempo y recursos)
    for img_path in imgs_path[0::100]:
        # Abrir la imagen, convertirla a un array y normalizar los valores de píxeles dividiendo entre 255
        mean = np.mean(np.array(Image.open(img_path).getdata()) / 255., axis=0)
        # Almacenar la media de la imagen
        means.append(mean)
    # Calcular la media global por cada canal RGB
    mu_rgb = np.mean(means, axis=0)

    # Cálculo de la varianza
    variances = []
    for img_path in imgs_path[0::100]:
        # Abrir la imagen, convertirla a un array y normalizar los valores de píxeles
        image_rgb = np.array(Image.open(img_path).getdata()) / 255.
        # Calcular la varianza por cada canal RGB de la imagen actual
        var = np.mean((image_rgb - mu_rgb) ** 2, axis=0)
        # Almacenar la varianza de la imagen
        variances.append(var)
    # Calcular la desviación estándar global (raíz cuadrada de la varianza promedio)
    std_rgb = np.sqrt(np.mean(variances, axis=0))

    # Retornar la media y la desviación estándar para cada canal RGB
    return mu_rgb, std_rgb

# Calcular la media y desviación estándar utilizando los directorios TRAIN_DIR y TEST_DIR
mean_list, std_list = mean_std([TRAIN_DIR, TEST_DIR])
# Mostrar los valores obtenidos
print(mean_list, std_list)

def image_transformations(image_size: int) -> (object, object):
    '''
        Retorna las transformaciones que se aplicarán.
        Entrada:
            image_size: int
        Salida:
            train_transformations: transformaciones que se aplicarán al conjunto de entrenamiento
            valid_tfms: transformaciones que se aplicarán al conjunto de validación o prueba
    '''

    mean = torch.tensor(mean_list)
    std = torch.tensor(std_list)

    # Transformaciones para el conjunto de entrenamiento
    train_trans = [
      T.Resize(image_size),  # Cambiar el tamaño de la imagen al especificado
      T.CenterCrop(image_size),  # Recortar la imagen centrada
      T.ToTensor(),  # Convertir la imagen a tensor
      T.Normalize(mean, std, inplace=True)  # Normalizar la imagen con la media y desviación estándar dadas
    ]

    # Transformaciones para el conjunto de validación
    val_trans = [
      T.Resize(image_size),  # Cambiar el tamaño de la imagen al especificado
      T.CenterCrop(image_size),  # Recortar la imagen centrada
      T.ToTensor(),  # Convertir la imagen a tensor
      T.Normalize(mean, std, inplace=True)  # Normalizar la imagen con la media y desviación estándar dadas
    ]

    train_transformations = T.Compose(train_trans)  # Componer las transformaciones de entrenamiento
    valid_tfms = T.Compose(val_trans)  # Componer las transformaciones de validación

    return train_transformations, valid_tfms

def get_train_dataset(image_size: int) -> (object, object):
    ''' Obtener el conjunto de datos de entrenamiento
        Entrada:
            image_size: int
        Salida:
            train_ds: objeto del conjunto de datos de entrenamiento
            val_ds: objeto del conjunto de datos de validación
    '''

    # Configurar la semilla para garantizar reproducibilidad
    np.random.seed(42)

    # Crear una máscara aleatoria para dividir los datos en entrenamiento y validación
    msk = np.random.rand(len(data_df)) < 0.9
    train_df = data_df[msk].reset_index()  # Datos de entrenamiento
    val_df = data_df[~msk].reset_index()  # Datos de validación

    # Obtener transformaciones según la arquitectura
    train_tfms, valid_tfms = image_transformations(image_size)

    # Obtener el conjunto de datos
    train_ds = GeneralDataset(train_df, TRAIN_DIR, transform=train_tfms)
    val_ds = GeneralDataset(val_df, TRAIN_DIR, transform=valid_tfms)
    return train_ds, val_ds

def get_train_dataloader(image_size: int, batch_size: int=64) -> (object, object):
    '''
        Retorna los cargadores de datos (dataloader) de entrenamiento y validación.
        Entrada:
            image_size: int
            batch_size: [opcional] int
        Salida:
            train_dl: objeto dataloader del conjunto de entrenamiento
            valid_dl: objeto dataloader del conjunto de validación
    '''

    # Obtener los conjuntos de datos de entrenamiento y validación
    train_ds, valid_ds = get_train_dataset(image_size)

    # Crear dataloaders para los conjuntos de datos
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                      num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2,
                    num_workers=2, pin_memory=True)

    return train_dl, valid_dl

def get_test_dataloader(image_size: int, batch_size: int=64) -> object:
    '''
        Retorna el cargador de datos (dataloader) del conjunto de prueba.
        Entrada:
            image_size: int
            batch_size: [opcional] int
        Salida:
            test_dl: objeto dataloader del conjunto de prueba
    '''

    # Leer el archivo CSV del conjunto de prueba
    test_df = pd.read_csv(TEST_CSV)

    # Obtener transformaciones, mismas que para el conjunto de validación
    _, valid_tfms = image_transformations(image_size)

    # Crear el conjunto de datos de prueba
    test_dataset = GeneralDataset(test_df, TEST_DIR, transform=valid_tfms)

    # Crear el dataloader para el conjunto de prueba
    test_dl = DataLoader(test_dataset, batch_size, num_workers=3, pin_memory=True)

    # Mover los datos al dispositivo (CPU o GPU)
    test_dl = DeviceDataLoader(test_dl, device)
    return test_dl

def get_default_device():
    """Seleccionar GPU si está disponible, de lo contrario usar CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Mover tensor(es) al dispositivo seleccionado"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Envolver un dataloader para mover datos a un dispositivo"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Generar un lote de datos después de moverlos al dispositivo"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Número de lotes"""
        return len(self.dl)

def F_score(output: list, label: list, threshold: float=THRESHOLD, beta: float=1.0) -> float:
    '''
        Retorna el F-score del modelo
        Entrada:
            output: array de salidas (outputs)
            label: array de etiquetas (labels)
            threshold: [opcional] float -> considerar la probabilidad de salida si está por encima del umbral
            beta: [opcional] float
        Salida:
            float -> F-score
    '''

    # Determinar las predicciones considerando el umbral
    prob = output > threshold
    label = label > threshold

    # Calcular verdaderos positivos (True Positives - TP)
    TP = (prob & label).sum(1).float()

    # Calcular verdaderos negativos (True Negatives - TN)
    TN = ((~prob) & (~label)).sum(1).float()

    # Calcular falsos positivos (False Positives - FP)
    FP = (prob & (~label)).sum(1).float()

    # Calcular falsos negativos (False Negatives - FN)
    FN = ((~prob) & label).sum(1).float()

    # Calcular precisión (precision)
    precision = torch.mean(TP / (TP + FP + 1e-12))

    # Calcular recall (sensibilidad)
    recall = torch.mean(TP / (TP + FN + 1e-12))

    # Calcular el F-score utilizando la fórmula con el parámetro beta
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    # Retornar el promedio del F-score
    return F2.mean(0)

# Modelos y parámteros de efficientnet

models = {
     "EfficientNet-b0": EfficientNet.from_pretrained('efficientnet-b0'),
     "EfficientNet-b1": EfficientNet.from_pretrained('efficientnet-b1'),
     "EfficientNet-b2": EfficientNet.from_pretrained('efficientnet-b2'),
     "EfficientNet-b3": EfficientNet.from_pretrained('efficientnet-b3'),
}

image_sizes = {
    "EfficientNet-b0": 224,
    "EfficientNet-b1": 240,
    "EfficientNet-b2": 260,
    'EfficientNet-b3': 300,
    'EfficientNet-b4': 380,
}

batch_sizes = {
    "EfficientNet-b0": 150,
    "EfficientNet-b1": 100,
    "EfficientNet-b2": 64,
    'EfficientNet-b3': 50,
    'EfficientNet-b4': 20
}

class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)                          # Generar predicciones
        loss = F.binary_cross_entropy(out, targets) # Calcular pérdida (loss)
        score = F_score(out, targets)               # Calcular la métrica F-score
        return {'loss': loss, 'score': score.detach()}

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)                           # Generar predicciones
        loss = F.binary_cross_entropy(out, targets)  # Calcular pérdida (loss)
        score = F_score(out, targets)                # Calcular la métrica F-score
        return {'val_loss': loss.detach(), 'val_score': score.detach() }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combinar las pérdidas (losses) de los lotes
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combinar las métricas F-score de los lotes
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def get_metrics_epoch_end(self, outputs, validation=True):
        if validation:
            loss_ = 'val_loss'
            score_ = 'val_score'
        else:
            loss_ = 'loss'
            score_ = 'score'

        # Combinar las pérdidas (losses) de los lotes
        batch_losses = [x[f'{loss_}'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        # Combinar las métricas F-score de los lotes
        batch_scores = [x[f'{score_}']
                        for x in outputs]
        epoch_scores = torch.stack(batch_scores).mean()

        return {f'{loss_}': epoch_loss.item(), f'{score_}': epoch_scores.item()}

    def epoch_end(self, epoch, result, epochs):
        # Mostrar el resumen al final de cada época
        print(f"Epoch: {epoch+1}/{epochs} -> last_lr: {result['lrs'][-1]:.4f}, train_loss: {result['loss']:.4f}, train_score: {result['score']:.4f}, val_loss: {result['val_loss']:.4f}, val_score: {result['val_score']:.4f}")

class GeneralModel(MultilabelImageClassificationBase):

    @staticmethod
    def get_sequential(num_ftrs):
        # Definir las capas secuenciales de la red neuronal
        linear_layers = nn.Sequential(
                nn.BatchNorm1d(num_features=num_ftrs),  # Normalización batch para las características de entrada
                nn.Linear(num_ftrs, 512),              # Capa totalmente conectada con 512 neuronas
                nn.ReLU(),                             # Activación ReLU
                nn.BatchNorm1d(512),                   # Normalización batch para las 512 características
                nn.Linear(512, 128),                   # Capa totalmente conectada con 128 neuronas
                nn.ReLU(),                             # Activación ReLU
                nn.BatchNorm1d(num_features=128),      # Normalización batch para las 128 características
                nn.Dropout(0.4),                       # Regularización dropout con una probabilidad de 0.4
                nn.Linear(128, 2),                     # Capa totalmente conectada con 2 salidas (etiquetas)
            )
        return linear_layers

    def __init__(self, model_name=None, model=None, input_size=None):
        super().__init__()

        # Usar un modelo preentrenado
        self.model_name = model_name
        self.model = copy.deepcopy(model)
        self.IS = input_size

        # Reemplazar la última capa del modelo preentrenado
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = GeneralModel.get_sequential(self.num_ftrs)

    def forward(self, xb):
        # Pasar los datos a través del modelo y aplicar una función sigmoide
        return torch.sigmoid(self.model(xb))

    def freeze(self):
        # Congelar (freeze) las capas residuales del modelo
        for param in self.model.parameters():
            param.require_grad = False

        # Asegurarse de que las capas completamente conectadas no estén congeladas
        for param in self.model._fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Descongelar (unfreeze) todas las capas del modelo
        for param in self.model.parameters():
            param.require_grad = True

    def __repr__(self):
        # Retornar una representación en cadena del modelo
        return f"{self.model}"

    def __str__(self):
        # Generar un resumen del modelo
        summary(self.model, (3, self.IS, self.IS))
        text_ = \
        f'''
            Model Name: {self.model_name}
            FC Layer input: {self.num_ftrs}
        '''
        return text_

@torch.no_grad()
def evaluate(model: object, val_loader: object) -> dict:
    '''
        Evalúa el modelo en el conjunto de validación
        Entrada:
            model: objeto del modelo entrenado
            val_loader: objeto del cargador de datos de validación
        Salida:
            métricas de validación
    '''

    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.get_metrics_epoch_end(outputs=outputs, validation=True)


def get_lr(optimizer: object) -> float:
    ''' Devuelve la tasa de aprendizaje actual'''

    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_model(model_name: str,
              model: object,
              epochs: int,
              lr: float,
              train_loader: object,
              val_loader: object,
              opt_func: object=torch.optim.SGD):
    '''
        Esta función se encarga de entrenar nuestro modelo.
        Usamos una política de tasa de aprendizaje One Cycle para actualizar nuestra tasa de aprendizaje
        con cada época.
        El mejor modelo se guarda durante cada época.
        Entrada:
            model_name: str -> Nombre del modelo
            model: object -> Objeto del modelo
            epochs: int -> Número máximo de épocas
            lr: float -> Tasa de aprendizaje
            train_loader: cargador de datos del conjunto de entrenamiento
            val_loader: cargador de datos del conjunto de validación
            opt_func: objeto optimizador
        Salida:
            history: lista de métricas
    '''

    torch.cuda.empty_cache()
    BEST_VAL_SCORE = 0.0  # Para hacer seguimiento del mejor puntaje del modelo
    history = []

    optimizer = opt_func(model.parameters(), lr)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=max_lr,
    #                                                 epochs=epochs,
    #                                                 steps_per_epoch=len(train_loader))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, cooldown=1)

    print(f"Transfer Learning -> Modelo: {model_name}")
    for epoch in range(epochs):

        # Descongelar (unfreeze) durante el último 50% de las épocas
        if epoch == (epochs // 2):
            model.unfreeze()
            print(f"Fine-tuning: {model_name}")
#             optimizer = opt_func(model.parameters(), unfreeze_max_lr,
#                                  weight_decay=weight_decay)

#             scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
#                                                             max_lr=unfreeze_max_lr,
#                                                             epochs=epochs//2,
#                                                             steps_per_epoch=len(train_loader))

        # Registrar las métricas de la época
        train_history = []
        lrs = []

        # Fase de entrenamiento
        model.train()

        for batch in tqdm(train_loader, desc=f'Época: {epoch+1}/{epochs}'):
            info = model.training_step(batch)
            loss = info['loss']
            # Contiene la pérdida y la precisión del lote para la fase de entrenamiento
            train_history.append(info)
            loss.backward()

            # Clipping de gradiente
            nn.utils.clip_grad_value_(model.parameters(), 1e-4)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            # scheduler.step()

        train_result = model.get_metrics_epoch_end(train_history, validation=False)
        val_result = evaluate(model, val_loader)
        result = {**train_result, **val_result}

        # Llamar al scheduler para verificar la pérdida de validación
        scheduler.step(result['val_score'])

        result['lrs'] = lrs
        model.epoch_end(epoch, result, epochs)

        # Guardar el mejor modelo
        if result['val_score'] > BEST_VAL_SCORE:
            BEST_VAL_SCORE = result['val_score']
            save_name = f"{model_name}_epoch-{epoch+1}_score-{round(result['val_score'], 4)}.pth"
            !rm -f '{model_name}'_*
            torch.save(model.state_dict(), save_name)

            # Generar predicciones usando el mejor modelo
            print("Generando predicciones en el conjunto de prueba")
            generate_prediction(model, image_sizes[model_name])

        history.append(result)
    return history


# Función para cargar el mejor modelo
def load_best(model_name: str) -> object:
    ''' Devuelve el mejor modelo'''

    model = models[model_name]
    image_size = image_sizes[model_name]
    best_model = GeneralModel(model_name, model, image_size)

    # Cargar pesos entrenados
    path = r"./"
    file_path = ''

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and i.startswith('EfficientNet'):
            file_path = os.path.join(path, i)

    print(f"Cargando modelo: {file_path[2:]}")
    best_model.load_state_dict(torch.load(file_path))
    # Mover el modelo a la GPU
    best_model = to_device(best_model, device)
    return best_model


@torch.no_grad()
def generate_prediction(model: object, image_size: int) -> None:
    '''Generar predicciones en el conjunto de prueba y crear un archivo csv'''

    test_dl = get_test_dataloader(image_size)

    model.eval()
    # Vaciar la caché de CUDA
    torch.cuda.empty_cache()

    # Generar predicciones
    batch_probs = []
    for xb, _ in test_dl:
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    test_preds = [decode_target(x) for x in batch_probs]

    # Generar archivo de envío
    submission_df = pd.read_csv(TEST_CSV)
    submission_df.Label = test_preds
    sub_fname = f'submission_{model_name}.csv'
    submission_df.to_csv(sub_fname, index=False)
    print(f"Archivo de predicción: {sub_fname} generado\n")


def end_to_end(model_name: str, parameters: dict=None) -> dict:
    '''
        Una función simple de entrenamiento y prueba de extremo a extremo en el modelo seleccionado.
        Entradas:
            model_name: str -> Nombre del modelo elegido
            parameters: dict -> Diccionario de hiperparámetros para el modelo
        Salidas:
            history: dict -> Diccionario que contiene las métricas del modelo (pérdida, puntaje, lr)

    '''
    torch.cuda.empty_cache()

    # Hiperparámetros
    image_size = image_sizes[model_name]
    BATCH_SIZE = batch_sizes[model_name]
    epochs = parameters["epochs"]
    lr = parameters["lr"]
    opt_func = parameters["opt_func"]

    # Obtener conjunto de datos transformado
    train_dl, valid_dl = get_train_dataloader(image_size, batch_size=BATCH_SIZE)
    # Mover el conjunto de datos para usar la GPU
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    # Obtener modelo
    model = models[model_name]
    model = GeneralModel(model_name, model, image_size)
    # Convertir a CUDA
    model = to_device(model, device)

    # Mover modelo a la GPU
    model = to_device(model, device)

    # Entrenar el modelo
    history = fit_model(
                model_name,
                model,
                epochs,
                lr,
                train_dl,
                valid_dl,
                opt_func
            )

    # Limpieza
    torch.cuda.empty_cache()

    return history

# Parámetros de entranamiento
training_parameters = {
    "epochs": 4,
    "lr": 0.001,
    "opt_func": torch.optim.Adam,
}

model_name = "EfficientNet-b0"

history = end_to_end(model_name, training_parameters)

# Representar en un gráfico las métricas de entrenamiento

def plot_accuracies(history):
    train_score = [r['score'] for r in history]
    val_score = [r['val_score'] for r in history]
    plt.plot(train_score, '-kx', label="train_score")
    plt.plot(val_score, '-rx', label="val_score")
    plt.legend()
    _ = plt.xticks(ticks=range(len(train_score)),
                   labels=[str(i) for i in range(1, len(train_score)+1)])
    plt.xlabel('epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs. epochs')

def plot_losses(history):
    train_losses = [r['loss'] for r in history]
    val_losses = [r['val_loss'] for r in history]
    plt.plot(train_losses, '-kx', label="train_loss")
    plt.plot(val_losses, '-rx', label="val_loss")
    plt.legend()
    _ = plt.xticks(ticks=range(len(train_losses)),
                   labels=[str(i) for i in range(1, len(train_losses)+1)])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. epochs')

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');

# Mostrar el gráfico de las métricas de entrenamiento
plt.figure(figsize=(25, 6))
plt.subplot(1, 3, 1)
plot_accuracies(history)
plt.subplot(1, 3, 2)
plot_losses(history)

plt.subplot(1, 3, 3)
plot_lrs(history)