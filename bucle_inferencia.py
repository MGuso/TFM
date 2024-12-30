import os
from PIL import Image
import torch
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
import copy
import cv2
from torchvision import transforms
import numpy as np
import model_utils

def inferencia(lista_imagenes):
    """
    Ejecutar la inferencia sobre un lote de imágenes dado.
    Utiliza un modelo previamente cargado para calcular las probabilidades de las clases.
    """
    batch = model_utils.to_device(lista_imagenes, device)  # Mueve el lote de imágenes al dispositivo (CPU o GPU).
    with torch.no_grad():  # Desactiva el cálculo del gradiente para reducir el consumo de memoria.
        output = torch.nn.functional.softmax(model(batch), dim=1)  # Calcula las probabilidades de las clases.
    return output

def visualizar_recortes(imagen_og, coordenadas, ancho):
    """
    Recorta las regiones de interés de la imagen original basadas en las coordenadas
    y el ancho proporcionados, y devuelve una lista de las imágenes recortadas.
    """
    lista_imagenes = []
    for idx, coordenada in enumerate(coordenadas):
        imagen = crop_zone(imagen_og, coordenada[0], coordenada[1], ancho[idx])  # Recorta la región.
        lista_imagenes.append(imagen)  # Añade la imagen recortada a la lista.
    return lista_imagenes

def cambiar_orden_coordenadas(lista_coordenadas):
    """
    Intercambia la posición x por y en una lista de coordenadas.
    """
    lista_nueva = []
    for coordenada in lista_coordenadas:
        lista_nueva.append((coordenada[1], coordenada[0]))  # Cambia el orden de las coordenadas.
    return lista_nueva

def crop_zone(image, height_center, width_center, wide):
    """
    Recorta una región cuadrada de una imagen dada la posición central y el ancho.
    """
    expansion = wide / 2.0
    left = width_center - expansion
    top = height_center - expansion
    right = width_center + expansion
    bottom = height_center + expansion
    image = image.crop((left, top, right, bottom))  # Recorta la imagen en base a las coordenadas.
    return image

def visualizar_coordenadas(imagen, coordenadas, anchuras, espesor):
    """
    Dibuja rectángulos en las coordenadas especificadas para visualizar las áreas de interés.
    """
    for idx, (x, y) in enumerate(coordenadas):
        ancho = int(anchuras[idx] / 2)
        cv2.rectangle(imagen, (x-ancho, y-ancho), (x+ancho, y+ancho), (255, 0, 0), espesor)  # Dibuja el rectángulo.

def mostrar_resultados(frame, coordenadas, output):
    """
    Muestra los resultados de la inferencia sobre una imagen con rectángulos coloreados
    según las predicciones de las clases.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convierte el formato de color a BGR.
    for idx, (x, y) in enumerate(coordenadas):
        if output[idx][0].item() > output[idx][1].item():  # Clasificación según la probabilidad más alta.
            color = (0, 0, 255)  # Rojo si pertenece a la clase 0.
        else:
            color = (0, 255, 0)  # Verde si pertenece a la clase 1.
        ancho_mitad = int(lista_anchuras[idx] / 2)
        cv2.rectangle(frame, (x-ancho_mitad, y-ancho_mitad), (x+ancho_mitad, y+ancho_mitad), color, thickness)  # Dibuja el rectángulo.
    frame = cv2.resize(frame, (1008, 760))  # Ajusta el tamaño de la imagen.
    cv2.imshow('resultados', frame)  # Muestra la imagen con resultados.
    cv2.imwrite('resultados_loop.jpg', frame)  # Guarda la imagen con los resultados.

def load_batch(im_path_list, lib_read, mean, std):
    """
    Carga y preprocesa un lote de imágenes según los parámetros de normalización.
    """
    trans = [
        transforms.Resize(image_size),  # Ajusta el tamaño de la imagen.
        transforms.CenterCrop(image_size),  # Realiza un recorte centrado.
        transforms.ToTensor(),  # Convierte la imagen en un tensor.
        transforms.Normalize(mean, std, inplace=True)  # Normaliza la imagen.
    ]
    transf = transforms.Compose(trans)

    # Carga y transforma las imágenes según la biblioteca utilizada (PIL o OpenCV).
    if lib_read == 'pil':
        batch = torch.stack([transf(im) for im in im_path_list]).to(device)
    if lib_read == 'cv2':
        batch = torch.stack([transf(im) for im in im_path_list]).to(device)

    return batch

def undistort_image(img):
    """
    Corrige la distorsión de barril de una imagen utilizando parámetros definidos.
    """
    width  = img.shape[1]
    height = img.shape[0]

    distCoeff = np.zeros((4,1),np.float64)

    k1 = -2.3e-6
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = 5.0*width/10.0
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y

    dst = cv2.undistort(img,cam,distCoeff)

    return dst

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=4032,
    capture_height=3040,
    display_width=4032,
    display_height=3040,
    framerate=1,
    flip_method=0,
):
    """
    Construye una cadena de GStreamer para capturar imágenes desde la cámara.
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Configuración y carga del modelo
model_name = "EfficientNet-b0"
models = {"EfficientNet-b0": EfficientNet.from_pretrained('efficientnet-b0')}
image_sizes = {"EfficientNet-b0": 224}
image_size = image_sizes[model_name]
model = models[model_name]
model = model_utils.GeneralModel(model_name, model, image_size)
device = model_utils.get_default_device()
model.load_state_dict(torch.load('Modelos/m_va_3.pth', map_location=device))  # Carga los pesos del modelo.
model.eval()  # Establece el modelo en modo de evaluación.
model = model_utils.to_device(model, device)  # Mueve el modelo al dispositivo.

# Coordenadas y parámetros de recorte
# (Ver definiciones detalladas en el código original...)

def inference_loop():
    """
    Bucle principal de inferencia. Captura imágenes de la cámara,
    permite recortar zonas de interés y ejecuta la inferencia sobre estas.
    """
    # Configuración de la cámara
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    pass
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):  # Salir del programa con la tecla 'q'.
                    break
                if keyCode == ord('c'):  # Captura de imagen con la tecla 'c'.
                    # Procesamiento de la imagen capturada
                    # (Ver detalles en el código original...)
        finally:
            video_capture.release()  # Libera la cámara.
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    inference_loop()
