"""
A partir del algoritmo de creacion del conjunto de datos,
visualizar en la imagen las zonas a recortar
Guardar las zonas recortadas automaticamente
para usarlas en el entrenamiento de la red neuronal
"""

import cv2
import numpy as np
import os
import copy

def undistort_image(img):
    """
    Corrige la distorsión de barril de una imagen utilizando parámetros definidos.
    """
    width = img.shape[1]
    height = img.shape[0]
    
    distCoeff = np.zeros((4, 1), np.float64)  # Coeficientes de distorsión.
    
    k1 = -2.2e-6  # Factor de distorsión radial.
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    
    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2
    
    cam = np.eye(3, dtype=np.float32)  # Matriz de cámara.
    cam[0, 2] = 5.0 * width / 10.0
    cam[1, 2] = height / 2.0
    cam[0, 0] = 10.0
    cam[1, 1] = 10.0
    
    dst = cv2.undistort(img, cam, distCoeff)  # Corrige la imagen.
    
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

def draw_lines(img, n_lineas, espaciado, color, thickness):
    """
    Dibuja líneas horizontales y una línea vertical en la imagen dada.

    Parámetros:
    - img: Imagen sobre la cual se dibujarán las líneas.
    - n_lineas: Número total de líneas horizontales que se dibujarán (excluyendo los bordes).
    - espaciado: Distancia entre cada línea horizontal.
    - color: Color de las líneas en formato BGR (por ejemplo, (255, 0, 0) para azul).
    - thickness: Grosor de las líneas.
    """

    # Dibujar las líneas horizontales desde la parte superior hacia abajo.
    for linea in range(1, n_lineas):
        # Línea horizontal desde la parte superior hacia abajo, con el espaciado especificado.
        cv2.line(img, (0, 0 + espaciado * linea), (1008, 0 + espaciado * linea), color, thickness)
        # Línea horizontal desde la parte inferior hacia arriba, con el mismo espaciado.
        cv2.line(img, (0, 760 - espaciado * linea), (1008, 760 - espaciado * linea), color, thickness)
    
    # Dibujar una línea vertical en el centro de la imagen.
    cv2.line(img, (504, 0), (504, 760), color, thickness)

def carpeta():
    """
    Crea una carpeta llamada 'raw_im' si no existe y determina el último número 
    de las imágenes almacenadas en una subcarpeta específica ('raw_im/OG_1/no_op').

    Retorna:
    - ultimo_numero: El número siguiente al último archivo encontrado en la lista,
                     o 0 si no hay archivos en la carpeta.
    """

    # Obtener el directorio de trabajo actual
    cwd = os.getcwd()

    # Verificar si la carpeta 'raw_im' no existe en el directorio actual
    if 'raw_im' not in os.listdir(cwd):
        # Crear la carpeta 'raw_im' en el directorio actual
        os.makedirs(cwd + 'raw_im')

    # Listar el contenido de la subcarpeta 'raw_im/OG_1/no_op'
    im_list = os.listdir('raw_im/OG_1/no_op')

    # Verificar si la lista de archivos en la subcarpeta no está vacía
    if len(im_list) > 0:
        # Obtener el último número de la lista de imágenes. 
        # Se asume que el nombre del archivo empieza con un número de 4 dígitos.
        ultimo_numero = int(im_list[-1][0:4]) + 1
    else:
        # Si la lista está vacía, asignar 0 como el primer número
        ultimo_numero = 0

    # Retornar el último número identificado o 0 si no hay archivos
    return ultimo_numero

def gen_coord(centro, area, grad):
    """
    Genera una lista de coordenadas en una cuadrícula centrada en un punto específico.

    Parámetros:
    - centro: Una tupla (x, y) que define el punto central de la cuadrícula.
    - area: Una tupla (ancho, alto) que define el tamaño del área donde se generará la cuadrícula.
    - grad: Número de divisiones (grado de la cuadrícula) en cada dirección.

    Retorna:
    - lista_coordenadas: Una lista de tuplas (x, y) que representan las coordenadas generadas.
    """
    # Calcular el espaciado entre las líneas de la cuadrícula en los ejes x e y
    espaciado_x = int(area[0] / grad)  # Distancia horizontal entre puntos
    espaciado_y = int(area[1] / grad)  # Distancia vertical entre puntos

    # Coordenadas iniciales (esquina superior izquierda) de la cuadrícula
    coord_x_0 = int(centro[0] - area[0] / 2)
    coord_y_0 = int(centro[1] - area[1] / 2)

    # Lista para almacenar las coordenadas generadas
    lista_coordenadas = []

    # Generar las coordenadas en la cuadrícula
    for n_espacio_x in range(0, grad + 1):  # Iterar en el eje x
        coord_x = coord_x_0 + espaciado_x * n_espacio_x  # Calcular la coordenada x actual
        for n_espacio_y in range(0, grad + 1):  # Iterar en el eje y
            coord_y = coord_y_0 + espaciado_y * n_espacio_y  # Calcular la coordenada y actual
            # Agregar la coordenada actual a la lista
            lista_coordenadas.append((coord_x, coord_y))

    # Retornar la lista de coordenadas generadas
    return lista_coordenadas


def aumento_coord(coordenadas, area, grad):
    """
    Aumenta una lista de coordenadas generando cuadrículas adicionales alrededor de cada coordenada.

    Parámetros:
    - coordenadas: Lista de coordenadas originales (tuplas (x, y)).
    - area: Una tupla (ancho, alto) que define el tamaño del área para generar cuadrículas adicionales.
    - grad: Número de divisiones (grado de la cuadrícula) en cada dirección.

    Retorna:
    - coordenadas_aug: Lista ampliada de coordenadas que incluye las cuadrículas generadas.
    """
    # Lista para almacenar las coordenadas ampliadas
    coordenadas_aug = []

    # Generar cuadrículas alrededor de cada coordenada original
    for coordenada in coordenadas:
        # Generar coordenadas adicionales usando gen_coord
        coord_aug = gen_coord(coordenada, area, grad)
        # Agregar las nuevas coordenadas a la lista ampliada
        coordenadas_aug = coordenadas_aug + coord_aug

    # Retornar la lista ampliada de coordenadas
    return coordenadas_aug

def visualizar_coordenadas(imagen, coordenadas, anchuras, espesor):
    """
    Dibuja rectángulos en las coordenadas especificadas para visualizar las áreas de interés.
    """
    for idx, (x, y) in enumerate(coordenadas):
        ancho = int(anchuras[idx] / 2)
        cv2.rectangle(imagen, (x-ancho, y-ancho), (x+ancho, y+ancho), (255, 0, 0), espesor)  # Dibuja el rectángulo.

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

def recorte_imagen(imagen_og, coordenadas, ancho):
    """
    Recorta zonas específicas de una imagen original y genera una lista de imágenes recortadas.

    Parámetros:
    - imagen_og: La imagen original de la que se realizarán los recortes.
    - coordenadas: Lista de tuplas (x, y) que representan las coordenadas centrales 
      de las zonas a recortar en la imagen original.
    - ancho: Ancho de las áreas cuadradas a recortar alrededor de cada coordenada.

    Retorna:
    - lista_imagenes: Lista de imágenes recortadas.
    """
    # Crear una lista vacía para almacenar las imágenes recortadas
    lista_imagenes = []

    # Iterar sobre cada coordenada en la lista de coordenadas
    for coordenada in coordenadas:
        # Realizar el recorte de la imagen en la coordenada específica
        # usando la función crop_zone, que recorta un área cuadrada
        imagen = crop_zone(imagen_og, coordenada[0], coordenada[1], ancho)
        # Agregar la imagen recortada a la lista
        lista_imagenes.append(imagen)

    # Devolver la lista de imágenes recortadas
    return lista_imagenes

def cambiar_orden_coordenadas(lista_coordenadas):
    """
    Intercambia la posición x por y en una lista de coordenadas.
    """
    lista_nueva = []
    for coordenada in lista_coordenadas:
        lista_nueva.append((coordenada[1], coordenada[0]))  # Cambia el orden de las coordenadas.
    return lista_nueva

# Coordenadas de las operaciones en la imagen
y=310 #altura doble linea superior
y_dl_in=1650
coordenadas = [#(530,y),(790,y),(1050,y),(1310,y),(1580,y),(1850,y),(2160,y),
(2440,y),(2700,y),(2960,y),(3240,y),(3500,y), # doble linea superior
(550,y_dl_in),(810,y_dl_in),(1070,y_dl_in),(1330,y_dl_in),(1595,y_dl_in),(1860,y_dl_in),
(2160,y_dl_in),(2440,y_dl_in),(2700,y_dl_in),(2960,y_dl_in),(3230,y_dl_in),(3490,y_dl_in), # doble linea inferior
(490,1000),(490,2040),(870,610),(880,2430),(3540,1000),(3140,610),(3527,2036),(3139,2424), # hundidura circular
(1722,484),(2584,484),(1726,2526),(2296,2526),#hundidura interior semicirculo
(1366,1002),(2658,1002),#pestanas
(1976,2352),#agujero solo
(1720,277),(2300,275), #hundidura lateral semicirculo
(1510,541), #agujero rectangular
(1890,2043), #agujero complejo
(1382,1795),(2048,1629),(1142,2311),(2646,2293),(2886,2311),(1136,738),(2888,732), #hundidura pestana
(35,1022),(3993,1028), #hundidura lateral rectangular
]

coordenadas = aumento_coord(coordenadas,(30,30),3)

def dataset_generator():
    """
    Función principal para la generación del conjunto de datos.
    El proceso consiste en capturar una imagen utilizando la cámara,
    identificar y recortar zonas específicas de interés basadas en
    coordenadas y dimensiones predefinidas, y finalmente almacenar
    las imágenes recortadas en una carpeta designada. Cada imagen se
    guarda con un nombre único, utilizando un código numérico secuencial
    para facilitar su identificación.
    """
    window_title = "CSI Camera"
    color = (0,0,255)
    thickness = 1
    n_lineas = 5
    espaciado = 30
    carpeta = 'raw_im/og_2/op/'
    ultimo_numero = 576
    ancho=224/2

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
                
                # Parar el programa con la tecla ESC o 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

                # Guardar recortes
                if keyCode == ord('c'):
                    # Visualizar zonas a recortar
                    frame = undistort_image(frame)
                    frame_c = copy.deepcopy(frame)
                    visualizar_coordenadas(frame_c,coordenadas,ancho,thickness)
                    frame_r = cv2.resize(frame_c,(1008,760))
                    draw_lines(frame,n_lineas,espaciado,color,thickness)
                    cv2.imshow(window_title, frame_r)
                    cv2.waitKey(0)
                    coordenadas_n = cambiar_orden_coordenadas(coordenadas)
                    lista_imagenes = recorte_imagen(frame,coordenadas_n,ancho)
                    
                    # Almacenar imágenes
                    for imagen in lista_imagenes:
                        cv2.imwrite(carpeta + ("%04d" % ultimo_numero) + '.jpg',imagen)
                        ultimo_numero += 1
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    dataset_generator()
