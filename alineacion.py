import cv2

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

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


def alineacion():
    """
    Visualizar el centro geométrico de la imagen a través de
    dos líneas transversales, una vertical y otra horizontal.
    Además, se muestran líneas horizontales simétricas cerca del
    borde superior e inferior para ayudar en el proceso de ajuste
    de la posición de la cámara respecto de la pieza.
    """
    window_title = "CSI Camera"
    color = (0,0,255)
    thickness = 1
    n_lineas = 5
    espaciado = 30

    # Configuración de la cámara
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
		    #n_frame = cv2.resize(frame,(1008,760))
                    # Dibujar líneas de referencia en la imagen
                    for linea in range(1,n_lineas):
                        cv2.line(frame, (0,0+espaciado*linea), (1008,0+espaciado*linea), color, thickness)
                        cv2.line(frame, (0,760-espaciado*linea), (1008,760-espaciado*linea), color, thickness)
                    cv2.line(frame, (504,0), (504,760), color, thickness)
                    cv2.line(frame, (0,380), (1008,380), color, thickness)
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Parar el programa con la tecla ESC o 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

		# Guardar imagen
                if keyCode == ord('c'):
                    cv2.imwrite('raw_im/captura.jpg',frame)
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    alineacion()
