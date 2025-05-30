from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO
model = YOLO(r"D:\Detector_Emociones\best.pt")

# Iniciar la cámara (0 para webcam integrada)
cap = cv2.VideoCapture(0)

# Verificamos si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Realizar inferencia
    results = model.predict(source=frame, save=False, conf=0.5, show=False)

    # Dibujar los resultados sobre el frame
    annotated_frame = results[0].plot()

    # Mostrar el frame
    cv2.imshow("Detección en Tiempo Real - YOLO", annotated_frame)

    # Salir al presionar 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

