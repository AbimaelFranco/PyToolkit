import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import os

# =====================================================
# CONFIGURACIÓN
# =====================================================

VIDEO_ENTRADA = "baile.mp4"
VIDEO_TEMP = "resultado_sin_audio.mp4"
VIDEO_SALIDA = "resultado.mp4"

# =====================================================
# MODELO
# =====================================================

model = YOLO("yolo11n-pose.pt")

# =====================================================
# ESQUELETO COCO
# =====================================================

SKELETON = [
    (0,1),(0,2),
    (1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

LEFT = {5,7,9,11,13,15}
RIGHT = {6,8,10,12,14,16}

COLOR_LEFT = (0,0,255)       # rojo
COLOR_RIGHT = (0,255,0)      # verde
COLOR_CENTER = (255,255,255)

# =====================================================
# ABRIR VIDEO
# =====================================================

video = cv2.VideoCapture(VIDEO_ENTRADA)

if not video.isOpened():
    raise FileNotFoundError(f"No se pudo abrir {VIDEO_ENTRADA}")

fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    VIDEO_TEMP,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =====================================================
# PROCESAMIENTO
# =====================================================

frame_actual = 0
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_actual += 1

    print(f"\rProcesando frame {frame_actual}/{total_frames}", end="")

    results = model(frame, verbose=False)

    salida = frame.copy()

    for result in results:

        if result.keypoints is None:
            continue

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        areas = []

        for box in boxes:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

        idx = int(np.argmax(areas))

        persona = result.keypoints.xy.cpu().numpy()[idx]
        confianza = result.keypoints.conf.cpu().numpy()[idx]

        # Dibujar líneas

        for a, b in SKELETON:

            if confianza[a] < 0.5 or confianza[b] < 0.5:
                continue

            x1, y1 = persona[a]
            x2, y2 = persona[b]

            if a in LEFT and b in LEFT:
                color = COLOR_LEFT
            elif a in RIGHT and b in RIGHT:
                color = COLOR_RIGHT
            else:
                color = COLOR_CENTER

            cv2.line(
                salida,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                3,
                cv2.LINE_AA
            )

        # Dibujar puntos

        for i, (x, y) in enumerate(persona):

            if confianza[i] < 0.5:
                continue

            if i in LEFT:
                color = COLOR_LEFT
            elif i in RIGHT:
                color = COLOR_RIGHT
            else:
                color = COLOR_CENTER

            cv2.circle(
                salida,
                (int(x), int(y)),
                6,
                color,
                -1,
                cv2.LINE_AA
            )

    writer.write(salida)

video.release()
writer.release()

print("\nVideo procesado.")

# =====================================================
# RECUPERAR AUDIO ORIGINAL
# =====================================================

print("Agregando audio...")

# subprocess.run([
#     "ffmpeg",
#     "-y",
#     "-i", VIDEO_TEMP,
#     "-i", VIDEO_ENTRADA,
#     "-c:v", "copy",
#     "-c:a", "aac",
#     "-map", "0:v:0",
#     "-map", "1:a:0",
#     VIDEO_SALIDA
# ])

if os.path.exists(VIDEO_TEMP):
    os.remove(VIDEO_TEMP)

print(f"\nProceso terminado.")
# print(f"Video guardado en: {VIDEO_SALIDA}")
print(f"Video guardado en: {VIDEO_TEMP}")