# Basketball Analytics POC — Estado al 2026-06-15

## Qué hace este proyecto

Pipeline de visión por computadora que procesa video de básquet y detecta
eventos de juego (posesión, pase, tiro, canasta) sin ninguna anotación manual.

```
video.mp4 → YOLO11 (detección) → ByteTrack (tracking) → EventEngine → stats.json
```

Tecnologías principales: `supervision`, `ultralytics/YOLO11`, `PyAV`, `OpenCV`.

---

## Arquitectura de archivos

```
basketball_poc_v3.py   Pipeline principal — CLI, render, loop de frames
court_detector.py      Detección automática de las 4 esquinas de la cancha (HSV + Hough)
ball_tracker.py        Historial de posición de la pelota, velocidad, distancia al aro
event_engine.py        Máquina de estados: DEAD_BALL → POSSESSION → SHOT → BASKET
stats_collector.py     Acumula stats por jugador y equipo, exporta JSON
video_reader.py        Wrapper PyAV para leer AV1/H.264/H.265/VP9 (OpenCV no soporta AV1)
benchmark.py           Compara stats del programa contra box scores oficiales
ground_truth.json      Box scores oficiales de los 5 partidos descargados
run_all_benchmarks.sh  Corre el pipeline en los 5 videos y compara resultados
```

---

## Fixes implementados en esta sesión

### Fix 1 — Soporte AV1 (`video_reader.py`)
OpenCV instalado vía pip no incluye el codec AV1 (libdav1d). Los videos 1-3 y 5
son AV1 y se decodificaban en 0 frames silenciosamente.
**Solución:** `video_reader.py` usando PyAV (que sí incluye libdav1d).
Todos los lugares que usaban `sv.get_video_frames_generator` y
`sv.VideoInfo.from_video_path` fueron reemplazados.

### Fix 2 — Velocidades falsas de pelota (`ball_tracker.py`)
`_calc_speed()` promediaba posiciones incluyendo frames interpolados.
Cuando la pelota reaparecía después de un gap largo, la velocidad calculada
llegaba a 85 m/s generando falsos SHOT events.
**Solución:** filtrar solo muestras `detected=True`; resetear a 0 si el gap
entre dos detecciones reales supera 1 segundo; cap de 35 m/s.

### Fix 3 — Equipo B nunca detectado (`basketball_poc_v3.py`)
La función `assign_team()` usaba umbral fijo `> 110` en brillo de jersey.
En muchos clips todos los jugadores quedaban asignados a un solo equipo.
**Solución:** clase `TeamAssigner` con k-means adaptativo (k=2 sobre brillo
del recorte de torso). Los centros se re-ajustan cada vez que aparece un
jugador nuevo. Requiere mínimo 6 jugadores para activarse.

### Fix 4 — Orden ByteTrack / DetectionsSmoother (`basketball_poc_v3.py`)
El smoother se aplicaba *antes* de ByteTrack, por lo que nunca recibía
`tracker_id` y no podía suavizar trayectorias.
**Solución:** ByteTrack primero → smoother después.

### Fix 5 — Tracks efímeros con IDs muy altos (`basketball_poc_v3.py`)
ByteTrack perdía tracks rápido y asignaba IDs crecientes (100, 200, 300…).
**Solución:** `lost_track_buffer=fps*3` (3 segundos antes de dar track por
perdido) + `minimum_consecutive_frames=3` (evita tracks de 1 frame).

### Fix 6 — Detección adaptativa de cancha (`court_detector.py`)
Los rangos HSV fijos no coincidían con el color real del piso en todos
los videos. Los videos con cancha clara o inusual usaban el fallback
(bordes del frame) dando coordenadas incorrectas.
**Solución:** `_adaptive_hsv_range()` muestrea el color real del piso en
la zona central-inferior del frame y construye el rango HSV dinámicamente.
Los rangos fijos quedan como fallback.

### Fix 7 — Filtro de tamaño de pelota (`basketball_poc_v3.py`)
YOLO clase 32 detecta objetos redondos que no son pelotas (cabezas, señales).
**Solución:** filtro post-YOLO: alto 6–80px, aspect ratio 0.5–2.0.

### Fix 8 — Umbral de canasta más amplio (`event_engine.py`)
`BASKET_RIM_MAX_M = 0.8` era demasiado estricto dado el error de la
transformación de perspectiva (la cancha no es perfectamente plana en cámara).
**Solución:** subido a 1.5m.

### Nuevo: `--court-corners` override manual (`basketball_poc_v3.py`)
Cuando la detección automática falla, se pueden pasar las esquinas
directamente por CLI:
```bash
python basketball_poc_v3.py --source video.mp4 \
  --court-corners "64,684;1216,684;1216,36;64,36"
```
Orden: INF-IZQ ; INF-DER ; SUP-DER ; SUP-IZQ (en píxeles del frame).

---

## Benchmark — estado actual

Se corrió el pipeline en `video_1_9F5q5jCbrMI.mp4` (clip de 90s, Duke vs Clemson,
ACC Tournament SF 2026). Box score oficial: Duke 73 – Clemson 61.

| Métrica | Resultado | Ideal |
|---|---|---|
| Eventos detectados | 68 total | — |
| SHOT | 11 | ~detectables en el clip |
| BASKET | 0 | >0 |
| Dominancia tiros | Correcto (trivial) | Correcto |
| Dominancia canastas | Incorrecto (ambos 0) | Duke > Clemson |
| Error ratio tiros | 0.476 | < 0.15 |
| Equipo B eventos | 0 | ~50% |

El error de ratio de tiros (0.476) es el peor posible: el programa atribuye el
100% de los tiros a Equipo A porque Equipo B nunca se activa (k-means necesita
≥6 jugadores visibles simultáneamente, lo cual no se cumple en este clip corto).

---

## Limitaciones conocidas y próximos pasos

### Problema principal: detección de pelota muy esporádica
YOLO11n (y YOLO11s en pruebas) no detecta la pelota durante el arco del tiro —
cuando más importa para detectar canastas. La pelota es pequeña, rápida y
a veces motion-blurred.

**En curso → fine-tuning en GPU de Colab.** Ver [`COLAB_TRAINING.md`](./COLAB_TRAINING.md)
y el notebook [`train_basketball_yolo.ipynb`](./train_basketball_yolo.ipynb): entrena
`yolo11x` en un dataset de básquet de Roboflow (jugadores + pelota + aro) en GPU de
Colab. `basketball_poc_v3.py` ahora acepta `--player-class` y `--ball-class` para usar
el esquema de clases del modelo fine-tuneado (los defaults siguen siendo COCO 0/32).

El MCP `colab-mcp` se evaluó para manejar Colab desde el agente, pero **requiere correr
Claude Code localmente** (necesita el navegador del usuario), así que no funciona desde
la sesión web remota. La config quedó versionada en `.mcp.json` para uso local.

Otras opciones complementarias:
- Kalman filter dedicado para predecir trayectoria de la pelota entre detecciones
- Entrenar con `imgsz=1280` (A100) para mejor recall del objeto chico

### Problema secundario: TeamAssigner no converge en clips cortos
`MIN_PLAYERS = 6` no se alcanza cuando ByteTrack pierde tracks frecuentemente.
**Fix sugerido:** bajar a 4 jugadores mínimos + timeout de refit forzado.

### No hay distinción 2pt/3pt/FT
Requeriría conocer la posición del arco de 3 puntos en coordenadas de cancha,
lo cual requiere calibración más precisa o anotación manual de la línea.

### Los videos son clips, no partidos completos
Los conteos absolutos no son comparables con box scores oficiales.
Solo se pueden comparar ratios y dominancias relativas entre equipos.

---

## Uso rápido

```bash
# Procesar un video (modo interactivo)
python basketball_poc_v3.py --source video.mp4

# Batch / headless (sin ventanas)
python basketball_poc_v3.py --source video.mp4 --headless --max-frames 5400

# Con esquinas manuales (si la auto-detección falla)
python basketball_poc_v3.py --source video.mp4 --headless \
  --court-corners "64,684;1216,684;1216,36;64,36"

# Comparar contra box score oficial
python benchmark.py --stats video_1_9F5q5jCbrMI_stats.json --ground-truth ground_truth.json

# Correr todos los videos y benchmark
bash run_all_benchmarks.sh
```
