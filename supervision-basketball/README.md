# Basketball Analytics POC

Análisis de partidos de básquet a partir de video. Detecta jugadores,
pelota, equipos y eventos del juego (posesión, pase, tiro, canasta).

## Setup

```bash
# 1. Clonar el repo
git clone https://github.com/sebastianjlopez/pocs.git
cd pocs/supervision-basketball

# 2. Crear entorno virtual
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac / Linux:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Básico — solo pasarle el video
python basketball_poc_v3.py --source mi_video.mp4

# Guardar el video anotado
python basketball_poc_v3.py --source mi_video.mp4 --output resultado.mp4

# Con modelo más preciso (más lento)
python basketball_poc_v3.py --source mi_video.mp4 --weights yolo11x.pt
```

El modelo YOLO se descarga automáticamente la primera vez (~6 MB para yolo11n).

## Qué pasa cuando corrés el programa

1. **Detecta la cancha automáticamente** analizando los primeros frames del video
2. Muestra una preview con las esquinas detectadas — presioná **ENTER** para continuar
3. Procesa el video frame a frame mostrando en pantalla:
   - Jugadores marcados por equipo (blanco vs azul)
   - Trayectoria de la pelota
   - Velocidad de la pelota en m/s
   - Feed de eventos en tiempo real (POSESION / PASE / TIRO / CANASTA)
   - Mini-mapa de la cancha en la esquina
4. Al terminar muestra las **estadísticas por jugador** en consola y las guarda en JSON

## Controles durante el video

| Tecla | Acción |
|---|---|
| `Q` | Salir |
| `P` | Pausar / continuar |
| `S` | Guardar screenshot del frame actual |
| `H` | Activar / desactivar heatmap |
| `M` | Activar / desactivar mini-mapa |

## Archivos que genera

| Archivo | Contenido |
|---|---|
| `{video}_tracking.csv` | Posición de cada jugador en cada frame |
| `{video}_stats.json` | Estadísticas: posesiones, pases, tiros, canastas por jugador |
| `screenshot_NNNN.jpg` | Screenshots manuales (tecla S) |
| `resultado.mp4` | Video anotado (si usás --output) |

## Modelos YOLO disponibles

| Modelo | Velocidad | Precisión | Recomendado para |
|---|---|---|---|
| `yolo11n.pt` | Muy rápido | Básica | Prueba inicial, CPU lenta |
| `yolo11s.pt` | Rápido | Media | CPU normal |
| `yolo11m.pt` | Medio | Buena | Recomendado |
| `yolo11x.pt` | Lento | Máxima | GPU o si la precisión importa mucho |

## Descripción de los archivos

| Archivo | Qué hace |
|---|---|
| `basketball_poc_v3.py` | Programa principal — coordina todo |
| `court_detector.py` | Detecta las 4 esquinas de la cancha automáticamente |
| `ball_tracker.py` | Rastrea la pelota: velocidad, rebote, jugador más cercano |
| `event_engine.py` | Detecta eventos: posesión, pase, tiro, canasta |
| `stats_collector.py` | Acumula estadísticas por jugador y equipo |
| `calibrate_court.py` | Alternativa manual: click en 4 esquinas si la detección automática falla |

## Si la detección de cancha falla

La detección automática funciona mejor con canchas de madera natural (color naranja/tan).
Si tu video tiene una cancha pintada de otro color o la cámara está muy lejos, podés
calibrar manualmente:

```bash
python calibrate_court.py --source mi_video.mp4
# Hacé click en las 4 esquinas → guarda court_config.json

# Luego pasar la configuración al POC (editar basketball_poc_v3.py
# para cargar court_config.json en vez de usar CourtDetector)
```
