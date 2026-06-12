# Basketball Analytics POC

Pipeline completo para análisis de partidos de básquet:
- Detección con YOLOv11
- Tracking con ByteTrack (supervision)
- Asignación de equipos con CLIP (zero-shot)
- Proyección a coordenadas de cancha real (ViewTransformer)
- Zonas tácticas, velocidad, heatmap, tiempo en zona
- Exportación CSV + video anotado

## Setup

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar video de prueba
python download_video.py
```

## Uso

### Modo rápido (sin calibración)
```bash
python basketball_poc_v2.py --source test_basketball.mp4
```

### Con video de salida guardado
```bash
python basketball_poc_v2.py --source test_basketball.mp4 --output output.mp4
```

### Calibrar las esquinas de la cancha (recomendado)
```bash
python calibrate_court.py --source test_basketball.mp4
```
Hacé click en las 4 esquinas de la cancha en el orden que indica la pantalla.
El script guarda las coordenadas en `court_config.json`.

### Con calibración personalizada
```bash
python basketball_poc_v2.py --source test_basketball.mp4 --court-config court_config.json
```

### Especificar colores de equipos (mejora CLIP)
```bash
python basketball_poc_v2.py \
  --source test_basketball.mp4 \
  --team-a "white jersey basketball player" \
  --team-b "blue jersey basketball player"
```

## Controles durante la ejecución
- `Q` — salir
- `H` — toggle heatmap
- `M` — toggle mini-mapa
- `S` — guardar screenshot del frame actual
- `P` — pausar / continuar

## Output
- `{nombre_video}_tracking.csv` — datos de tracking frame a frame
- `{nombre_video}_zone_summary.json` — resumen de tiempo en zona por jugador
- `{nombre_video}_output.mp4` — video anotado (si se especifica --output)

## Notas de calibración
Para el ViewTransformer necesitás marcar las 4 esquinas de la cancha visible
en el frame (no tiene que ser la cancha completa — puede ser una mitad).
El orden es: inferior-izquierda, inferior-derecha, superior-derecha, superior-izquierda.

Las dimensiones reales de una cancha NBA son 28.65m × 15.24m.
Si solo ves media cancha, usar 14.325m × 15.24m.
