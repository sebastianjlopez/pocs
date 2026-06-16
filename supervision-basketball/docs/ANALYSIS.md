# Roboflow Supervision — Análisis para Basketball Analytics

## ¿Qué es `supervision`?

`supervision` es una librería open-source (MIT) de Roboflow que actúa como **capa de post-procesamiento y visualización** sobre cualquier modelo de detección/segmentación/pose. No es un modelo en sí mismo — es el toolkit que conecta la salida de YOLO, Transformers, MMDetection, RF-DETR, etc. con lógica analítica reutilizable.

---

## Módulos principales y lo que ofrecen

### 1. `Detections` — Núcleo de datos
El objeto universal que estandariza la salida de cualquier modelo:
```
xyxy, confidence, class_id, tracker_id, mask, data, metadata
```
Connectors nativos: `from_ultralytics`, `from_inference`, `from_transformers`, `from_mmdetection`, etc.

### 2. Annotators (`sv.BoxAnnotator`, `sv.TraceAnnotator`, etc.)
23 anotadores distintos, todos composables:
- `BoxAnnotator`, `EllipseAnnotator`, `CircleAnnotator`, `DotAnnotator`
- `TraceAnnotator` — dibuja trayectorias históricas
- `HeatMapAnnotator` — mapa de calor de presencia
- `LabelAnnotator`, `RichLabelAnnotator` — texto + íconos
- `PercentageBarAnnotator` — barra de porcentaje por detección
- `TriangleAnnotator` — triángulo debajo del objeto
- `BlurAnnotator`, `PixelateAnnotator` — privacidad

### 3. ByteTrack (`sv.ByteTrack`)
Tracker multi-objeto integrado (ByteTrack + Kalman Filter). Asigna `tracker_id` persistente frame-a-frame.

### 4. Zonas (`PolygonZone`, `LineZone`)
- `PolygonZone`: define un polígono arbitrario y filtra/cuenta detecciones dentro de él
- `LineZone`: detecta cruces de línea (en/fuera) con soporte multiclase
- Ambas aceptan **anchors configurables** (BOTTOM_CENTER, CENTER, TOP_LEFT, etc.)

### 5. ViewTransformer (homografía)
Transforma coordenadas de imagen a coordenadas del mundo real mediante perspectiva:
```python
view_transformer = ViewTransformer(source=SOURCE_POINTS, target=TARGET_POINTS)
real_coords = view_transformer.transform_points(pixel_coords)
```
Usado en speed estimation para convertir píxeles → metros.

### 6. KeyPoints (`sv.KeyPoints`)
Soporte para pose estimation con esqueletos COCO/GHUM. Annotators: `VertexAnnotator`, `EdgeAnnotator`, `VertexEllipseAnnotator`.

### 7. `DetectionsSmoother`
Suaviza detecciones ruidosas a lo largo de frames para reducir flickering.

### 8. `InferenceSlicer` (SAHI)
Divide imágenes grandes en tiles para detectar objetos pequeños — útil en tomas aéreas o canchas completas.

### 9. Métricas (`MeanAveragePrecision`, `F1Score`, `ConfusionMatrix`, etc.)
Suite completa de evaluación estilo COCO para benchmarking de modelos.

### 10. Dataset tools
Lectura/escritura de COCO, Pascal VOC, YOLOv8, CVAT, Albumentations.

---

## Aplicaciones directas al Básquet

| Feature de Supervision | Caso de uso Básquet | Factibilidad |
|---|---|---|
| `ByteTrack` + `TraceAnnotator` | Tracking de jugadores y pelota frame-a-frame | Alta |
| `ViewTransformer` (homografía) | Mapear posiciones de la cancha TV → mapa 2D (28m x 15m) | Alta |
| `PolygonZone` | Zonas tácticas: pintura, perímetro 3pt, media cancha | Alta |
| `HeatMapAnnotator` | Mapa de calor de presencia por jugador/equipo | Alta |
| `LineZone` | Detección de cruces (p.ej. transición ataque/defensa) | Media-Alta |
| `PercentageBarAnnotator` | Mostrar % de tiempo en zona por jugador | Media |
| `DetectionsSmoother` | Reducir ruido en detecciones de cámaras TV | Alta |
| `KeyPoints` + skeletons COCO | Análisis de postura (rebotes, tiros, defensas) | Media |
| `InferenceSlicer` | Tomas aéreas o drones de cancha completa | Alta |
| `CSVSink` / `JSONSink` | Exportar coordenadas para análisis estadístico | Alta |
| `PolygonZone.current_count` | Conteo de jugadores en la pintura en tiempo real | Alta |

---

## Pipeline propuesto para Basketball Analytics

```
Video (TV/drone)
    ↓
Modelo YOLO (detección jugadores + pelota)   ←── fine-tuned en basketball
    ↓
sv.Detections (estandarización)
    ↓
sv.ByteTrack (tracking persistente, tracker_id por jugador)
    ↓
ViewTransformer (perspectiva → coordenadas cancha real)
    ↓
┌─────────────────────────────────────────────┐
│  PolygonZone (pintura, 3pt, media cancha)   │
│  HeatMap por jugador/equipo                 │
│  LineZone (transiciones)                    │
│  Speed estimation (velocidad en m/s)        │
│  Time-in-zone (tiempo en cada zona)         │
└─────────────────────────────────────────────┘
    ↓
CSVSink / JSONSink (datos para estadísticas)
    +
VideoSink (video anotado)
```

---

## Lo que NO cubre supervision (y requiere trabajo adicional)

1. **Modelo de detección especializado**: Necesitás entrenar o usar un YOLO fine-tuned para detectar jugadores de básquet + pelota + árbitros con camisetas diferenciadas por equipo.
2. **Re-identificación (ReID)**: ByteTrack pierde el ID cuando un jugador sale del encuadre. Para mantener el número de camiseta necesitás un modelo ReID adicional (p.ej. OSNet, FastReID).
3. **Detección de equipo por color**: Hay que agregar un clasificador de jersey-color o cluster por color sobre los crops de jugadores.
4. **Eventos del juego**: Detección de posesión de pelota, tiros, rebotes requiere lógica adicional sobre las coordenadas.
5. **Calibración de cámara**: ViewTransformer requiere calibrar SOURCE_POINTS manualmente por cada cámara/ángulo.

---

## Conclusión

`supervision` es **perfectamente aplicable** a básquet y elimina gran parte del boilerplate:
- El tracking, anotación, zonas, heatmaps y exportación de datos están listos
- El gap principal es el **modelo de detección fine-tuned** y la **re-identificación de jugadores**
- El ViewTransformer es la pieza clave para analytics espaciales (posiciones en la cancha real)

Roboflow tiene modelos pre-entrenados de básquet en `roboflow.com/universe` que se pueden usar como base.
