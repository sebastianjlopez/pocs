# Mejorar el modelo con GPU de Colab

Guía para fine-tunear un YOLO11x especializado en básquet y conectarlo al POC,
usando GPU de Google Colab — sin depender del hardware local.

## Por qué

YOLO11 genérico (COCO) detecta `person` y `sports ball`, pero:
- **Pierde la pelota en el arco del tiro** (chica, rápida, con motion-blur) → 0 canastas.
- No distingue árbitro de jugador ni el aro.

Un modelo fine-tuneado en un dataset de básquet resuelve ambos. Como `yolo11x`
es pesado para entrenar en CPU, lo entrenamos en GPU de Colab.

---

## Opción A — Entrenar a mano (no requiere MCP)

1. Subí `train_basketball_yolo.ipynb` a [Google Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → GPU** (T4 alcanza; A100 permite `imgsz=1280`).
3. Conseguí una API key gratuita de [Roboflow](https://app.roboflow.com) (Settings → API)
   y pegala en la celda 3.
4. Ejecutá todas las celdas. Al final descargás `best.pt`.
5. Anotá los `class_id` que imprime la celda 4 (orden de `names` en `data.yaml`).

---

## Opción B — Manejar Colab desde Claude Code con `colab-mcp`

`colab-mcp` ([repo](https://github.com/googlecolab/colab-mcp)) hace de puente entre
Claude Code y una sesión de Colab **abierta en tu navegador**.

> ⚠️ **Tiene que correr en tu máquina local.** El servidor necesita acceso a tu
> navegador y a tu sesión de Colab, así que **no funciona desde Claude Code on the
> web** (este entorno remoto). Usá el CLI de Claude Code instalado localmente.

### Setup

1. Instalá [`uv`](https://docs.astral.sh/uv/) (provee `uvx`).
2. Instalá el CLI de Claude Code en tu compu y abrí este repo localmente.
3. La config ya está versionada en [`.mcp.json`](./.mcp.json):
   ```json
   {
     "mcpServers": {
       "colab-mcp": {
         "command": "uvx",
         "args": ["git+https://github.com/googlecolab/colab-mcp"],
         "timeout": 30000
       }
     }
   }
   ```
   Claude Code lo detecta automáticamente al iniciar en el directorio del proyecto
   (te va a pedir aprobar el server MCP del proyecto la primera vez).
   Verificá con `/mcp` que `colab-mcp` aparezca como conectado.
4. Abrí un notebook nuevo en Colab en tu navegador con runtime **GPU**.
5. Desde Claude Code local, pedile que abra/ejecute `train_basketball_yolo.ipynb`
   en Colab vía el MCP, o que corra celdas de entrenamiento directamente.

---

## Integrar los pesos al POC

Una vez que tenés `best.pt` y conocés los `class_id` del dataset:

```bash
python basketball_poc_v3.py --source video.mp4 --weights best.pt \
    --player-class <ID_player> --ball-class <ID_ball>
```

> COCO usa `player=0`, `ball=32` (defaults). Un modelo de Roboflow suele tener
> otro orden — por ejemplo `0: ball, 1: player, 2: referee, 3: hoop`, en cuyo caso
> sería `--player-class 1 --ball-class 0`. Siempre confirmá con el `data.yaml`.

Después corré el benchmark para comparar contra el modelo viejo:

```bash
python basketball_poc_v3.py --source video_1_9F5q5jCbrMI.mp4 --headless \
    --weights best.pt --player-class 1 --ball-class 0
python benchmark.py --stats video_1_9F5q5jCbrMI_stats.json --ground-truth ground_truth.json
```

La meta es que **BASKET > 0** y que el ratio de tiros baje del 0.476 actual.

---

## Notas

- El dataset de ejemplo en el notebook es un placeholder de Roboflow Universe;
  buscá ["basketball" en Universe](https://universe.roboflow.com/search?q=basketball)
  y elegí el que mejor cubra pelota + aro si querés detectar canastas.
- Para mejor recall de la pelota, entrená con `imgsz=1280` (necesita A100) y/o
  un dataset con muchas instancias de pelota en vuelo.
- Guardá `best.pt` en Drive desde el notebook para no perderlo al cerrar Colab.
