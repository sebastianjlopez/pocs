#!/usr/bin/env bash
# run_all_benchmarks.sh
# Corre el pipeline en los 5 videos en modo headless y luego ejecuta el benchmark.
#
# Uso (desde la raíz del repo):
#   bash scripts/run_all_benchmarks.sh
#   bash scripts/run_all_benchmarks.sh models/yolo11x.pt
#
# Override de directorio de videos:
#   VIDEOS_DIR=/ruta/a/videos bash scripts/run_all_benchmarks.sh
#
# Requisitos:
#   pip install ultralytics supervision opencv-python

set -euo pipefail

WEIGHTS="${1:-models/yolo11n.pt}"
GROUND_TRUTH="data/ground_truth.json"
REPORT_JSON="outputs/benchmark_report.json"
VIDEOS_DIR="${VIDEOS_DIR:-videos}"

VIDEOS=(
    "video_1_9F5q5jCbrMI.mp4"
    "video_2_k49qTm4Low.mp4"
    "video_3_uJs693eNfuQ.mp4"
    "video_4_NuXMh3PCyR0.mp4"
    "video_5_lryAoahRZQk.mp4"
)

mkdir -p outputs

echo "========================================================================"
echo "  Basketball Analytics — Benchmark Runner"
echo "  Modelo: $WEIGHTS"
echo "========================================================================"
echo ""

# ── Paso 1: correr el pipeline en cada video ──────────────────────────────────
for VIDEO in "${VIDEOS[@]}"; do
    # Buscar video en VIDEOS_DIR o en raíz
    if [[ -f "$VIDEOS_DIR/$VIDEO" ]]; then
        SRC="$VIDEOS_DIR/$VIDEO"
    elif [[ -f "$VIDEO" ]]; then
        SRC="$VIDEO"
    else
        echo "⚠  $VIDEO no encontrado — saltando."
        continue
    fi

    STEM="${VIDEO%.mp4}"
    STATS_FILE="outputs/${STEM}_stats.json"

    if [[ -f "$STATS_FILE" ]]; then
        echo "✓  $STATS_FILE ya existe — reutilizando (borrá para re-procesar)."
        continue
    fi

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Procesando: $VIDEO"
    echo "────────────────────────────────────────────────────────────────────"

    python scripts/run_pipeline.py \
        --source "$SRC" \
        --weights "$WEIGHTS" \
        --no-heatmap \
        --no-minimap \
        --headless

    # El pipeline guarda stats en la raíz; moverlos a outputs/
    [[ -f "${STEM}_stats.json"    ]] && mv "${STEM}_stats.json"    "$STATS_FILE"
    [[ -f "${STEM}_tracking.csv"  ]] && mv "${STEM}_tracking.csv"  "outputs/${STEM}_tracking.csv"

    if [[ -f "$STATS_FILE" ]]; then
        echo "  ✓ Stats: $STATS_FILE"
    else
        echo "  ✗ No se generó $STATS_FILE — revisar errores arriba."
    fi
    echo ""
done

# ── Paso 2: correr el benchmark ───────────────────────────────────────────────
echo "========================================================================"
echo "  Ejecutando benchmark.py ..."
echo "========================================================================"
echo ""

python scripts/benchmark.py \
    --all \
    --ground-truth "$GROUND_TRUTH" \
    --output "$REPORT_JSON"

echo ""
echo "Reporte JSON: $REPORT_JSON"
echo "Done."
