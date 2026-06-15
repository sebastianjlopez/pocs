#!/usr/bin/env bash
# run_all_benchmarks.sh
# Corre el pipeline en los 5 videos en modo headless y luego ejecuta el benchmark.
#
# Uso:
#   bash run_all_benchmarks.sh
#   bash run_all_benchmarks.sh --weights yolo11x.pt   (modelo más preciso, más lento)
#
# Requisitos:
#   pip install ultralytics supervision opencv-python
#   Los archivos de video deben estar en el mismo directorio.

set -euo pipefail

WEIGHTS="${1:---weights}"
WEIGHTS_VAL="${2:-yolo11n.pt}"

# Si no se pasó --weights, usar el default
if [[ "$WEIGHTS" != "--weights" ]]; then
    WEIGHTS_VAL="$WEIGHTS"
    WEIGHTS="--weights"
fi

GROUND_TRUTH="ground_truth.json"
REPORT_JSON="benchmark_report.json"

VIDEOS=(
    "video_1_9F5q5jCbrMI.mp4"
    "video_2_k49qTm4Low.mp4"
    "video_3_uJs693eNfuQ.mp4"
    "video_4_NuXMh3PCyR0.mp4"
    "video_5_lryAoahRZQk.mp4"
)

echo "========================================================================"
echo "  Basketball Analytics POC — Benchmark Runner"
echo "  Modelo: $WEIGHTS_VAL"
echo "========================================================================"
echo ""

# ── Paso 1: correr el pipeline en cada video ──────────────────────────────────
for VIDEO in "${VIDEOS[@]}"; do
    if [[ ! -f "$VIDEO" ]]; then
        echo "⚠  $VIDEO no encontrado — saltando."
        continue
    fi

    STEM="${VIDEO%.mp4}"
    STATS_FILE="${STEM}_stats.json"

    if [[ -f "$STATS_FILE" ]]; then
        echo "✓  $STATS_FILE ya existe — reutilizando (borrá el archivo para re-procesar)."
        continue
    fi

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Procesando: $VIDEO"
    echo "────────────────────────────────────────────────────────────────────"

    python basketball_poc_v3.py \
        --source "$VIDEO" \
        "$WEIGHTS" "$WEIGHTS_VAL" \
        --no-heatmap \
        --no-minimap \
        --headless

    if [[ -f "$STATS_FILE" ]]; then
        echo "  ✓ Stats exportados: $STATS_FILE"
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

python benchmark.py \
    --all \
    --ground-truth "$GROUND_TRUTH" \
    --output "$REPORT_JSON"

echo ""
echo "Reporte JSON guardado en: $REPORT_JSON"
echo "Done."
