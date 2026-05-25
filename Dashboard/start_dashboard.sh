#!/bin/bash
# Inicia el dashboard de ICFES en segundo plano en el puerto 8501.
# Ejecutar desde la raíz del proyecto: bash start_dashboard.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activar el entorno virtual del proyecto
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: No se encontró venv/. Crea el entorno con: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi
source venv/bin/activate

# Verificar que los resultados existan
RESULTS_DIR="${RESULTS_DIR:-data/results}"
REQUIRED=("metrics_xgb.json" "predictions_sample.parquet" "metrics_rf.json" "feature_importance.parquet" "elbow.json" "cluster_profiles.parquet")
MISSING=()
for f in "${REQUIRED[@]}"; do
    [[ ! -f "$RESULTS_DIR/$f" ]] && MISSING+=("$f")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ADVERTENCIA: Faltan archivos de resultados en $RESULTS_DIR/:"
    for f in "${MISSING[@]}"; do echo "  - $f"; done
    echo ""
    echo "Genera los resultados primero:"
    echo "  python ml_score_prediction.py"
    echo "  python ml_profiling.py"
    echo ""
    read -p "¿Iniciar el dashboard de todos modos? [s/N] " confirm
    [[ "$confirm" != "s" && "$confirm" != "S" ]] && exit 1
fi

LOG_FILE="$SCRIPT_DIR/dashboard/streamlit.log"
mkdir -p "$SCRIPT_DIR/dashboard"

# Matar instancia previa si existe
OLD_PID=$(lsof -ti tcp:8501 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "Deteniendo instancia previa (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "✅ Dashboard iniciado"
echo "   URL:  http://$(hostname -f 2>/dev/null || hostname):8501"
echo "   PID:  $NEW_PID"
echo "   Logs: $LOG_FILE"
