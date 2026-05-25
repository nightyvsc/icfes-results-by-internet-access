"""
Dashboard — ICFES × Acceso a Internet
Muestra los resultados de los modelos XGBoost y de perfilamiento (Random Forest + K-Means).

Ejecutar:
    streamlit run dashboard/app.py

Los archivos de resultados deben estar en data/results/ (o la ruta indicada
por la variable de entorno RESULTS_DIR), generados por:
    python ml_score_prediction.py   →  metrics_xgb.json, predictions_sample.parquet
    python ml_profiling.py          →  metrics_rf.json, feature_importance.parquet,
                                       elbow.json, cluster_profiles.parquet
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RESULTS_DIR = os.environ.get("RESULTS_DIR", "data/results")

SCORE_LABELS = {
    "punt_global":              "Puntaje Global",
    "punt_matematicas":         "Matemáticas",
    "punt_ingles":              "Inglés",
    "punt_lectura_critica":     "Lectura Crítica",
    "punt_sociales_ciudadanas": "Sociales y Ciudadanas",
    "punt_c_naturales":         "Ciencias Naturales",
}

st.set_page_config(
    page_title="ICFES × Internet — Resultados",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def _check_results() -> list[str]:
    required = [
        "metrics_xgb.json",
        "predictions_sample.parquet",
        "metrics_rf.json",
        "feature_importance.parquet",
        "elbow.json",
        "cluster_profiles.parquet",
    ]
    return [f for f in required if not os.path.exists(_path(f))]


@st.cache_data(show_spinner=False)
def load_metrics_xgb() -> dict:
    with open(_path("metrics_xgb.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics_rf() -> dict:
    with open(_path("metrics_rf.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(_path("predictions_sample.parquet"))


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return pd.read_parquet(_path("feature_importance.parquet"))


@st.cache_data(show_spinner=False)
def load_elbow() -> dict:
    with open(_path("elbow.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    return pd.read_parquet(_path("cluster_profiles.parquet"))


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("📊 ICFES × Acceso a Internet")
st.caption(
    "Resultados de los modelos entrenados sobre datos del ICFES Saber 11, "
    "conectividad municipal (MinTIC) y cobertura educativa (MEN)."
)

# Verificar archivos antes de renderizar
missing = _check_results()
if missing:
    st.error(
        f"**Archivos faltantes en `{RESULTS_DIR}/`:** {', '.join(missing)}\n\n"
        "Genera los resultados ejecutando:\n"
        "```\n"
        "python ml_score_prediction.py\n"
        "python ml_profiling.py\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_xgb, tab_profiling = st.tabs(
    ["🤖 XGBoost — Predicción de Puntajes", "🔍 Perfilamiento Socioeconómico"]
)

# ===========================================================================
# TAB 1 — XGBoost
# ===========================================================================

with tab_xgb:
    metrics_xgb = load_metrics_xgb()

    # --- Tabla de métricas ---
    st.subheader("Métricas por puntaje (conjunto de prueba, 15 %)")

    rows = []
    for target, vals in metrics_xgb.items():
        rows.append(
            {
                "Puntaje": SCORE_LABELS.get(target, target),
                "RMSE": vals["rmse"],
                "MAE": vals["mae"],
                "R²": vals["r2"],
                "N prueba": f"{vals['n_test']:,}",
                "N entrenamiento": f"{vals['n_train_val']:,}",
            }
        )
    df_metrics = pd.DataFrame(rows).set_index("Puntaje")
    st.dataframe(
        df_metrics.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )

    st.divider()

    # --- Gráfico de barras RMSE y R² ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE por puntaje")
        df_bar = pd.DataFrame(
            [
                {"Puntaje": SCORE_LABELS.get(t, t), "RMSE": v["rmse"]}
                for t, v in metrics_xgb.items()
            ]
        ).sort_values("RMSE")
        fig_rmse = px.bar(
            df_bar,
            x="RMSE",
            y="Puntaje",
            orientation="h",
            color="RMSE",
            color_continuous_scale="Blues",
            text_auto=".2f",
        )
        fig_rmse.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        st.subheader("R² por puntaje")
        df_r2 = pd.DataFrame(
            [
                {"Puntaje": SCORE_LABELS.get(t, t), "R²": v["r2"]}
                for t, v in metrics_xgb.items()
            ]
        ).sort_values("R²", ascending=False)
        fig_r2 = px.bar(
            df_r2,
            x="R²",
            y="Puntaje",
            orientation="h",
            color="R²",
            color_continuous_scale="Greens",
            text_auto=".4f",
        )
        fig_r2.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    # --- Predicción vs Real (punt_global) ---
    st.subheader("Predicción vs Real — Puntaje Global (muestra de prueba)")
    preds = load_predictions()

    fig_scatter = px.scatter(
        preds,
        x="label",
        y="prediction",
        opacity=0.35,
        labels={"label": "Puntaje real", "prediction": "Puntaje predicho"},
        color_discrete_sequence=["#1f77b4"],
    )
    max_val = max(preds["label"].max(), preds["prediction"].max())
    min_val = min(preds["label"].min(), preds["prediction"].min())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="Predicción perfecta",
        )
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        f"Muestra de {len(preds):,} estudiantes del conjunto de prueba. "
        "La línea roja representa predicción perfecta."
    )

# ===========================================================================
# TAB 2 — Perfilamiento Socioeconómico
# ===========================================================================

with tab_profiling:

    # --- Métricas RF ---
    st.subheader("Random Forest — Métricas (objetivo: Puntaje Global)")
    rf_metrics = load_metrics_rf()

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rf_metrics['rmse']:.3f}")
    c2.metric("MAE", f"{rf_metrics['mae']:.3f}")
    c3.metric("R²", f"{rf_metrics['r2']:.4f}")

    st.divider()

    # --- Importancia de variables ---
    st.subheader("Importancia de variables (Random Forest)")
    fi = load_feature_importance()
    fi_sorted = fi.sort_values("Importance", ascending=True).tail(15)

    fig_fi = px.bar(
        fi_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Oranges",
        text_auto=".4f",
        labels={"Feature": "Variable", "Importance": "Importancia"},
    )
    fig_fi.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "Top 15 variables por importancia. "
        "Valores más altos indican mayor peso predictivo sobre el puntaje global."
    )

    st.divider()

    # --- Curva del codo (K-Means) ---
    st.subheader("K-Means — Método del codo (Silhouette Score)")
    elbow = load_elbow()
    df_elbow = pd.DataFrame({"K": elbow["k"], "Silhouette": elbow["silhouette"]})
    best_k = int(df_elbow.loc[df_elbow["Silhouette"].idxmax(), "K"])

    fig_elbow = px.line(
        df_elbow,
        x="K",
        y="Silhouette",
        markers=True,
        labels={"K": "Número de clusters (K)", "Silhouette": "Silhouette Score"},
    )
    fig_elbow.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="red",
        annotation_text=f"K óptimo = {best_k}",
        annotation_position="top right",
    )
    fig_elbow.update_layout(margin=dict(l=0, r=0))
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.divider()

    # --- Perfiles de cluster ---
    st.subheader(f"Perfiles socioeconómicos — K-Means (K = {best_k})")
    clusters = load_cluster_profiles()

    rename_map = {
        "prediction":             "Perfil",
        "n_estudiantes":          "N Estudiantes",
        "puntaje_promedio":       "Puntaje Promedio",
        "tasa_internet_hogar":    "Tasa Internet Hogar",
        "accesos_per_cap":        "Accesos per cápita",
        "estrato_frecuente":      "Estrato frecuente",
        "educ_madre_frecuente":   "Educ. madre frecuente",
        "conectividad_municipio": "Conectividad municipio",
    }
    clusters_display = clusters.rename(
        columns={k: v for k, v in rename_map.items() if k in clusters.columns}
    )
    if "Perfil" in clusters_display.columns:
        clusters_display["Perfil"] = clusters_display["Perfil"].apply(
            lambda x: f"Perfil {int(x) + 1}"
        )

    fmt = {}
    if "Puntaje Promedio" in clusters_display.columns:
        fmt["Puntaje Promedio"] = "{:.1f}"
    if "Tasa Internet Hogar" in clusters_display.columns:
        fmt["Tasa Internet Hogar"] = "{:.2f}"
    if "Accesos per cápita" in clusters_display.columns:
        fmt["Accesos per cápita"] = "{:.4f}"
    if "N Estudiantes" in clusters_display.columns:
        fmt["N Estudiantes"] = "{:,}"

    st.dataframe(
        clusters_display.style.format(fmt, na_rep="—"),
        use_container_width=True,
    )
    st.caption(
        "Cada fila representa un perfil de estudiante identificado por el algoritmo K-Means. "
        "Los perfiles se ordenan de mayor a menor puntaje promedio."
    )
