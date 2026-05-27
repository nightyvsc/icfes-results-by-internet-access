"""
Dashboard — ICFES × Acceso a Internet
Muestra los resultados de los modelos XGBoost, perfilamiento socioeconómico
y detección de anomalías territoriales (Autoencoder).

Ejecutar:
    streamlit run dashboard/app.py

Archivos requeridos en data/results/ (o RESULTS_DIR):
    ml_score_prediction.py  →  metrics_xgb.json, predictions_sample.parquet
    ml_profiling.py         →  metrics_rf.json, feature_importance.parquet,
                               elbow.json, cluster_profiles.parquet
    encoder_pipeline.py     →  ae_summary.json, ae_training_history.json,
                               ae_anomaly_results.parquet,
                               ae_feature_contributions.parquet,
                               ae_anomaly_summary.parquet
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
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

ANOMALY_COLORS = {
    "TÍPICO":               "#2ecc71",
    "ANOMALÍA_POSITIVA":    "#3498db",
    "ANOMALÍA_NEGATIVA":    "#e74c3c",
    "ANOMALÍA_SIN_PUNTAJE": "#95a5a6",
}

st.set_page_config(
    page_title="ICFES × Internet — Resultados",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


CORE_FILES = [
    "metrics_xgb.json",
    "predictions_sample.parquet",
    "metrics_rf.json",
    "feature_importance.parquet",
    "elbow.json",
    "cluster_profiles.parquet",
]

AE_FILES = [
    "ae_summary.json",
    "ae_training_history.json",
    "ae_anomaly_results.parquet",
    "ae_feature_contributions.parquet",
    "ae_anomaly_summary.parquet",
]


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


@st.cache_data(show_spinner=False)
def load_ae_summary() -> dict:
    with open(_path("ae_summary.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_ae_history() -> dict:
    with open(_path("ae_training_history.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_ae_results() -> pd.DataFrame:
    return pd.read_parquet(_path("ae_anomaly_results.parquet"))


@st.cache_data(show_spinner=False)
def load_ae_contributions() -> pd.DataFrame:
    return pd.read_parquet(_path("ae_feature_contributions.parquet"))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ICFES × Acceso a Internet")
st.caption(
    "Resultados de los modelos entrenados sobre datos del ICFES Saber 11, "
    "conectividad municipal (MinTIC) y cobertura educativa (MEN)."
)

missing_core = [f for f in CORE_FILES if not os.path.exists(_path(f))]
if missing_core:
    st.error(
        f"**Archivos faltantes en `{RESULTS_DIR}/`:** {', '.join(missing_core)}\n\n"
        "Genera los resultados ejecutando:\n"
        "```\n"
        "python ml_score_prediction.py\n"
        "python ml_profiling.py\n"
        "```"
    )
    st.stop()

ae_available = all(os.path.exists(_path(f)) for f in AE_FILES)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_xgb, tab_profiling, tab_ae = st.tabs([
    "XGBoost — Predicción de Puntajes",
    "Perfilamiento Socioeconómico",
    "Autoencoder — Anomalías Territoriales",
])

# ===========================================================================
# TAB 1 — XGBoost
# ===========================================================================

with tab_xgb:
    metrics_xgb = load_metrics_xgb()

    st.subheader("Métricas por puntaje (conjunto de prueba, 15 %)")
    rows = []
    for target, vals in metrics_xgb.items():
        rows.append({
            "Puntaje":        SCORE_LABELS.get(target, target),
            "RMSE":           vals["rmse"],
            "MAE":            vals["mae"],
            "R²":             vals["r2"],
            "N prueba":       f"{vals['n_test']:,}",
            "N entrenamiento": f"{vals['n_train_val']:,}",
        })
    df_metrics = pd.DataFrame(rows).set_index("Puntaje")
    st.dataframe(
        df_metrics.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RMSE por puntaje")
        df_bar = pd.DataFrame([
            {"Puntaje": SCORE_LABELS.get(t, t), "RMSE": v["rmse"]}
            for t, v in metrics_xgb.items()
        ]).sort_values("RMSE", ascending=False)
        fig_rmse = px.bar(
            df_bar, x="RMSE", y="Puntaje", orientation="h",
            color="RMSE", color_continuous_scale="Blues", text_auto=".2f",
        )
        fig_rmse.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        st.subheader("R² por puntaje")
        df_r2 = pd.DataFrame([
            {"Puntaje": SCORE_LABELS.get(t, t), "R²": v["r2"]}
            for t, v in metrics_xgb.items()
        ]).sort_values("R²")
        fig_r2 = px.bar(
            df_r2, x="R²", y="Puntaje", orientation="h",
            color="R²", color_continuous_scale="Greens", text_auto=".4f",
        )
        fig_r2.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()

    st.subheader("Predicción vs Real — Puntaje Global (muestra de prueba)")
    preds = load_predictions()
    max_val = max(preds["label"].max(), preds["prediction"].max())
    min_val = min(preds["label"].min(), preds["prediction"].min())
    fig_scatter = px.density_heatmap(
        preds, x="label", y="prediction",
        nbinsx=50, nbinsy=50,
        color_continuous_scale="Blues",
        labels={"label": "Puntaje real", "prediction": "Puntaje predicho"},
    )
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color="red", dash="dash", width=1.5),
        name="Predicción perfecta",
    ))
    fig_scatter.update_layout(margin=dict(l=0, r=0), coloraxis_showscale=False)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        f"Muestra de {len(preds):,} estudiantes del conjunto de prueba. "
        "Celdas más oscuras indican mayor concentración de predicciones. "
        "La línea roja representa predicción perfecta."
    )

# ===========================================================================
# TAB 2 — Perfilamiento Socioeconómico
# ===========================================================================

with tab_profiling:
    st.subheader("Random Forest — Métricas (objetivo: Puntaje Global)")
    rf_metrics = load_metrics_rf()
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rf_metrics['rmse']:.3f}")
    c2.metric("MAE",  f"{rf_metrics['mae']:.3f}")
    c3.metric("R²",   f"{rf_metrics['r2']:.4f}")

    st.divider()

    st.subheader("Importancia de variables (Random Forest)")
    fi = load_feature_importance()
    fi_sorted = fi.sort_values("Importance", ascending=True).tail(15)
    fig_fi = px.bar(
        fi_sorted, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Oranges", text_auto=".4f",
        labels={"Feature": "Variable", "Importance": "Importancia"},
    )
    fig_fi.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "Top 15 variables por importancia. "
        "Valores más altos indican mayor peso predictivo sobre el puntaje global."
    )

    st.divider()

    st.subheader("K-Means — Método del codo (Silhouette Score)")
    elbow = load_elbow()
    df_elbow = pd.DataFrame({"K": elbow["k"], "Silhouette": elbow["silhouette"]})
    best_k = int(df_elbow.loc[df_elbow["Silhouette"].idxmax(), "K"])
    fig_elbow = px.line(
        df_elbow, x="K", y="Silhouette", markers=True,
        labels={"K": "Número de clusters (K)", "Silhouette": "Silhouette Score"},
    )
    fig_elbow.add_vline(
        x=best_k, line_dash="dash", line_color="red",
        annotation_text=f"K óptimo = {best_k}", annotation_position="top right",
    )
    fig_elbow.update_layout(margin=dict(l=0, r=0))
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.divider()

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
    if "Puntaje Promedio"    in clusters_display.columns: fmt["Puntaje Promedio"]    = "{:.1f}"
    if "Tasa Internet Hogar" in clusters_display.columns: fmt["Tasa Internet Hogar"] = "{:.2f}"
    if "Accesos per cápita"  in clusters_display.columns: fmt["Accesos per cápita"]  = "{:.4f}"
    if "N Estudiantes"       in clusters_display.columns: fmt["N Estudiantes"]       = "{:,}"
    st.dataframe(
        clusters_display.style.format(fmt, na_rep="—"),
        use_container_width=True,
    )
    st.caption(
        "Cada fila representa un perfil de estudiante identificado por el algoritmo K-Means. "
        "Los perfiles se ordenan de mayor a menor puntaje promedio."
    )

# ===========================================================================
# TAB 3 — Autoencoder: Anomalías Territoriales
# ===========================================================================

with tab_ae:
    if not ae_available:
        st.warning(
            f"Resultados del autoencoder no encontrados en `{RESULTS_DIR}/`.\n\n"
            "Genera los archivos ejecutando:\n"
            "```\npython encoder_pipeline.py\n```"
        )
        st.stop()

    ae_sum   = load_ae_summary()
    history  = load_ae_history()
    ae_res   = load_ae_results()
    contribs = load_ae_contributions()

    tr = ae_sum["training"]
    th = ae_sum["threshold"]
    ac = ae_sum["anomaly_counts"]
    n_anom = sum(v for k, v in ac.items() if k != "TÍPICO")

    # --- KPI cards ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mejor Val Loss",        f"{tr['best_val_loss']:.5f}",
              f"época {tr['best_val_epoch']}")
    k2.metric("Épocas ejecutadas",     str(tr["epochs_run"]))
    k3.metric("Umbral adaptativo",     f"{th['threshold']:.5f}",
              f"método: {th['method']}")
    k4.metric("% Municipios anómalos", f"{ae_sum['pct_anomalies']:.1f} %",
              f"{n_anom:,} de {ae_sum['n_test']:,}")

    st.divider()

    col_left, col_right = st.columns(2)

    # --- Training loss curve ---
    with col_left:
        st.subheader("Curva de entrenamiento")
        df_hist = pd.DataFrame({
            "Época":      list(range(1, len(history["train_loss"]) + 1)),
            "Train Loss": history["train_loss"],
            "Val Loss":   history["val_loss"],
        })
        fig_loss = px.line(
            df_hist.melt(id_vars="Época", var_name="Conjunto", value_name="Loss"),
            x="Época", y="Loss", color="Conjunto",
            color_discrete_map={"Train Loss": "#1f77b4", "Val Loss": "#ff7f0e"},
        )
        fig_loss.add_vline(
            x=tr["best_val_epoch"], line_dash="dash", line_color="green",
            annotation_text=f"Mejor época ({tr['best_val_epoch']})",
            annotation_position="top right",
        )
        fig_loss.update_layout(margin=dict(l=0, r=0))
        st.plotly_chart(fig_loss, use_container_width=True)

    # --- Anomaly distribution ---
    with col_right:
        st.subheader("Distribución de clasificaciones")
        df_cls = pd.DataFrame([
            {"Clase": cls, "N": count} for cls, count in ac.items()
        ]).sort_values("N", ascending=False)
        fig_cls = px.bar(
            df_cls, x="Clase", y="N",
            color="Clase", color_discrete_map=ANOMALY_COLORS,
            text_auto=True, labels={"N": "Municipios-año"},
        )
        fig_cls.update_layout(showlegend=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_cls, use_container_width=True)

    st.divider()

    # --- Reconstruction error distribution ---
    st.subheader("Distribución del error de reconstrucción")
    fig_hist = px.histogram(
        ae_res, x="reconstruction_error", color="anomaly_class",
        nbins=60, barmode="overlay", opacity=0.7,
        color_discrete_map=ANOMALY_COLORS,
        labels={
            "reconstruction_error": "Error de reconstrucción (MSE)",
            "anomaly_class": "Clase",
        },
    )
    fig_hist.add_vline(
        x=th["threshold"], line_dash="dash", line_color="black", line_width=2,
        annotation_text=f"Umbral = {th['threshold']:.4f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(margin=dict(l=0, r=0))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # --- Feature contributions ---
    st.subheader("Contribución promedio por feature — Municipios anómalos")
    anom_contribs = contribs[contribs["anomaly_class"] != "TÍPICO"]
    if not anom_contribs.empty:
        feat_avg = (
            anom_contribs.groupby("feature")["feature_error_pct"]
            .mean().reset_index()
            .sort_values("feature_error_pct", ascending=True)
        )
        fig_feat = px.bar(
            feat_avg, x="feature_error_pct", y="feature", orientation="h",
            color="feature_error_pct", color_continuous_scale="Reds", text_auto=".1f",
            labels={
                "feature_error_pct": "Contribución promedio al error (%)",
                "feature": "Feature",
            },
        )
        fig_feat.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0))
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption(
            "Porcentaje promedio con el que cada feature contribuye al error total "
            "de reconstrucción en municipios clasificados como anómalos."
        )
    else:
        st.info("No se detectaron municipios anómalos en el test set.")

    st.divider()

    # --- Top anomalies tables ---
    st.subheader("Top municipios anómalos por error de reconstrucción")
    for cls_label, label in [
        ("ANOMALÍA_NEGATIVA", "ANOMALÍA_NEGATIVA — Top 10"),
        ("ANOMALÍA_POSITIVA", "ANOMALÍA_POSITIVA — Top 10"),
    ]:
        subset = ae_res[ae_res["anomaly_class"] == cls_label]
        if subset.empty:
            continue
        top = subset.nlargest(10, "reconstruction_error").reset_index(drop=True)
        top.index += 1
        cols_show = [c for c in [
            "municipio", "departamento", "year_int",
            "reconstruction_error", "anomaly_score_pct",
            "puntaje_global_promedio",
        ] if c in top.columns]
        fmt_top = {k: v for k, v in {
            "reconstruction_error":    "{:.5f}",
            "anomaly_score_pct":       "{:.1f}",
            "puntaje_global_promedio": "{:.1f}",
        }.items() if k in cols_show}
        st.markdown(f"**{label}**")
        st.dataframe(
            top[cols_show].style.format(fmt_top, na_rep="—"),
            use_container_width=True,
        )