from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from data import PARQUET_DIR
from ml_score_prediction import (
    PARQUET_CLEAN_DIR,
    RESULTS_DIR,
    build_ml_spark_session,
)
from transform_clean import (
    ICFES_SCORE_COLS,
    _cast_existing_numeric,
    _icfes_period_year_expr,
    _norm_muni,
    build_municipio_internet_cobertura,
    prepare_bachillerato,
    prepare_internet,
)

# Variables de interes socioeconomico y rendimiento
PROFILING_CAT_FEATURES = [
    "nivel_conectividad_municipio",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_tienecomputador",
    "fami_tieneautomovil",
    "fami_cuartoshogar",
    "fami_personashogar",
    "estu_genero",
    "cole_naturaleza",
    "cole_jornada",
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
]

# Numeric features (from municipality context)
PROFILING_NUM_FEATURES = [
    "flag_internet_hogar",      
    "year_icfes",
    "accesos_por_habitante",    
    "total_accesos_internet",   
    "cobertura_neta", 
]

ALL_REQUIRED_COLS = (
    PROFILING_CAT_FEATURES
    + PROFILING_NUM_FEATURES
    + ICFES_SCORE_COLS
    + ["cod_municipio_norm", "year_icfes"]
)


def build_profiling_dataset(
    spark: SparkSession,
    sample_n: int | None = None,
    parquet_clean_dir: str = PARQUET_CLEAN_DIR,
) -> DataFrame:
    """
    Construye el dataset especifico para profiling cargando datos limpios.
    """
    icfes_path = os.path.join(parquet_clean_dir, "icfes")
    muni_path = os.path.join(parquet_clean_dir, "municipio_internet_cobertura")
    
    print(f"Cargando dataset ICFES limpio desde: {icfes_path}")
    raw_icfes = spark.read.parquet(icfes_path)
    print("Ruta real leída:", icfes_path)          # <- aquí
    print("Columnas en raw_icfes:", raw_icfes.columns) 
    cols_to_select = [c for c in ALL_REQUIRED_COLS if c in raw_icfes.columns]
    
    # Aseguramos incluir las llaves para el join
    for key in ["cod_municipio_norm", "year_icfes"]:
        if key not in cols_to_select and key in raw_icfes.columns:
            cols_to_select.append(key)



    df = raw_icfes.select(*cols_to_select)
    print("Columnas seleccionadas del ICFES:", cols_to_select)

    if sample_n is not None:
        df = df.orderBy(F.rand(42)).limit(sample_n)
        
    print(f"Cargando dataset municipal limpio desde: {muni_path}")
    muni = spark.read.parquet(muni_path).select(
        "cod_municipio_norm",
        F.col("year_int").alias("year_icfes"),
        "total_accesos_internet",
        "cobertura_neta",
        "accesos_por_habitante",
        "nivel_conectividad_municipio",
        "departamento",
    )
    
    df = df.join(muni, on=["cod_municipio_norm", "year_icfes"], how="left")
    
    # Drop rows without global score for profiling
    if "punt_global" in df.columns:
        df = df.filter(F.col("punt_global").isNotNull() & (F.col("punt_global") >= 0))
        
    return df.cache()


def build_preprocessing_pipeline(cat_cols: list[str], num_cols: list[str], include_target_in_features: bool = True) -> Pipeline:
    """
    Pipeline: StringIndexer -> OneHotEncoder para categoricas, Imputer -> StandardScaler para numericas.
    """
    stages = []
    
    # 1. Categoricas
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
        for c in cat_cols
    ]
    stages.extend(indexers)
    stages.extend(encoders)
    
    # 2. Numericas
    imputers = [
        Imputer(inputCol=c, outputCol=f"{c}_imputed", strategy="median")
        for c in num_cols
    ]
    stages.extend(imputers)
    
    # 3. Ensamblar todo
    ohe_cols = [f"{c}_ohe" for c in cat_cols]
    imputed_cols = [f"{c}_imputed" for c in num_cols]
    
    assembler_inputs = ohe_cols + imputed_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip")
    stages.append(assembler)
    
    # 4. Escalar (K-Means es sensible a la escala, al igual que los puntajes respecto a cantidad de cuartos)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    stages.append(scaler)
    
    return Pipeline(stages=stages)


def find_optimal_k_elbow(
    df: DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    max_k: int = 10
) -> tuple[list[int], list[float]]:
    """
    Evalua KMeans para k=2..max_k y calcula el costo (Silhouette) o Within Set Sum of Squared Errors.
    KMeans en PySpark evalua Silhouette por defecto en ClusteringEvaluator.
    """
    # Features para el clustering incluyen puntaje global
    clustering_num_cols = num_cols + ["punt_global"]
    
    # Filtrar columnas existentes
    actual_cat = [c for c in cat_cols if c in df.columns]
    actual_num = [c for c in clustering_num_cols if c in df.columns]
    
    pipeline = build_preprocessing_pipeline(actual_cat, actual_num)
    model = pipeline.fit(df)
    dataset = model.transform(df)
    
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
    
    k_values = list(range(2, max_k + 1))
    silhouette_scores = []
    
    print("\n--- Iniciando busqueda del K optimo (Metodo de la Silueta / Codo) ---")
    for k in k_values:
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        km_model = kmeans.fit(dataset)
        predictions = km_model.transform(dataset)
        score = evaluator.evaluate(predictions)
        silhouette_scores.append(score)
        print(f"  K={k} -> Silhouette Score = {score:.4f}")
        
    return k_values, silhouette_scores


def train_random_forest_importance(
    df: DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target_col: str = "punt_global"
) -> pd.DataFrame:
    """
    Entrena un Random Forest para predecir el puntaje global usando el contexto socioeconomico y municipal.
    Retorna la importancia de las variables (Feature Importance).
    """
    actual_cat = [c for c in cat_cols if c in df.columns]
    actual_num = [c for c in num_cols if c in df.columns]
    
    print("\n--- Entrenando Random Forest para Importancia de Variables ---")
    
    # Random Forest maneja variables categoricas (indexadas) nativamente muy bien sin OneHotEncoder, 
    # pero para simplicidad del Pipeline podemos usar el mismo ensamblador o usar solo StringIndexer
    stages = []
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in actual_cat
    ]
    stages.extend(indexers)
    
    imputers = [
        Imputer(inputCol=c, outputCol=f"{c}_imputed", strategy="median")
        for c in actual_num
    ]
    stages.extend(imputers)
    
    assembler_inputs = [f"{c}_idx" for c in actual_cat] + [f"{c}_imputed" for c in actual_num]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    stages.append(assembler)
    
    rf = RandomForestRegressor(featuresCol="features", labelCol=target_col, numTrees=50, maxDepth=8, seed=42)
    stages.append(rf)
    
    pipeline = Pipeline(stages=stages)
    
    # Split para evaluar el modelo (Train/Test)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)
    
    # Calcular y mostrar metricas de evaluacion
    predictions = model.transform(test_df)
    eval_rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
    
    print("\n--- Metricas de Evaluacion (Random Forest) ---")
    rmse_val = eval_rmse.evaluate(predictions)
    mae_val  = eval_mae.evaluate(predictions)
    r2_val   = eval_r2.evaluate(predictions)
    print(f"RMSE (Root Mean Squared Error): {rmse_val:.3f}")
    print(f"MAE (Mean Absolute Error):      {mae_val:.3f}")
    print(f"R² (Coeficiente Determ.):       {r2_val:.3f}")

    rf_metrics = {
        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        "r2": round(r2_val, 4),
    }
    
    # Extraer importancias
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    
    # Mapear importancias a nombres de features
    feature_names = actual_cat + actual_num
    
    df_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print(df_importances.to_string(index=False))
    return df_importances, rf_metrics


def export_profiling_results(
    fi_df: pd.DataFrame,
    rf_metrics: dict,
    k_vals: list[int],
    silhouette_scores: list[float],
    cluster_profile_pd: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Exporta resultados del perfilamiento a archivos locales para el dashboard.
    Llama esta función al final de main(), antes de spark.stop().
    """
    os.makedirs(results_dir, exist_ok=True)

    # feature_importance.parquet
    fi_path = os.path.join(results_dir, "feature_importance.parquet")
    fi_df.to_parquet(fi_path, index=False)
    print(f"[export] Importancia de features → {fi_path}")

    # metrics_rf.json
    rf_path = os.path.join(results_dir, "metrics_rf.json")
    with open(rf_path, "w") as f:
        json.dump(rf_metrics, f, indent=2)
    print(f"[export] Métricas Random Forest → {rf_path}")

    # elbow.json — curva de codo (silhouette por K)
    elbow_path = os.path.join(results_dir, "elbow.json")
    with open(elbow_path, "w") as f:
        json.dump({"k": k_vals, "silhouette": silhouette_scores}, f, indent=2)
    print(f"[export] Curva del codo → {elbow_path}")

    # cluster_profiles.parquet
    cp_path = os.path.join(results_dir, "cluster_profiles.parquet")
    cluster_profile_pd.to_parquet(cp_path, index=False)
    print(f"[export] Perfiles de cluster → {cp_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfilamiento Socioeconomico y ML ICFES.")
    parser.add_argument("--sample", type=int, default=50000, help="Muestra para desarrollo.")
    parser.add_argument("--max-k", type=int, default=8, help="K maximo para el metodo del codo.")
    args = parser.parse_args()

    spark = build_ml_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    df = build_profiling_dataset(spark, sample_n=args.sample)
    
    # 1. Metodo del codo (Silhouette) para K-Means
    k_vals, scores = find_optimal_k_elbow(df, PROFILING_CAT_FEATURES, PROFILING_NUM_FEATURES, max_k=args.max_k)
    
    # Evaluacion del mejor modelo K-Means
    best_k = k_vals[scores.index(max(scores))]
    print(f"\n--- Evaluando el mejor modelo K-Means (K={best_k}) ---")
    actual_cat = [c for c in PROFILING_CAT_FEATURES if c in df.columns]
    actual_num = [c for c in PROFILING_NUM_FEATURES + ["punt_global"] if c in df.columns]
    km_pipeline = build_preprocessing_pipeline(actual_cat, actual_num)
    km_dataset = km_pipeline.fit(df).transform(df)
    
    final_kmeans = KMeans(featuresCol="features", k=best_k, seed=42)
    final_preds = final_kmeans.fit(km_dataset).transform(km_dataset)
    
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
    print(f"Silhouette Score (Mejor Modelo): {evaluator.evaluate(final_preds):.4f}")
    print("Distribucion de estudiantes por perfil (Cluster):")
    final_preds.groupBy("prediction").count().orderBy("prediction").show()
    
    # 2. Random Forest Feature Importance (excluye punt_global de las features, lo usa como target)
    fi_df, rf_metrics = train_random_forest_importance(df, PROFILING_CAT_FEATURES, PROFILING_NUM_FEATURES, target_col="punt_global")
    # Perfil de cada cluster
    cluster_profile = final_preds.groupBy("prediction").agg(
        F.count("*").alias("n_estudiantes"),
        F.round(F.mean("punt_global"), 1).alias("puntaje_promedio"),
        F.round(F.mean("flag_internet_hogar"), 2).alias("tasa_internet_hogar"),
        F.round(F.mean("accesos_por_habitante"), 4).alias("accesos_per_cap"),
        # Moda de variables categóricas
        F.first(
            F.col("fami_estratovivienda"), ignorenulls=True
        ).alias("estrato_frecuente"),
        F.first(
            F.col("fami_educacionmadre"), ignorenulls=True
        ).alias("educ_madre_frecuente"),
        F.first(
            F.col("nivel_conectividad_municipio"), ignorenulls=True
        ).alias("conectividad_municipio"),
    ).orderBy("puntaje_promedio", ascending=False)

    cluster_profile.show(truncate=False)
    export_profiling_results(
        fi_df=fi_df,
        rf_metrics=rf_metrics,
        k_vals=k_vals,
        silhouette_scores=scores,
        cluster_profile_pd=cluster_profile.toPandas(),
    )
    df.unpersist()
    spark.stop()
    


if __name__ == "__main__":
    main()
    
