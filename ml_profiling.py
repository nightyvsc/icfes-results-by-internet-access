from __future__ import annotations

import argparse
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
from ml_train import (
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
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_tieneautomovil",
    "fami_cuartoshogar",
    "fami_personashogar",
]

# Numeric features (from municipality context)
PROFILING_NUM_FEATURES = [
    "accesos_por_habitante", # Del municipio
    "cobertura_neta", # Del municipio
]

ALL_REQUIRED_COLS = (
    PROFILING_CAT_FEATURES
    + PROFILING_NUM_FEATURES
    + ICFES_SCORE_COLS
    + ["cole_cod_mcpio_ubicacion", "periodo"]
)


def build_profiling_dataset(
    spark: SparkSession,
    sample_n: int | None = None,
    parquet_dir: str = PARQUET_DIR,
) -> DataFrame:
    """
    Construye el dataset especifico para profiling sociodemografico.
    """
    icfes_path = os.path.join(parquet_dir, "icfes")
    raw_icfes = spark.read.parquet(icfes_path)
    
    # Seleccionamos las variables si existen
    cols_to_select = [c for c in ALL_REQUIRED_COLS if c in raw_icfes.columns]
    df = raw_icfes.select(*cols_to_select)
    
    if sample_n is not None:
        df = df.orderBy(F.rand(42)).limit(sample_n)

    df = _cast_existing_numeric(df, ICFES_SCORE_COLS)

    if "cole_cod_mcpio_ubicacion" in df.columns:
        df = df.withColumn("cod_municipio_norm", _norm_muni("cole_cod_mcpio_ubicacion"))
    else:
        df = df.withColumn("cod_municipio_norm", F.lit(None).cast(StringType()))

    if "periodo" in df.columns:
        df = df.withColumn("year_icfes", _icfes_period_year_expr())
    else:
        df = df.withColumn("year_icfes", F.lit(None).cast(DoubleType()))

    df = df.filter(F.col("cod_municipio_norm").isNotNull())

    # Cargar datos municipales
    raw_inet = spark.read.parquet(os.path.join(parquet_dir, "internet"))
    raw_bach = spark.read.parquet(os.path.join(parquet_dir, "bachillerato"))
    inet = prepare_internet(raw_inet)
    bach = prepare_bachillerato(raw_bach)
    muni = build_municipio_internet_cobertura(inet, bach).select(
        "cod_municipio_norm",
        F.col("year_int").alias("year_icfes"),
        "total_accesos_internet",
        "cobertura_neta",
        "accesos_por_habitante",
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
    print(f"RMSE (Root Mean Squared Error): {eval_rmse.evaluate(predictions):.3f}")
    print(f"MAE (Mean Absolute Error):      {eval_mae.evaluate(predictions):.3f}")
    print(f"R² (Coeficiente Determ.):       {eval_r2.evaluate(predictions):.3f}")
    
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
    return df_importances


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
    train_random_forest_importance(df, PROFILING_CAT_FEATURES, PROFILING_NUM_FEATURES, target_col="punt_global")
    
    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
