"""
Entrenamiento XGBoost Spark (SparkXGBRegressor) para puntajes ICFES.

Lee ICFES completo desde HDFS, une indicadores municipales de internet/cobertura,
entrena un regresor por cada columna en ICFES_SCORE_COLS y persiste PipelineModel
en HDFS.

Variables de entorno: mismas que data.py (SPARK_MASTER_URL, SPARK_DRIVER_MEMORY, …).

Uso:
  python ml_train.py --sample 100000
  python ml_train.py --target punt_global
  python ml_train.py
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from xgboost.spark import SparkXGBRegressor

from data import HDFS_URI, PARQUET_DIR
from transform_clean import (
    _align_spark_home_with_pyspark_package,
    _quiet_log4j_options,
    _spark_local_driver_memory,
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

PARQUET_CLEAN_DIR = f"{HDFS_URI}/data/parquet_clean"
MODELS_DIR = f"{HDFS_URI}/data/models"

# Categóricas alineadas con la hipótesis del proyecto (hogar, estudiante, colegio, territorio).
CATEGORICAL_FEATURES = [
    "fami_tieneinternet",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "estu_genero",
    "cole_naturaleza",
    "cole_jornada",
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
    "departamento",
]

NUMERIC_FEATURES = [
    "year_icfes",
    "accesos_por_habitante",
    "total_accesos_internet",
    "cobertura_neta",
]

RANDOM_SEED = 42
TRAIN_WEIGHT = 0.70
VAL_WEIGHT = 0.15
TEST_WEIGHT = 0.15

# Columnas ICFES para ML (evita codegen Janino >64KB con las 51 columnas).
ICFES_ML_COLS = (
    ICFES_SCORE_COLS
    + CATEGORICAL_FEATURES
    + ["cole_cod_mcpio_ubicacion", "periodo", "estu_estadoinvestigacion"]
)


def _prefer_java11_for_spark() -> None:
    """XGBoost Spark + PyArrow suele fallar en Java 21 (Unsafe); forzar Java 11 si existe."""
    if os.environ.get("ML_USE_SYSTEM_JAVA", "").strip().lower() in ("1", "true", "yes"):
        return
    for candidate in (
        "/usr/lib/jvm/java-11-openjdk",
        "/usr/lib/jvm/jre-11-openjdk",
    ):
        if os.path.isdir(candidate):
            prev = os.environ.get("JAVA_HOME", "")
            os.environ["JAVA_HOME"] = candidate
            if prev and prev != candidate:
                print(
                    f"JAVA_HOME cambiado {prev} -> {candidate} (requerido para XGBoost Spark)",
                    flush=True,
                )
            else:
                print(f"JAVA_HOME={candidate}", flush=True)
            return


def build_ml_spark_session() -> SparkSession:
    """Spark para ML: wholeStage off + JVM opts para PyArrow (XGBoost Spark)."""
    _prefer_java11_for_spark()
    _ensure_arrow_java_opts()
    master_url = os.environ.get("SPARK_MASTER_URL", "").strip()
    quiet_java = _quiet_log4j_options() or ""
    arrow_java = os.environ.get("SPARK_DRIVER_EXTRA_JAVA_OPTIONS", "")
    driver_java = f"{quiet_java} {arrow_java}".strip()
    exec_java = os.environ.get("SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS", arrow_java).strip()

    builder = (
        SparkSession.builder.appName("ML_XGBoost_ICFES_Internet")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.logConf", "false")
        .config("spark.sql.debug.maxToStringFields", "8")
        .config("spark.sql.codegen.wholeStage", "false")
    )
    if driver_java:
        builder = builder.config("spark.driver.extraJavaOptions", driver_java)
    if exec_java:
        builder = builder.config("spark.executor.extraJavaOptions", exec_java)

    if master_url:
        instances = int(os.environ.get("SPARK_EXECUTOR_INSTANCES", "3"))
        cores = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))
        exec_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "8g")
        driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
        default_par = os.environ.get(
            "SPARK_DEFAULT_PARALLELISM", str(max(1, instances * cores))
        )
        print(
            f"\nModo clúster: master={master_url} | "
            f"executors={instances}×{cores} cores, {exec_mem} cada uno | "
            f"driver={driver_mem}"
        )
        builder = (
            builder.master(master_url)
            .config("spark.executor.instances", str(instances))
            .config("spark.executor.cores", str(cores))
            .config("spark.executor.memory", exec_mem)
            .config("spark.driver.memory", driver_mem)
            .config("spark.default.parallelism", default_par)
        )
    else:
        driver_mem = _spark_local_driver_memory()
        print(f"\nModo local: master=local[*], driver.memory={driver_mem}")
        builder = builder.master("local[*]").config("spark.driver.memory", driver_mem)

    _align_spark_home_with_pyspark_package()
    return builder.getOrCreate()


def _ensure_arrow_java_opts() -> None:
    """PyArrow en Spark + Java 17+ requiere --add-opens para mapInPandas (XGBoost Spark)."""
    opens = (
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED"
    )
    for key in ("SPARK_DRIVER_EXTRA_JAVA_OPTIONS", "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS"):
        cur = os.environ.get(key, "")
        if "java.nio=ALL-UNNAMED" not in cur:
            os.environ[key] = f"{cur} {opens}".strip()


def _num_xgb_workers() -> int:
    """Workers XGBoost: 2 en clúster; 1 en local[*] (evita barrier con pocos cores)."""
    if os.environ.get("SPARK_MASTER_URL", "").strip():
        return int(os.environ.get("XGB_NUM_WORKERS", "2"))
    return int(os.environ.get("XGB_NUM_WORKERS", "1"))


def build_ml_dataset(
    spark: SparkSession,
    *,
    parquet_dir: str = PARQUET_DIR,
    parquet_clean_dir: str = PARQUET_CLEAN_DIR,
    sample_n: int | None = None,
) -> DataFrame:
    """ICFES (parquet completo) + join municipio internet/cobertura."""
    icfes_path = os.path.join(parquet_dir, "icfes")
    muni_path = os.path.join(parquet_clean_dir, "municipio_internet_cobertura")

    raw_icfes = spark.read.parquet(icfes_path)
    icfes_cols = [c for c in ICFES_ML_COLS if c in raw_icfes.columns]
    df = raw_icfes.select(*icfes_cols)
    if sample_n is not None:
        df = df.orderBy(F.rand(RANDOM_SEED)).limit(sample_n)

    df = _cast_existing_numeric(df, ICFES_SCORE_COLS)

    if "cole_cod_mcpio_ubicacion" in df.columns:
        df = df.withColumn("cod_municipio_norm", _norm_muni("cole_cod_mcpio_ubicacion"))
    else:
        df = df.withColumn("cod_municipio_norm", F.lit(None).cast(StringType()))

    if "periodo" in df.columns:
        df = df.withColumn("year_icfes", _icfes_period_year_expr())
    else:
        df = df.withColumn("year_icfes", F.lit(None).cast(IntegerType()))

    if "estu_estadoinvestigacion" in df.columns:
        st = F.upper(F.trim(F.col("estu_estadoinvestigacion").cast(StringType())))
        df = df.filter(st == "PUBLICAR")

    df = df.filter(F.col("cod_municipio_norm").isNotNull())

    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    muni_exists = jvm.org.apache.hadoop.fs.FileSystem.get(
        jvm.java.net.URI(muni_path), hconf
    ).exists(jvm.org.apache.hadoop.fs.Path(muni_path))

    if muni_exists:
        muni = spark.read.parquet(muni_path).select(
            "cod_municipio_norm",
            F.col("year_int").alias("year_icfes"),
            "total_accesos_internet",
            "cobertura_neta",
            "accesos_por_habitante",
            "departamento",
        )
    else:
        print(
            f"Aviso: no existe {muni_path}; construyendo join municipio desde parquet crudo.",
            flush=True,
        )
        from transform_clean import BACH_REQUIRED_COLS, INTERNET_REQUIRED_COLS

        raw_inet = spark.read.parquet(os.path.join(parquet_dir, "internet"))
        raw_bach = spark.read.parquet(os.path.join(parquet_dir, "bachillerato"))
        inet_cols = [c for c in INTERNET_REQUIRED_COLS if c in raw_inet.columns]
        bach_cols = [c for c in BACH_REQUIRED_COLS if c in raw_bach.columns]
        inet = prepare_internet(raw_inet.select(*inet_cols))
        bach = prepare_bachillerato(raw_bach.select(*bach_cols))
        muni = build_municipio_internet_cobertura(inet, bach).select(
            "cod_municipio_norm",
            F.col("year_int").alias("year_icfes"),
            "total_accesos_internet",
            "cobertura_neta",
            "accesos_por_habitante",
            "departamento",
        )

    df = df.join(muni, on=["cod_municipio_norm", "year_icfes"], how="left")
    df = df.withColumn(
        "has_muni_features",
        F.col("accesos_por_habitante").isNotNull(),
    )

    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df = df.withColumn(c, F.trim(F.col(c).cast(StringType())))

    return df.cache()


def _feature_columns_for_target(target: str, available: set[str]) -> tuple[list[str], list[str]]:
    """Devuelve (categóricas, numéricas) presentes, sin otras columnas punt_*."""
    forbidden = {c for c in ICFES_SCORE_COLS if c != target}
    cat = [c for c in CATEGORICAL_FEATURES if c in available and c not in forbidden]
    num = [c for c in NUMERIC_FEATURES if c in available and c not in forbidden]
    return cat, num


def _prepare_labeled_df(df: DataFrame, target: str) -> DataFrame:
    """Filtra filas con label válido en [0, 500]."""
    return df.filter(
        F.col(target).isNotNull()
        & (F.col(target) >= 0)
        & (F.col(target) <= 500)
    ).withColumn("label", F.col(target).cast(DoubleType()))


def _split_train_val_test(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Partición 70/15/15 en una sola pasada (sin union) para compatibilidad
    con el modo barrier de SparkXGBRegressor.
    """
    u = F.rand(RANDOM_SEED)
    split_expr = (
        F.when(u < TRAIN_WEIGHT, F.lit("train"))
        .when(u < TRAIN_WEIGHT + VAL_WEIGHT, F.lit("val"))
        .otherwise(F.lit("test"))
    )
    tagged = df.withColumn("_split", split_expr)
    train_val = (
        tagged.filter(F.col("_split").isin("train", "val"))
        .withColumn("is_val", F.col("_split") == "val")
        .drop("_split")
    )
    test = tagged.filter(F.col("_split") == "test").drop("_split")
    return train_val, test


def _build_preprocess_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep",
        )
        for c in cat_cols
    ]
    assembled_inputs = [f"{c}_idx" for c in cat_cols] + num_cols
    assembler = VectorAssembler(
        inputCols=assembled_inputs,
        outputCol="features",
        handleInvalid="keep",
    )
    return Pipeline(stages=indexers + [assembler])


def _build_xgb_regressor(num_workers: int) -> SparkXGBRegressor:
    return SparkXGBRegressor(
        features_col="features",
        label_col="label",
        validation_indicator_col="is_val",
        num_workers=num_workers,
        max_depth=6,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200,
        early_stopping_rounds=20,
        seed=RANDOM_SEED,
    )


def evaluate_regression(predictions: DataFrame) -> dict[str, float]:
    """RMSE, MAE y R² sobre el conjunto pasado (típicamente test)."""
    metrics = {}
    for name in ("rmse", "mae", "r2"):
        ev = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName=name,
        )
        metrics[name] = ev.evaluate(predictions)
    return metrics


def train_one_target(
    spark: SparkSession,
    df: DataFrame,
    target: str,
    *,
    models_dir: str = MODELS_DIR,
    num_workers: int | None = None,
    write_model: bool = True,
) -> dict[str, Any]:
    """Entrena y evalúa un regresor para un puntaje."""
    if num_workers is None:
        num_workers = _num_xgb_workers()

    available = set(df.columns)
    cat_cols, num_cols = _feature_columns_for_target(target, available)
    if not cat_cols and not num_cols:
        raise ValueError(f"Sin features disponibles para target={target}")

    labeled = _prepare_labeled_df(df, target)
    train_val, test = _split_train_val_test(labeled)

  # Preprocesamiento y XGBoost por separado: el Pipeline único con union/barrier falla en Spark 3.5.
    preprocess = _build_preprocess_pipeline(cat_cols, num_cols)
    preprocess_model = preprocess.fit(train_val)
    train_xgb = preprocess_model.transform(train_val)
    test_xgb = preprocess_model.transform(test)

    xgb = _build_xgb_regressor(num_workers)
    xgb_model = xgb.fit(train_xgb)

    pred_test = xgb_model.transform(test_xgb)
    metrics = evaluate_regression(pred_test)
    n_test = test.count()

    out_path = os.path.join(models_dir, f"xgb_{target}")
    if write_model:
        preprocess_path = os.path.join(out_path, "preprocess")
        xgb_path = os.path.join(out_path, "xgb")
        preprocess_model.write().overwrite().save(preprocess_path)
        xgb_model.write().overwrite().save(xgb_path)

    return {
        "target": target,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_test": n_test,
        "n_train_val": train_val.count(),
        "model_path": out_path,
        "cat_features": cat_cols,
        "num_features": num_cols,
        "preprocess_model": preprocess_model,
        "xgb_model": xgb_model,
        "test_predictions": pred_test,
    }


def train_all_targets(
    spark: SparkSession,
    df: DataFrame,
    *,
    targets: list[str] | None = None,
    models_dir: str = MODELS_DIR,
    sample_n: int | None = None,
) -> list[dict[str, Any]]:
    if targets is None:
        targets = list(ICFES_SCORE_COLS)

    if sample_n is not None and "cached" not in str(df.storageLevel):
        pass  # caller should pass already-built df

    results = []
    for target in targets:
        print(f"\n--- Entrenando target: {target} ---")
        res = train_one_target(spark, df, target, models_dir=models_dir)
        print(
            f"  test RMSE={res['rmse']:.3f}  MAE={res['mae']:.3f}  R²={res['r2']:.4f}  "
            f"n_test={res['n_test']:,}"
        )
        results.append(res)
    return results


def load_preprocess_model(target: str, models_dir: str = MODELS_DIR) -> PipelineModel:
    return PipelineModel.load(os.path.join(models_dir, f"xgb_{target}", "preprocess"))


def load_xgb_model(target: str, models_dir: str = MODELS_DIR):
    from xgboost.spark import SparkXGBRegressorModel

    return SparkXGBRegressorModel.load(os.path.join(models_dir, f"xgb_{target}", "xgb"))


def predict_scores(
    spark: SparkSession,
    raw_row: dict[str, Any],
    *,
    targets: list[str] | None = None,
    models_dir: str = MODELS_DIR,
) -> dict[str, float]:
    """
    Predice puntajes para una fila nueva (dict con columnas crudas del dataset ICFES
    + opcionalmente campos municipales si ya se conocen).
    """
    if targets is None:
        targets = list(ICFES_SCORE_COLS)

    preds: dict[str, float] = {}
    for target in targets:
        preprocess_model = load_preprocess_model(target, models_dir=models_dir)
        xgb_model = load_xgb_model(target, models_dir=models_dir)
        row_df = spark.createDataFrame([raw_row])
        # Aplicar mismas transformaciones de join si faltan columnas municipales
        if "cod_municipio_norm" not in row_df.columns and "cole_cod_mcpio_ubicacion" in row_df.columns:
            row_df = row_df.withColumn(
                "cod_municipio_norm",
                _norm_muni("cole_cod_mcpio_ubicacion"),
            )
        if "year_icfes" not in row_df.columns and "periodo" in row_df.columns:
            row_df = row_df.withColumn("year_icfes", _icfes_period_year_expr())

        feat_df = preprocess_model.transform(row_df)
        out = xgb_model.transform(feat_df)
        val = out.select("prediction").first()[0]
        preds[target] = float(max(0.0, min(500.0, val)))
    return preds


def print_metrics_table(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print(f"{'target':<28} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'n_test':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['target']:<28} {r['rmse']:10.3f} {r['mae']:10.3f} "
            f"{r['r2']:10.4f} {r['n_test']:10,}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena XGBoost Spark sobre ICFES.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limitar filas ICFES (desarrollo rápido).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Un solo puntaje (ej. punt_global). Por defecto los 6.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=MODELS_DIR,
        help="Directorio HDFS para PipelineModel.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="No persistir modelos en HDFS.",
    )
    args = parser.parse_args()

    spark = build_ml_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("Construyendo dataset ML…")
    df = build_ml_dataset(spark, sample_n=args.sample)
    n = df.count()
    muni_pct = df.filter(F.col("has_muni_features")).count() / max(n, 1) * 100
    print(f"  Filas: {n:,}  |  Con features municipales: {muni_pct:.1f}%")

    targets = [args.target] if args.target else list(ICFES_SCORE_COLS)
    if args.target and args.target not in ICFES_SCORE_COLS:
        raise SystemExit(f"--target debe ser uno de: {ICFES_SCORE_COLS}")

    results = []
    for target in targets:
        res = train_one_target(
            spark,
            df,
            target,
            models_dir=args.models_dir,
            write_model=not args.no_write,
        )
        results.append(res)

    print_metrics_table(results)
    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
