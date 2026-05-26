"""
encoder_pipeline.py
===================
Pipeline completo de detección de anomalías territoriales con autoencoder.

Reemplaza el notebook 06_autoencoder_anomaly_detection.ipynb con un script
ejecutable directamente, sin dependencias de Jupyter.

Fases:
  1. Carga de datos limpios desde HDFS (parquet_clean)
  2. Feature engineering: join municipio + ICFES agregado + puntaje_global_promedio
  3. Train / Val / Test split (70 / 15 / 15)  →  guarda splits en HDFS
  4. Entrenamiento distribuido con Spark TorchDistributor + PyTorch DDP
  5. Evaluación: errores de reconstrucción + umbral adaptativo (Shapiro-Wilk)
  6. Clasificación tripartita: ANOMALÍA_POSITIVA / ANOMALÍA_NEGATIVA / TÍPICO
  7. Reporte detallado: contribución por feature + outputs separados en HDFS

Uso:
  python encoder_pipeline.py [opciones]

Opciones clave:
  --epochs N        Épocas de entrenamiento (default: 50)
  --workers N       Workers de TorchDistributor (default: 2)
  --latent-dim N    Dimensión bottleneck (default: 4)
  --skip-training   Carga modelo ya entrenado desde HDFS, salta la fase 4
  --local           Fuerza modo local (spark master=local[*])

Umbral de anomalía:
  Se determina automáticamente con Shapiro-Wilk sobre los errores de
  reconstrucción del test set:
    - Distribución normal (p > 0.05):  umbral = media + 2σ
    - Distribución no normal:          umbral = percentil 75

Requisitos del entorno:
  - HDFS_URI          (default: hdfs://spark-master:9000)
  - SPARK_MASTER_URL  (default: spark://spark-master:7077)
  - SPARK_EXECUTOR_INSTANCES, SPARK_EXECUTOR_CORES, SPARK_EXECUTOR_MEMORY
  - El archivo debe estar en /mnt/icfes-results-by-internet-access-1/
    junto a ml_autoencoder.py (misma carpeta, accesible en todos los nodos
    vía NFS mount).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constantes de rutas y features
# ---------------------------------------------------------------------------

# Ruta del proyecto en todos los nodos (NFS mount compartido)
PROJECT_PATH = str(Path(__file__).resolve().parent)

# Features derivadas de transform_clean.py:
#   municipio_internet_cobertura → accesos_por_habitante, total_accesos_internet,
#                                   cobertura_neta, poblaci_n_5_16
#   municipio_agg_icfes          → prom_estrato_hogar, pct_internet_hogar,
#                                   pct_educacion_madre_superior/padre, pct_colegio_privado,
#                                   pct_area_urbana, promedio_personas_hogar, pct_hogar_computador
#   internet agregado            → velocidad_bajada (avg por municipio-año)
FEATURE_COLUMNS = [
    "accesos_por_habitante",
    "total_accesos_internet",
    "cobertura_neta",
    "poblaci_n_5_16",
    "prom_estrato_hogar",
    "pct_internet_hogar",
    "pct_educacion_madre_superior",
    "pct_educacion_padre_superior",
    "pct_colegio_privado",
    "pct_area_urbana",
    "promedio_personas_hogar",
    "pct_hogar_computador",
    "velocidad_bajada",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ae_pipeline")


def _section(title: str) -> None:
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Fase 1 + 2: Spark session y carga de datos
# ---------------------------------------------------------------------------

def build_spark(local: bool = False):
    """Crea SparkSession con configuración de clúster o local."""
    from pyspark.sql import SparkSession

    master_url = "local[*]" if local else os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    hdfs_uri   = os.environ.get("HDFS_URI", "hdfs://spark-master:9000")

    builder = (
        SparkSession.builder
        .appName("AutoencoderAnomalyDetection")
        .master(master_url)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.debug.maxToStringFields", "8")
        .config("spark.sql.codegen.wholeStage", "false")
        # Evita conflictos con torch.distributed que también usa netty
        .config("spark.port.maxRetries", "32")
    )

    if not local:
        instances   = int(os.environ.get("SPARK_EXECUTOR_INSTANCES", "3"))
        cores       = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))
        exec_mem    = os.environ.get("SPARK_EXECUTOR_MEMORY", "8g")
        driver_mem  = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
        parallelism = str(max(1, instances * cores))

        builder = (
            builder
            .config("spark.executor.instances", str(instances))
            .config("spark.executor.cores", str(cores))
            .config("spark.executor.memory", exec_mem)
            .config("spark.driver.memory", driver_mem)
            .config("spark.default.parallelism", parallelism)
            # Distribuye el módulo a los executors (para UDFs/RDDs de Spark)
            # Nota: TorchDistributor usa sys.path vía PROJECT_PATH, no addPyFile
        )
        log.info(
            f"Spark clúster: master={master_url} | "
            f"executors={instances}×{cores}c {exec_mem} | driver={driver_mem}"
        )
    else:
        log.info("Spark modo local: local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark, master_url


def load_and_prepare_data(spark, parquet_clean_dir: str):
    """
    Fases 1 + 2: carga parquet_clean, agrega ICFES a nivel municipio,
    hace join con municipio_internet_cobertura, devuelve DataFrame Spark
    con las features disponibles.

    Nota sobre computación distribuida:
    - groupBy + agg sobre icfes (7M filas) corre en paralelo en los executors
    - el join es un shuffle distribuido
    - toPandas() solo ocurre al final, sobre ~7.4k filas de municipios
    """
    from pyspark.sql import functions as F

    _section("Fase 1: Carga de datos desde HDFS")

    # --- Municipio internet + cobertura ---
    mun_path = f"{parquet_clean_dir}/municipio_internet_cobertura"
    log.info(f"Leyendo municipio_internet_cobertura: {mun_path}")
    mun_df = spark.read.parquet(mun_path)
    log.info(f"  municipio_internet_cobertura: {mun_df.count():,} filas | cols: {mun_df.columns}")

    # --- ICFES limpio ---
    icfes_path = f"{parquet_clean_dir}/icfes"
    log.info(f"Leyendo ICFES: {icfes_path}")
    icfes_df = spark.read.parquet(icfes_path)
    log.info(f"  ICFES: {icfes_df.count():,} registros")

    # --- Internet limpio (para velocidad_bajada por municipio) ---
    internet_path = f"{parquet_clean_dir}/internet"
    log.info(f"Leyendo internet: {internet_path}")
    internet_df = spark.read.parquet(internet_path)
    log.info(f"  internet: {internet_df.count():,} registros")

    _section("Fase 2: Feature engineering distribuido")

    # Agrega ICFES a nivel municipio (computación distribuida sobre 7M filas)
    log.info("Agregando ICFES por municipio [distribuido]...")
    icfes_agg = _aggregate_icfes(icfes_df)
    log.info(f"  ICFES agregado: {icfes_agg.count():,} municipios-año")

    # Agrega velocidad_bajada por municipio-año (media)
    log.info("Agregando velocidad_bajada por municipio [distribuido]...")
    vel_agg = (
        internet_df
        .filter(F.col("year_int").isNotNull() & F.col("cod_municipio_norm").isNotNull())
        .groupBy("cod_municipio_norm", "year_int")
        .agg(F.avg("velocidad_bajada").alias("velocidad_bajada"))
    )

    # Join: municipio + ICFES + velocidad
    log.info("Haciendo joins [shuffle distribuido]...")
    combined = (
        mun_df
        .join(icfes_agg,  on=["cod_municipio_norm", "year_int"], how="inner")
        .join(vel_agg,    on=["cod_municipio_norm", "year_int"], how="left")
    )
    total = combined.count()
    log.info(f"  Dataset combinado: {total:,} municipio-año")

    # Columnas disponibles (algunas features pueden no existir)
    available_cols = [c for c in FEATURE_COLUMNS if c in combined.columns]
    missing_cols   = [c for c in FEATURE_COLUMNS if c not in combined.columns]
    if missing_cols:
        log.warning(f"  Features no disponibles (ignoradas): {missing_cols}")
    log.info(f"  Features disponibles ({len(available_cols)}): {available_cols}")

    # Seleccionar solo las features + claves de identificación
    id_cols  = ["cod_municipio_norm", "year_int", "municipio", "departamento",
                "nivel_conectividad_municipio", "puntaje_global_promedio"]
    id_cols  = [c for c in id_cols if c in combined.columns]
    final_df = combined.select(id_cols + available_cols)

    # Diagnóstico de nulls por columna (una pasada distribuida con agg)
    log.info("  Diagnóstico de nulls por feature:")
    null_counts = final_df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in available_cols
    ]).collect()[0].asDict()
    cols_with_nulls = {c: n for c, n in null_counts.items() if n > 0}
    if cols_with_nulls:
        for c, n in sorted(cols_with_nulls.items(), key=lambda x: -x[1]):
            log.warning(f"    {c}: {n:,} nulls ({100*n/total:.1f}%)")
    else:
        log.info("    Sin nulls — todas las features completas.")

    # Imputar nulls por mediana por columna (más robusto que media para ML)
    if cols_with_nulls:
        log.info("  Imputando nulls por mediana...")
        fill_map = {}
        for col_name in cols_with_nulls:
            median_val = final_df.approxQuantile(col_name, [0.5], relativeError=0.01)
            if median_val and median_val[0] is not None:
                fill_map[col_name] = float(median_val[0])
                log.info(f"    {col_name}: mediana={fill_map[col_name]:.4f}")
            else:
                fill_map[col_name] = 0.0
                log.warning(f"    {col_name}: mediana no disponible, imputando con 0")
        final_df = final_df.fillna(fill_map)

    # Eliminar filas que sigan con nulls tras imputación (edge case: debería ser 0)
    final_df = final_df.dropna(subset=available_cols)
    final_count = final_df.count()
    log.info(f"  Filas finales tras imputación: {final_count:,} (eliminadas {total - final_count:,})")

    return final_df, available_cols, id_cols


def _aggregate_icfes(icfes_df):
    """
    Agrega variables socioeconómicas del ICFES a nivel municipio-año.
    Replica la lógica de build_municipio_agg_icfes de transform_clean.py.
    Se ejecuta de forma distribuida en los executors de Spark.

    Notas sobre parsing:
    - fami_estratovivienda llega como "Estrato 1".."Estrato 6" o "Sin Estrato"
      → se extrae el dígito con regexp_extract en lugar de cast directo.
    - fami_personashogar llega como "1".."9 o más" → cast seguro con regex.
    - flag_internet_hogar: la columna ya puede existir en el parquet limpio
      (creada por prepare_icfes); si no, se reconstruye.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, DoubleType

    df = icfes_df

    # --- flag_internet_hogar ---
    if "flag_internet_hogar" not in df.columns:
        if "fami_tieneinternet" in df.columns:
            inet = F.upper(F.trim(F.col("fami_tieneinternet").cast("string")))
            df = df.withColumn(
                "flag_internet_hogar",
                F.when(inet == "SI", F.lit(1))
                 .when(inet == "NO", F.lit(0))
                 .otherwise(F.lit(None).cast(IntegerType()))
            )
        else:
            df = df.withColumn("flag_internet_hogar", F.lit(None).cast(IntegerType()))

    # --- flag_edu_madre_superior ---
    if "fami_educacionmadre" in df.columns:
        df = df.withColumn(
            "flag_edu_madre_superior",
            F.when(F.col("fami_educacionmadre").like("%Superior%"), F.lit(1)).otherwise(F.lit(0))
        )
    else:
        df = df.withColumn("flag_edu_madre_superior", F.lit(0))

    # --- flag_edu_padre_superior ---
    if "fami_educacionpadre" in df.columns:
        df = df.withColumn(
            "flag_edu_padre_superior",
            F.when(F.col("fami_educacionpadre").like("%Superior%"), F.lit(1)).otherwise(F.lit(0))
        )
    else:
        df = df.withColumn("flag_edu_padre_superior", F.lit(0))

    # --- flag_privado ---
    if "cole_naturaleza" in df.columns:
        df = df.withColumn(
            "flag_privado",
            F.when(F.col("cole_naturaleza") == "NO OFICIAL", F.lit(1)).otherwise(F.lit(0))
        )
    else:
        df = df.withColumn("flag_privado", F.lit(0))

    # --- flag_urbano ---
    if "cole_area_ubicacion" in df.columns:
        df = df.withColumn(
            "flag_urbano",
            F.when(F.col("cole_area_ubicacion") == "URBANA", F.lit(1)).otherwise(F.lit(0))
        )
    else:
        df = df.withColumn("flag_urbano", F.lit(0))

    # --- flag_computador ---
    if "fami_tienecomputador" in df.columns:
        df = df.withColumn(
            "flag_computador",
            F.when(F.upper(F.trim(F.col("fami_tienecomputador").cast("string"))) == "SI", F.lit(1))
             .otherwise(F.lit(0))
        )
    else:
        df = df.withColumn("flag_computador", F.lit(0))

    # --- estrato_num ---
    # fami_estratovivienda puede ser "Estrato 1"..."Estrato 6" o "Sin Estrato" o ya int.
    # regexp_extract saca el primer dígito; si no hay dígito devuelve "" → null.
    if "fami_estratovivienda" in df.columns:
        raw = F.col("fami_estratovivienda").cast("string")
        extracted = F.regexp_extract(F.trim(raw), r"(\d+)", 1)
        df = df.withColumn(
            "estrato_num",
            F.when((extracted.isNull()) | (extracted == ""), F.lit(None).cast(DoubleType()))
             .otherwise(extracted.cast(DoubleType()))
        )
    else:
        df = df.withColumn("estrato_num", F.lit(None).cast(DoubleType()))

    # --- personas_num ---
    # fami_personashogar puede ser "1".."9 o más" (string) o ya int.
    # Extraemos el primer bloque numérico.
    if "fami_personashogar" in df.columns:
        raw_p = F.col("fami_personashogar").cast("string")
        ext_p = F.regexp_extract(F.trim(raw_p), r"(\d+)", 1)
        df = df.withColumn(
            "personas_num",
            F.when((ext_p.isNull()) | (ext_p == ""), F.lit(None).cast(DoubleType()))
             .otherwise(ext_p.cast(DoubleType()))
        )
    else:
        df = df.withColumn("personas_num", F.lit(None).cast(DoubleType()))

    # --- Normalizar cod_municipio_norm si no existe ---
    if "cod_municipio_norm" not in df.columns:
        if "cole_cod_mcpio_ubicacion" in df.columns:
            c = F.trim(F.col("cole_cod_mcpio_ubicacion").cast("string"))
            df = df.withColumn(
                "cod_municipio_norm",
                F.when((c.isNull()) | (c == ""), None).otherwise(F.lpad(c, 5, "0"))
            )
        else:
            df = df.withColumn("cod_municipio_norm", F.lit(None).cast("string"))

    # --- Normalizar year_int / year_icfes ---
    year_col = None
    if "year_int" in df.columns:
        year_col = "year_int"
    elif "year_icfes" in df.columns:
        year_col = "year_icfes"
    elif "periodo" in df.columns:
        p = F.trim(F.col("periodo").cast("string"))
        df = df.withColumn(
            "_year_icfes",
            F.when((p.isNull()) | (p == ""), None)
             .otherwise(F.substring(p, 1, 4).cast(IntegerType()))
        )
        year_col = "_year_icfes"
    else:
        df = df.withColumn("_year_icfes", F.lit(None).cast(IntegerType()))
        year_col = "_year_icfes"

    df = df.filter(F.col("cod_municipio_norm").isNotNull() & F.col(year_col).isNotNull())

    # Nombre de la columna de puntaje global puede variar según la versión del parquet
    punt_col = next(
        (c for c in df.columns if "punt_global" in c.lower() or "puntaje_global" in c.lower()),
        None,
    )

    agg_exprs = [
        F.round(F.avg("estrato_num"), 2).alias("prom_estrato_hogar"),
        (F.sum("flag_internet_hogar") / F.count("*") * 100).alias("pct_internet_hogar"),
        (F.sum("flag_edu_madre_superior") / F.count("*") * 100).alias("pct_educacion_madre_superior"),
        (F.sum("flag_edu_padre_superior") / F.count("*") * 100).alias("pct_educacion_padre_superior"),
        (F.sum("flag_privado") / F.count("*") * 100).alias("pct_colegio_privado"),
        (F.sum("flag_urbano") / F.count("*") * 100).alias("pct_area_urbana"),
        F.round(F.avg("personas_num"), 2).alias("promedio_personas_hogar"),
        (F.sum("flag_computador") / F.count("*") * 100).alias("pct_hogar_computador"),
    ]
    if punt_col:
        agg_exprs.append(F.round(F.avg(punt_col), 2).alias("puntaje_global_promedio"))

    return df.groupBy("cod_municipio_norm", F.col(year_col).alias("year_int")).agg(*agg_exprs)


# ---------------------------------------------------------------------------
# Fase 3: Split y guardado en HDFS
# ---------------------------------------------------------------------------

def split_and_save(spark, final_df, available_cols: list[str], id_cols: list[str],
                   ae_datasets_dir: str):
    """
    Fase 3: split 70/15/15 y guarda train/val/test como Parquet en HDFS.
    El split se hace en Spark (randomSplit) para mantener distribución.
    """
    _section("Fase 3: Split Train / Val / Test")

    # randomSplit es una operación distribuida en Spark
    train_df, val_df, test_df = final_df.randomSplit([0.70, 0.15, 0.15], seed=42)

    train_count = train_df.count()
    val_count   = val_df.count()
    test_count  = test_df.count()
    log.info(f"Train: {train_count:,} | Val: {val_count:,} | Test: {test_count:,}")

    train_path = f"{ae_datasets_dir}/train"
    val_path   = f"{ae_datasets_dir}/val"
    test_path  = f"{ae_datasets_dir}/test"

    # Guardar solo features en train/val (el trainer no necesita id_cols)
    log.info(f"Guardando splits en HDFS: {ae_datasets_dir}")
    train_df.select(available_cols).write.mode("overwrite").parquet(train_path)
    val_df.select(available_cols).write.mode("overwrite").parquet(val_path)
    # test incluye ids + puntaje_global_promedio para clasificación de anomalías
    test_df.select(id_cols + available_cols).write.mode("overwrite").parquet(test_path)
    log.info("  Splits guardados.")

    return train_path, val_path, test_path, test_count


# ---------------------------------------------------------------------------
# Fase 4: Entrenamiento distribuido con TorchDistributor
# ---------------------------------------------------------------------------

def run_distributed_training(
    spark,
    train_path: str,
    val_path: str,
    model_dir: str,
    available_cols: list[str],
    args,
):
    """
    Fase 4: entrenamiento distribuido con Spark TorchDistributor + PyTorch DDP.

    Diseño clave para evitar el Gloo timeout:
    - Los datos se leen desde HDFS en el driver (proceso principal, antes de
      lanzar TorchDistributor). Los splits tienen ~3k filas × 13 features = <5MB.
    - Los arrays numpy se serializan como bytes en el closure de worker_fn.
    - Dentro del worker, no se usa Spark en absoluto: los datos ya están en
      memoria. Todos los ranks llegan al barrier de DDP casi instantáneamente.

    Por qué el diseño anterior causaba el timeout:
    - Cada worker creaba una SparkSession y leía HDFS DESPUÉS de init_process_group.
    - Gloo quedaba esperando que ambos ranks llegaran a DistributedDataParallel(model),
      pero un rank podía tardar minutos en arrancar Spark. Timeout de 30 min.
    """
    from pyspark.ml.torch.distributor import TorchDistributor

    _section("Fase 4: Entrenamiento distribuido (TorchDistributor + DDP)")
    log.info(f"Workers: {args.workers} | Épocas: {args.epochs} | Latent dim: {args.latent_dim}")
    log.info(f"Input dim: {len(available_cols)} features")

    # -----------------------------------------------------------------------
    # Leer datos en el DRIVER antes de lanzar TorchDistributor.
    # Los workers recibirán los arrays ya en memoria via el closure.
    # -----------------------------------------------------------------------
    log.info("Cargando splits desde HDFS en el driver...")

    def _load_parquet_as_numpy(path: str, cols: list) -> tuple:
        df    = spark.read.parquet(path)
        avail = [c for c in cols if c in df.columns]
        arr   = df.select(avail).toPandas().values.astype(np.float32)
        return arr, avail

    train_np, avail_cols = _load_parquet_as_numpy(train_path, available_cols)
    val_np,   _          = _load_parquet_as_numpy(val_path,   available_cols)

    log.info(f"  train: {train_np.shape} | val: {val_np.shape}")

    # Normalizar en el driver (estadísticos del train)
    mean_np = train_np.mean(axis=0)
    std_np  = train_np.std(axis=0)
    std_np[std_np == 0] = 1.0
    train_np = (train_np - mean_np) / std_np
    val_np   = (val_np   - mean_np) / std_np

    # Serializar arrays como bytes para el closure (cloudpickle-safe)
    import io
    def _to_bytes(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    _train_bytes  = _to_bytes(train_np)
    _val_bytes    = _to_bytes(val_np)
    _mean_bytes   = _to_bytes(mean_np)
    _std_bytes    = _to_bytes(std_np)
    _avail_cols   = list(avail_cols)
    _input_dim    = train_np.shape[1]
    _model_dir    = model_dir
    _project_path = PROJECT_PATH
    _config = {
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "early_stopping": args.early_stopping,
        "input_dim":      _input_dim,
        "latent_dim":     args.latent_dim,
    }

    log.info(f"  Datos serializados ({len(_train_bytes)/1024:.1f} KB train, "
             f"{len(_val_bytes)/1024:.1f} KB val) — listos para los workers.")

    def worker_fn():
        """
        Worker DDP: sin Spark, sin I/O — solo PyTorch.
        Los datos ya vienen serializados en el closure; todos los ranks
        llegan al barrier de DDP en milisegundos, evitando el Gloo timeout.
        """
        import os, sys, io, json
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler
        import numpy as np

        # Insertar project path antes de imports del proyecto
        if _project_path not in sys.path:
            sys.path.insert(0, _project_path)
        from ml_autoencoder import AutoencoderMunicipal, reconstruction_error

        # --- Inicializar DDP (todos los ranks llegan aquí casi simultáneamente) ---
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank       = int(os.environ.get("RANK", "0"))
        torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        device = torch.device("cpu")

        if rank == 0:
            print(f"[DDP] world_size={world_size} | rank={rank} | input_dim={_input_dim}")

        # --- Deserializar datos desde el closure (ya normalizados) ---
        train_np = np.load(io.BytesIO(_train_bytes))
        val_np   = np.load(io.BytesIO(_val_bytes))
        mean_np  = np.load(io.BytesIO(_mean_bytes))
        std_np   = np.load(io.BytesIO(_std_bytes))

        # --- DataLoaders con DistributedSampler ---
        train_ds      = TensorDataset(torch.from_numpy(train_np))
        val_ds        = TensorDataset(torch.from_numpy(val_np))
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader  = DataLoader(train_ds, batch_size=_config["batch_size"], sampler=train_sampler)
        val_loader    = DataLoader(val_ds,   batch_size=_config["batch_size"], shuffle=False)

        # --- Modelo + DDP wrapper ---
        model = AutoencoderMunicipal(input_dim=_input_dim, latent_dim=_config["latent_dim"])
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)

        optimizer = optim.Adam(
            model.parameters(),
            lr=_config["lr"],
            weight_decay=_config["weight_decay"],
        )

        # --- Loop de entrenamiento ---
        history      = {"train_loss": [], "val_loss": []}
        best_val     = float("inf")
        patience_ctr = 0

        for epoch in range(_config["epochs"]):
            model.train()
            train_sampler.set_epoch(epoch)
            t_loss, t_n = 0.0, 0
            for (x,) in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                loss = reconstruction_error(x, model(x), reduction="mean")
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * x.size(0)
                t_n    += x.size(0)

            tl = torch.tensor([t_loss / max(1, t_n)], dtype=torch.float32)
            torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
            avg_train = tl.item() / world_size

            model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(device)
                    loss = reconstruction_error(x, model(x), reduction="mean")
                    v_loss += loss.item() * x.size(0)
                    v_n    += x.size(0)

            vl = torch.tensor([v_loss / max(1, v_n)], dtype=torch.float32)
            torch.distributed.all_reduce(vl, op=torch.distributed.ReduceOp.SUM)
            avg_val = vl.item() / world_size

            history["train_loss"].append(float(avg_train))
            history["val_loss"].append(float(avg_val))

            if rank == 0 and (epoch % 10 == 0 or epoch == _config["epochs"] - 1):
                print(f"  Epoch {epoch+1:>3}/{_config['epochs']} | "
                      f"train={avg_train:.6f} | val={avg_val:.6f}")

            if avg_val < best_val:
                best_val     = avg_val
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= _config["early_stopping"]:
                    if rank == 0:
                        print(f"  Early stopping en epoch {epoch+1}")
                    break

        # --- Guardar artefactos (solo rank 0) ---
        if rank == 0:
            os.makedirs(_model_dir, exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(_model_dir, "autoencoder_final.pt"),
            )
            with open(os.path.join(_model_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)
            with open(os.path.join(_model_dir, "scaler_params.json"), "w") as f:
                json.dump({
                    "mean":      mean_np.tolist(),
                    "std":       std_np.tolist(),
                    "features":  _avail_cols,
                    "input_dim": _input_dim,
                    "latent_dim": _config["latent_dim"],
                }, f, indent=2)
            print(f"  ✓ Modelo guardado en {_model_dir}")

        torch.distributed.destroy_process_group()

    # Lanzar con TorchDistributor
    t0 = time.time()
    TorchDistributor(num_processes=args.workers, use_gpu=False).run(worker_fn)
    log.info(f"Entrenamiento completado en {time.time() - t0:.1f}s")
    


def _calculate_threshold(errors: np.ndarray) -> tuple[float, dict]:
    """
    Determina el umbral de anomalía adaptándose a la distribución real de errores.

    Lógica:
      1. Corre Shapiro-Wilk sobre una muestra de hasta 5 000 valores.
         (Shapiro-Wilk requiere n < 5 000; para muestras mayores el test pierde
          potencia y prácticamente siempre rechaza normalidad.)
      2. Si p > 0.05 → distribución compatible con normal → umbral = media + 2σ
         Cubre ~97.7 % de la distribución; los municipios en la cola superior
         son genuinamente atípicos.
      3. Si p ≤ 0.05 → distribución sesgada/no normal (lo más probable con
         errores de reconstrucción) → umbral = percentil 75.
         Más conservador, captura el cuartil superior independientemente de la forma.

    Returns:
        threshold: valor del umbral
        stats_dict: métricas de la distribución + decisión tomada
    """
    mean_e = float(errors.mean())
    std_e  = float(errors.std())

    sample = errors if len(errors) <= 5000 else np.random.choice(errors, 5000, replace=False)
    _, p_value = stats.shapiro(sample)
    is_normal  = bool(p_value > 0.05)

    if is_normal:
        threshold = mean_e + 2 * std_e
        method    = "mean+2sigma"
    else:
        threshold = float(np.percentile(errors, 75))
        method    = "percentile_75"

    log.info(f"  Shapiro-Wilk: p={p_value:.4f} → {'normal' if is_normal else 'no normal'} "
             f"→ método={method} → umbral={threshold:.6f}")

    return threshold, {
        "method":           method,
        "is_normal":        is_normal,
        "shapiro_p_value":  float(p_value),
        "mean":             mean_e,
        "std":              std_e,
        "percentile_25":    float(np.percentile(errors, 25)),
        "percentile_75":    float(np.percentile(errors, 75)),
        "threshold":        threshold,
    }


def _classify_anomalies(errors_df) -> object:
    """
    Clasifica cada municipio-año en tres categorías cruzando error de
    reconstrucción con puntaje ICFES:

      ANOMALÍA_POSITIVA  — error alto + puntaje > mediana
        Municipios que rinden MEJOR de lo esperado dado su perfil de conectividad.
      ANOMALÍA_NEGATIVA  — error alto + puntaje ≤ mediana
        Municipios en situación CRÍTICA: bajo rendimiento y perfil atípico.
      TÍPICO             — error dentro del umbral (la mayoría)

    Requiere columnas: is_anomaly (bool), puntaje_global_promedio (float).
    Si puntaje_global_promedio no está disponible, todas las anomalías se
    marcan como ANOMALÍA_SIN_PUNTAJE.
    """
    import pandas as pd

    df = errors_df.copy()

    if "puntaje_global_promedio" in df.columns and df["puntaje_global_promedio"].notna().any():
        median_score = df["puntaje_global_promedio"].median()

        def _cls(row):
            if not row["is_anomaly"]:
                return "TÍPICO"
            return (
                "ANOMALÍA_POSITIVA"
                if row["puntaje_global_promedio"] > median_score
                else "ANOMALÍA_NEGATIVA"
            )

        df["anomaly_class"] = df.apply(_cls, axis=1)
        log.info(f"  Mediana puntaje ICFES: {median_score:.2f}")
    else:
        log.warning("  puntaje_global_promedio no disponible — usando clasificación binaria")
        df["anomaly_class"] = df["is_anomaly"].map(
            {True: "ANOMALÍA_SIN_PUNTAJE", False: "TÍPICO"}
        )

    return df


def _build_feature_contributions(feat_errors_np: np.ndarray,
                                  recon_errors: np.ndarray,
                                  classified_df,
                                  avail_feat: list) -> object:
    """
    Construye DataFrame largo (un registro por municipio × feature) con la
    contribución porcentual de cada feature al error total de ese municipio.

    Formato de salida:
      municipio | cod_municipio_norm | year_int | anomaly_class |
      feature | feature_error | global_error | feature_error_pct
    """
    import pandas as pd

    rows = []
    for i in range(len(classified_df)):
        row    = classified_df.iloc[i]
        mun    = row.get("municipio", f"mun_{i}")
        cod    = row.get("cod_municipio_norm", "")
        year   = row.get("year_int", "")
        cls    = row["anomaly_class"]
        g_err  = float(recon_errors[i])

        for j, feat in enumerate(avail_feat):
            f_err = float(feat_errors_np[i, j])
            rows.append({
                "municipio":         mun,
                "cod_municipio_norm": cod,
                "year_int":          year,
                "anomaly_class":     cls,
                "feature":           feat,
                "feature_error":     f_err,
                "global_error":      g_err,
                "feature_error_pct": round(f_err / (g_err + 1e-10) * 100, 2),
            })

    return pd.DataFrame(rows)


def _generate_report(classified_df) -> str:
    """Genera reporte de texto para consola."""
    lines = []
    lines.append("=" * 60)
    lines.append("  ANÁLISIS DE ANOMALÍAS TERRITORIALES — AUTOENCODER")
    lines.append("=" * 60)

    total     = len(classified_df)
    counts    = classified_df["anomaly_class"].value_counts().to_dict()
    positivas = counts.get("ANOMALÍA_POSITIVA", 0)
    negativas = counts.get("ANOMALÍA_NEGATIVA", 0)
    tipicas   = counts.get("TÍPICO", 0)

    lines.append("\nResumen:")
    lines.append(f"  Total municipios-año : {total:,}")
    lines.append(f"  Anomalías positivas  : {positivas:,}  ({100*positivas/total:.1f}%)")
    lines.append(f"  Anomalías negativas  : {negativas:,}  ({100*negativas/total:.1f}%)")
    lines.append(f"  Típicos              : {tipicas:,}  ({100*tipicas/total:.1f}%)")

    has_mun  = "municipio" in classified_df.columns
    has_dep  = "departamento" in classified_df.columns
    has_punt = "puntaje_global_promedio" in classified_df.columns

    for cls_label, cls_title in [
        ("ANOMALÍA_POSITIVA", "Top 5 Anomalías Positivas (Éxito inesperado)"),
        ("ANOMALÍA_NEGATIVA", "Top 5 Anomalías Negativas (Situación crítica)"),
    ]:
        subset = classified_df[classified_df["anomaly_class"] == cls_label]
        if subset.empty:
            continue
        top5 = subset.nlargest(5, "reconstruction_error")
        lines.append(f"\\n{cls_title}:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            mun = row["municipio"] if has_mun else row.get("cod_municipio_norm", "?")
            dep = f" ({row['departamento']})" if has_dep else ""
            punt = f" | puntaje={row['puntaje_global_promedio']:.1f}" if has_punt else ""
            lines.append(
                f"  {i}. {mun}{dep} | error={row['reconstruction_error']:.4f}{punt}"
            )

    lines.append("\\n" + "=" * 60)
    return "\\n".join(lines)


# ---------------------------------------------------------------------------
# Fase 5 + 6 + 7: Evaluación y reporte de anomalías
# ---------------------------------------------------------------------------

def evaluate_and_report(spark, test_path: str, model_dir: str, output_dir: str):
    """
    Fases 5-7: carga modelo, calcula errores, determina umbral adaptativo,
    clasifica anomalías en tres categorías, analiza contribución por feature
    y guarda outputs separados en HDFS.

    Outputs en output_dir/:
      anomalias_clasificadas/   — CSV completo con is_anomaly + anomaly_class
      feature_contributions/    — CSV largo (municipio × feature)
      top_anomalies/            — Top 20 de cada clase
      anomaly_summary/          — Estadísticas agrupadas por clase
    """
    import torch
    import pandas as pd

    _section("Fase 5: Evaluación del modelo")

    # --- Cargar scaler e historial ---
    scaler_path = os.path.join(model_dir, "scaler_params.json")
    with open(scaler_path) as f:
        scaler = json.load(f)

    mean_arr  = np.array(scaler["mean"], dtype=np.float32)
    std_arr   = np.array(scaler["std"],  dtype=np.float32)
    features  = scaler["features"]
    input_dim = scaler["input_dim"]
    log.info(f"Scaler cargado: {input_dim} features")

    hist_path = os.path.join(model_dir, "training_history.json")
    with open(hist_path) as f:
        history = json.load(f)

    epochs_run     = len(history["train_loss"])
    final_train    = history["train_loss"][-1]
    final_val      = history["val_loss"][-1]
    best_val_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss  = min(history["val_loss"])
    log.info(f"Historial: {epochs_run} épocas | train={final_train:.6f} | "
             f"val={final_val:.6f} | mejor val={best_val_loss:.6f} (época {best_val_epoch})")

    # --- Cargar modelo ---
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)
    from ml_autoencoder import AutoencoderMunicipal

    model_path = os.path.join(model_dir, "autoencoder_final.pt")
    model = AutoencoderMunicipal(input_dim=input_dim, latent_dim=scaler.get("latent_dim", 4))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    log.info(f"Modelo cargado: {model_path}")

    # --- Cargar test set desde HDFS ---
    test_df    = spark.read.parquet(test_path)
    test_count = test_df.count()
    log.info(f"Test set: {test_count:,} municipios-año")

    avail_feat = [c for c in features if c in test_df.columns]
    id_cols    = [c for c in test_df.columns if c not in avail_feat]

    test_feat_pd = test_df.select(avail_feat).toPandas()
    test_ids_pd  = test_df.select(id_cols).toPandas()

    # --- Forward pass ---
    test_np   = test_feat_pd.values.astype(np.float32)
    test_norm = (test_np - mean_arr[:len(avail_feat)]) / std_arr[:len(avail_feat)]

    with torch.no_grad():
        x_tensor       = torch.from_numpy(test_norm)
        x_recon        = model(x_tensor)
        recon_errors   = torch.mean((x_tensor - x_recon) ** 2, dim=1).numpy()
        feat_errors_np = ((x_tensor - x_recon) ** 2).numpy()   # (n, features)

    log.info(f"Errores — mín={recon_errors.min():.6f} | máx={recon_errors.max():.6f} | "
             f"media={recon_errors.mean():.6f} | std={recon_errors.std():.6f}")

    # --- Umbral adaptativo (Shapiro-Wilk) ---
    _section("Fase 6: Clasificación de anomalías")
    threshold, threshold_stats = _calculate_threshold(recon_errors)

    is_anomaly  = recon_errors > threshold
    n_anomalies = int(is_anomaly.sum())
    pct_anom    = 100.0 * n_anomalies / len(is_anomaly)
    log.info(f"Anomalías detectadas: {n_anomalies:,} / {test_count:,} ({pct_anom:.1f}%)")

    # --- Construir DataFrame base ---
    results_pd = test_ids_pd.copy()
    results_pd["reconstruction_error"] = recon_errors
    results_pd["error_log10"]          = np.log10(recon_errors + 1e-10)
    results_pd["anomaly_score_pct"]    = (
        pd.Series(recon_errors).rank(pct=True) * 100
    ).round(2).values
    results_pd["is_anomaly"] = is_anomaly

    # --- Clasificación tripartita ---
    results_pd = _classify_anomalies(results_pd)

    counts = results_pd["anomaly_class"].value_counts().to_dict()
    for cls, n in sorted(counts.items()):
        log.info(f"  {cls}: {n:,} ({100*n/test_count:.1f}%)")

    # --- Contribución por feature ---
    _section("Fase 7: Reporte y exportación")
    contrib_df = _build_feature_contributions(
        feat_errors_np, recon_errors, results_pd, avail_feat
    )

    # --- Guardar outputs en HDFS via Spark ---
    def _save_csv(df, name: str) -> str:
        path = f"{output_dir}/{name}"
        spark.createDataFrame(df).coalesce(1).write.mode("overwrite").option("header", True).csv(path)
        log.info(f"  Guardado: {path}")
        return path

    # anomalias_clasificadas: todo el test set con clasificación
    classified_out = results_pd.copy()
    classified_out["is_anomaly"] = classified_out["is_anomaly"].astype(int)
    _save_csv(classified_out, "anomalias_clasificadas")

    # feature_contributions: formato largo
    _save_csv(contrib_df, "feature_contributions")

    # top_anomalies: top 20 de cada clase que no sea TÍPICO
    top_parts = []
    for cls in [c for c in counts if c != "TÍPICO"]:
        subset = results_pd[results_pd["anomaly_class"] == cls]
        top_parts.append(subset.nlargest(20, "reconstruction_error"))
    if top_parts:
        top_df = pd.concat(top_parts).copy()
        top_df["is_anomaly"] = top_df["is_anomaly"].astype(int)
        _save_csv(top_df, "top_anomalies")

    # anomaly_summary: estadísticas agrupadas por clase
    agg_cols = {"reconstruction_error": ["count", "mean", "std", "min", "max"]}
    if "puntaje_global_promedio" in results_pd.columns:
        agg_cols["puntaje_global_promedio"] = ["mean", "std"]
    summary_df = results_pd.groupby("anomaly_class").agg(agg_cols).reset_index()
    summary_df.columns = ["_".join(c).strip("_") for c in summary_df.columns]
    _save_csv(summary_df, "anomaly_summary")

    # --- Reporte de consola ---
    report_str = _generate_report(results_pd)
    print(report_str)

    # --- Resumen JSON ---
    summary = {
        "threshold": threshold_stats,
        "n_test":    test_count,
        "anomaly_counts": counts,
        "pct_anomalies":  round(pct_anom, 2),
        "recon_error_stats": {
            "min":  float(recon_errors.min()),
            "max":  float(recon_errors.max()),
            "mean": float(recon_errors.mean()),
            "std":  float(recon_errors.std()),
        },
        "training": {
            "epochs_run":       epochs_run,
            "final_train_loss": final_train,
            "final_val_loss":   final_val,
            "best_val_loss":    best_val_loss,
            "best_val_epoch":   best_val_epoch,
        },
        "features":  avail_feat,
        "input_dim": input_dim,
    }
    summary_path = os.path.join(model_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Resumen JSON: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline autoencoder: carga → feature eng → split → entrenamiento distribuido → evaluación",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Entorno
    p.add_argument("--hdfs-uri",       default=os.environ.get("HDFS_URI", "hdfs://spark-master:9000"))
    p.add_argument("--local",          action="store_true",
                   help="Forzar modo local (local[*]), sin clúster Spark")

    # Rutas HDFS
    p.add_argument("--parquet-clean-dir", default=None,
                   help="Dir raíz de parquet_clean en HDFS (default: {hdfs-uri}/data/parquet_clean)")
    p.add_argument("--model-dir",         default=None,
                   help="Dir donde guardar el modelo (default: {hdfs-uri}/data/models/autoencoder_municipal)")
    p.add_argument("--output-dir",        default=None,
                   help="Dir para el reporte de anomalías (default: {hdfs-uri}/data/analysis/anomalies)")

    # Entrenamiento
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-5)
    p.add_argument("--early-stopping",  type=int,   default=10)
    p.add_argument("--latent-dim",      type=int,   default=4)
    p.add_argument("--workers",         type=int,   default=2,
                   help="Número de procesos TorchDistributor (workers DDP)")

    # Control de flujo
    p.add_argument("--skip-training",   action="store_true",
                   help="Omitir fase 4 y cargar modelo ya entrenado")
    p.add_argument("--skip-eval",       action="store_true",
                   help="Omitir fases 5-6 (solo entrenar)")

    return p.parse_args()


def main():
    args = parse_args()

    hdfs           = args.hdfs_uri
    parquet_clean  = args.parquet_clean_dir or f"{hdfs}/data/parquet_clean"
    model_dir      = args.model_dir         or f"{hdfs}/data/models/autoencoder_municipal"
    output_dir     = args.output_dir        or f"{hdfs}/data/analysis/anomalies"
    ae_datasets    = f"{parquet_clean}/ae_datasets"

    _section("Pipeline Autoencoder — Detección de Anomalías Territoriales")
    log.info(f"HDFS:          {hdfs}")
    log.info(f"parquet_clean: {parquet_clean}")
    log.info(f"model_dir:     {model_dir}")
    log.info(f"output_dir:    {output_dir}")
    log.info(f"project_path:  {PROJECT_PATH}")
    log.info(f"skip_training: {args.skip_training}")

    t_total = time.time()

    # -----------------------------------------------------------------------
    # Crear SparkSession
    # -----------------------------------------------------------------------
    spark, _ = build_spark(local=args.local)

    # -----------------------------------------------------------------------
    # Fases 1 + 2: Carga y feature engineering
    # -----------------------------------------------------------------------
    final_df, available_cols, id_cols = load_and_prepare_data(spark, parquet_clean)

    # -----------------------------------------------------------------------
    # Fase 3: Split + guardado en HDFS
    # -----------------------------------------------------------------------
    train_path, val_path, test_path, _ = split_and_save(
        spark, final_df, available_cols, id_cols, ae_datasets
    )

    # -----------------------------------------------------------------------
    # Fase 4: Entrenamiento distribuido
    # -----------------------------------------------------------------------
    if not args.skip_training:
        run_distributed_training(
            spark, train_path, val_path, model_dir, available_cols, args
        )
    else:
        log.info("Fase 4 omitida (--skip-training). Usando modelo existente.")

    # -----------------------------------------------------------------------
    # Fases 5 + 6: Evaluación + reporte
    # -----------------------------------------------------------------------
    if not args.skip_eval:
        summary = evaluate_and_report(
            spark, test_path, model_dir, output_dir
        )
        _section("Resumen final")
        th = summary["threshold"]
        log.info(f"Umbral: {th['threshold']:.6f} (método={th['method']}, "
                 f"shapiro_p={th['shapiro_p_value']:.4f})")
        for cls, n in summary["anomaly_counts"].items():
            pct = 100 * n / summary["n_test"]
            log.info(f"  {cls}: {n:,} ({pct:.1f}%)")
        log.info(f"Best val loss: {summary['training']['best_val_loss']:.6f} "
                 f"(época {summary['training']['best_val_epoch']})")

    spark.stop()
    log.info(f"Pipeline completado en {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()