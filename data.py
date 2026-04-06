"""
Ingesta desde datos.gov.co (Socrata) y conversión a Parquet con PySpark.

Variables de entorno:
  SOCRATA_APP_TOKEN     Obligatoria para la extracción. Token de aplicación Socrata.

  SPARK_MASTER_URL      Si está definida (ej. spark://spark-master:7077), se usa el
                        clúster Standalone; si no, modo local[*].

  SPARK_DRIVER_MEMORY   Por defecto: 2g (local) o 4g (clúster).
  SPARK_EXECUTOR_INSTANCES   Solo clúster. Por defecto: 3 (tres workers dedicados).
  SPARK_EXECUTOR_CORES       Solo clúster. Por defecto: 4 (VMs con 4 vCPU).
  SPARK_EXECUTOR_MEMORY      Solo clúster. Por defecto: 8g (VMs con 12 GB RAM).
  SPARK_DEFAULT_PARALLELISM  Por defecto: executor_instances * executor_cores en clúster;
                        en local, omite o usa el paralelismo por defecto de Spark.

  SPARK_HOME_USE_SYSTEM    Si es 1/true, no se ajusta SPARK_HOME (usa el del entorno).
                        Por defecto se alinea con el pyspark del venv para evitar el error
                        TypeError: 'JavaPackage' object is not callable (mezcla pip + tarball).

  PySpark 4.x requiere Java 17 o 21 (comprueba: java -version y JAVA_HOME).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


import requests

import pyspark
from pyspark.sql import SparkSession

DATASETS = {
    "icfes": "kgxf-xxbe",        # Resultados únicos Saber 11 (histórico)
    "internet": "n48w-gutb",     # Accesos internet por municipio
    "bachillerato": "nudc-7mev", # Cobertura educativa por municipio
    "pobreza": "gb6q-tu7m",
}

TOKEN_ENV = "SOCRATA_APP_TOKEN"
LIMIT = 50000
RAW_DIR = "data/raw_json"
PARQUET_DIR = "data/parquet"

# Tope de hilos al bajar varios datasets en paralelo (entre datasets, no entre páginas)
MAX_PARALLEL_DATASET_DOWNLOADS = 3


def get_app_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(
            f"Define {TOKEN_ENV} con tu token de aplicación Socrata, "
            f"por ejemplo: export {TOKEN_ENV}='tu-token'"
        )
    return token


def _extract_one_dataset(dataset_name: str, dataset_id: str, app_token: str, sample: int | None = None) -> None:
    headers = {"X-App-Token": app_token}
    print(f"\n--- Procesando dataset: {dataset_name} ({dataset_id}) ---")
    api_url = f"https://www.datos.gov.co/resource/{dataset_id}.json"
    dataset_raw_dir = os.path.join(RAW_DIR, dataset_name)
    os.makedirs(dataset_raw_dir, exist_ok=True)

    offset = 0
    chunk_idx = 0
    total_fetched = 0

    while True:
        if sample is not None:
            remaining = sample - total_fetched
            if remaining <= 0:
                break
            batch_size = min(LIMIT, remaining)
        else:
            batch_size = LIMIT

        print(f"[{dataset_name}] -> Pidiendo lote desde el offset {offset} (batch_size={batch_size})...")
        params = {
            "$limit": batch_size,
            "$offset": offset,
            "$order": ":id",
        }
        response = requests.get(api_url, headers=headers, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"[{dataset_name}] Fin del dataset (lote vacío).")
            break

        file_path = os.path.join(dataset_raw_dir, f"part_{chunk_idx:04d}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

        total_fetched += len(data)
        print(f"[{dataset_name}]    Guardado: {file_path} ({len(data)} registros, total={total_fetched})")

        if len(data) < batch_size:
            break

        offset += batch_size
        chunk_idx += 1

    if sample is not None:
        print(f"[{dataset_name}] Modo muestra: {total_fetched}/{sample} registros obtenidos.")


def extract_data_to_disk(*, parallel_datasets: bool, sample: int | None = None) -> None:
    app_token = get_app_token()
    if sample is not None:
        print(f"Iniciando extraccion en modo muestra ({sample} registros por dataset)...")
    else:
        print("Iniciando la extraccion estructurada desde datos.gov.co...")

    items = list(DATASETS.items())

    if parallel_datasets and len(items) > 1:
        workers = min(len(items), MAX_PARALLEL_DATASET_DOWNLOADS)
        print(f"Descarga paralela entre datasets (hasta {workers} hilos).")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_extract_one_dataset, name, did, app_token, sample): name
                for name, did in items
            }
            for fut in as_completed(futures):
                fut.result()
    else:
        for dataset_name, dataset_id in items:
            _extract_one_dataset(dataset_name, dataset_id, app_token, sample)

    print("\n====== Extraccion API completada para todos los datasets ======")


def _spark_local_driver_memory() -> str:
    return os.environ.get("SPARK_DRIVER_MEMORY", "2g")


def _align_spark_home_with_pyspark_package() -> None:
    """
    Si SPARK_HOME apunta a otra instalación de Spark que la de este intérprete,
    el driver JVM carga jars distintos a los del paquete pip y aparece
    'JavaPackage' object is not callable al crear SparkSession.
    """
    use_system = os.environ.get("SPARK_HOME_USE_SYSTEM", "").strip().lower()
    if use_system in ("1", "true", "yes"):
        return
    pkg_dir = os.path.dirname(os.path.abspath(pyspark.__file__))
    if not os.path.isdir(os.path.join(pkg_dir, "jars")):
        return
    os.environ["SPARK_HOME"] = pkg_dir


def _build_spark_session() -> SparkSession:
    master_url = os.environ.get("SPARK_MASTER_URL", "").strip()
    app_name = "Ingesta_Datos_ICFES_Internet_GovCO"
    builder = SparkSession.builder.appName(app_name)

    if master_url:
        instances = int(os.environ.get("SPARK_EXECUTOR_INSTANCES", "3"))
        cores = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))
        exec_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "8g")
        driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
        default_par = os.environ.get("SPARK_DEFAULT_PARALLELISM")
        if default_par is None:
            default_par = str(max(1, instances * cores))

        print(
            f"\nModo clúster: master={master_url} | "
            f"executors={instances}×{cores} cores, {exec_mem} cada uno | "
            f"driver={driver_mem}"
        )
        print(
            "Nota: con executors remotos, la ruta data/ debe existir en todos los nodos "
            "o usar almacenamiento compartido (NFS/HDFS); si no, usa solo modo local."
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
    try:
        return builder.getOrCreate()
    except TypeError as e:
        if "JavaPackage" in str(e):
            raise SystemExit(
                "Fallo al iniciar Spark (JavaPackage): suele ser Java incompatible o "
                "SPARK_HOME mezclando la instalación del curso con pyspark de pip.\n"
                "- Usa Java 17 o 21 para PySpark 4.x (export JAVA_HOME=...).\n"
                "- O export SPARK_HOME_USE_SYSTEM=1 si quieres forzar solo el Spark del sistema "
                "(y alinea la versión pip con esa instalación, o usa el Python de SPARK_HOME)."
            ) from e
        raise


def process_with_spark() -> None:
    """
    Lee los .jsonl del directorio crudo y escribe Parquet.
    """
    print("\nIniciando sesión de Spark...")
    spark = _build_spark_session()

    try:
        for dataset_name in DATASETS:
            dataset_raw_dir = os.path.join(RAW_DIR, dataset_name)
            dataset_parquet_dir = os.path.join(PARQUET_DIR, dataset_name)

            print(f"\nLeyendo archivos crudos desde '{dataset_raw_dir}' ...")
            df = spark.read.json(dataset_raw_dir)

            print(f"\nEsquema inferido para '{dataset_name}':")
            df.printSchema()

            total_rows = df.count()
            print(f"\nResumen: Total de registros para '{dataset_name}': {total_rows}")

            print(f"Guardando Parquet en '{dataset_parquet_dir}' ...")
            df.write.mode("overwrite").parquet(dataset_parquet_dir)

        print("\n====== Ingesta a Spark completada exitosamente ======")
    except Exception as e:
        print(f"Error procesando con Spark: {str(e)}")
        raise
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga datasets datos.gov.co y/o convierte JSONL a Parquet con Spark."
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Solo descargar JSONL (requiere SOCRATA_APP_TOKEN).",
    )
    parser.add_argument(
        "--spark-only",
        action="store_true",
        help="Solo ejecutar Spark sobre data/raw_json existente.",
    )
    parser.add_argument(
        "--parallel-downloads",
        action="store_true",
        help="Descargar varios datasets a la vez (paginación sigue siendo secuencial por dataset).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Descargar solo los primeros N registros por dataset (modo desarrollo).",
)
    args = parser.parse_args()

    if args.extract_only and args.spark_only:
        parser.error("No uses --extract-only y --spark-only a la vez.")

    run_extract = args.extract_only or not args.spark_only
    run_spark = args.spark_only or not args.extract_only

    if run_extract:
        extract_data_to_disk(parallel_datasets=args.parallel_downloads, sample=args.sample)

    if run_spark:
        process_with_spark()


if __name__ == "__main__":
    main()
