"""
Ingesta desde datos.gov.co (Socrata) y conversión a Parquet con PySpark.

Descarga registros desde la API de Socrata en páginas de 5,000 (LIMIT) y los agrupa
en archivos HDFS de ~80,000 registros (RECORDS_PER_FILE ≈ 128 MB) cada uno.

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

  PySpark 3.5.x requiere Java 11 (comprueba: java -version y JAVA_HOME).
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pyspark
from pyspark.sql import SparkSession

DATASETS = {
    "icfes": "kgxf-xxbe",
    "internet": "n48w-gutb",
    "bachillerato": "nudc-7mev",
}

TOKEN_ENV = "SOCRATA_APP_TOKEN"
LIMIT = 5000
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 5  # seconds: 5, 10, 20, 40, 80

# Se usará la URL del HDFS para lecturas y descargas directas WebHDFS
HDFS_URI = "hdfs://spark-master:9000"
WEBHDFS_URL = "http://spark-master:9870"
PARQUET_DIR = f"{HDFS_URI}/data/parquet"


def _build_http_session(app_token: str) -> requests.Session:
    """Crea una sesión HTTP con reintentos automáticos y backoff exponencial."""
    session = requests.Session()
    session.headers.update({"X-App-Token": app_token})
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_app_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(
            f"Define {TOKEN_ENV} con tu token de aplicación Socrata, "
            f"por ejemplo: export {TOKEN_ENV}='tu-token'"
        )
    return token


def _hdfs_client():
    """Cliente WebHDFS contra el NameNode (spark-master).

    Corrige el redirect CREATE: el NameNode puede enviar namenoderpcaddress con el
    puerto HTTP (9870); los DataNodes necesitan el puerto RPC (9000).
    """
    import re
    from hdfs import InsecureClient

    client = InsecureClient(WEBHDFS_URL)
    _create = client._create

    def _create_with_rpc_port(*args, **kwargs):
        res = _create(*args, **kwargs)
        loc = res.headers.get("location")
        if loc:
            res.headers["location"] = re.sub(
                r"(namenoderpcaddress=[^:&]+):9870",
                r"\1:9000",
                loc,
            )
        return res

    client._create = _create_with_rpc_port
    return client


def _get_existing_parts(hdfs_client, dataset_raw_hdfs_dir: str) -> set:
    """Escanea HDFS y retorna los índices de partes ya descargadas."""
    existing = set()
    try:
        files = hdfs_client.list(dataset_raw_hdfs_dir, status=False)
        for fname in files:
            # Formato: part_0000.jsonl -> extraer el índice 0
            if fname.startswith("part_") and fname.endswith(".jsonl"):
                try:
                    idx = int(fname.replace("part_", "").replace(".jsonl", ""))
                    existing.add(idx)
                except ValueError:
                    pass
    except Exception:
        pass  # Directorio no existe aún
    return existing


def _extract_one_dataset(dataset_name: str, dataset_id: str, app_token: str, parallel_downloads: bool, sample=None) -> None:
    http_session = _build_http_session(app_token)
    print(f"\n--- Procesando dataset: {dataset_name} ({dataset_id}) ---")
    api_url = f"https://www.datos.gov.co/resource/{dataset_id}.json"

    hdfs_client = _hdfs_client()
    dataset_raw_hdfs_dir = f"/data/raw_json/{dataset_name}"

    # Crear directorio si no existe (NO borrar datos previos para permitir reanudación)
    hdfs_client.makedirs(dataset_raw_hdfs_dir)

    total_records = 0
    if sample is not None:
        total_records = sample
        print(f"[{dataset_name}] Modo muestra activado: Limitado a {sample} registros.")
    else:
        print(f"[{dataset_name}] Determinando tamaño total del dataset en Socrata...")
        count_url = f"{api_url}?$select=count(*)"
        resp = http_session.get(count_url, timeout=60)
        resp.raise_for_status()
        total_records = int(resp.json()[0]["count"])
        print(f"[{dataset_name}] Registros totales estimados: {total_records}")

    if total_records == 0:
        print(f"[{dataset_name}] No hay datos en el dataset.")
        return

    offsets = list(range(0, total_records, LIMIT))

    # Detectar partes ya descargadas para reanudar
    existing_parts = _get_existing_parts(hdfs_client, dataset_raw_hdfs_dir)
    if existing_parts:
        print(f"[{dataset_name}] Reanudando: {len(existing_parts)}/{len(offsets)} lotes ya existen en HDFS, se omitirán.")

    def download_chunk(offset_tuple):
        chunk_idx, offset = offset_tuple

        # Saltar si ya existe en HDFS
        if chunk_idx in existing_parts:
            return 0

        remaining = total_records - offset
        batch_size = min(LIMIT, remaining)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[{dataset_name}] -> Pidiendo lote {chunk_idx:04d} (offset={offset}, batch_size={batch_size})...")
                params = {
                    "$limit": batch_size,
                    "$offset": offset,
                    "$order": ":id",
                }
                res = http_session.get(api_url, params=params, timeout=120)
                res.raise_for_status()
                data = res.json()

                if not data:
                    return 0

                # Guardar en HDFS via WebHDFS (zero local disk footprint)
                hdfs_path = f"{dataset_raw_hdfs_dir}/part_{chunk_idx:04d}.jsonl"
                with hdfs_client.write(hdfs_path, encoding="utf-8") as writer:
                    for record in data:
                        writer.write(json.dumps(record) + "\n")

                return len(data)

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"[{dataset_name}] ⚠ Lote {chunk_idx:04d} falló (intento {attempt}/{MAX_RETRIES}): {type(e).__name__}. Reintentando en {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"[{dataset_name}] Lote {chunk_idx:04d} falló tras {MAX_RETRIES} intentos.")

    total_fetched = 0
    pending = [(idx, off) for idx, off in enumerate(offsets) if idx not in existing_parts]
    print(f"[{dataset_name}] Lotes pendientes: {len(pending)} de {len(offsets)} totales.")

    if parallel_downloads:
        workers = min(10, len(pending))
        print(f"[{dataset_name}] Descarga concurrente ({workers} hilos).")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(download_chunk, (idx, off)): idx for idx, off in pending}
            for fut in as_completed(futures):
                total_fetched += fut.result()
    else:
        for idx, off in pending:
            total_fetched += download_chunk((idx, off))

    skipped = len(existing_parts)
    print(f"[{dataset_name}] Fin. Descargados {total_fetched} registros nuevos ({skipped} lotes reutilizados de HDFS).")


def extract_data_to_hdfs(*, parallel_downloads: bool, sample=None) -> None:
    app_token = get_app_token()
    if sample is not None:
        print(f"Iniciando extraccion en modo muestra hacia HDFS ({sample} registros por dataset)...")
    else:
        print("Iniciando la extraccion estructurada desde datos.gov.co directo a HDFS...")

    for dataset_name, dataset_id in DATASETS.items():
        _extract_one_dataset(dataset_name, dataset_id, app_token, parallel_downloads, sample)

    print("\n====== Extraccion API hacia HDFS completada ======")


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
    Lee los .jsonl distribuidos en HDFS y escribe Parquet borrando crudos despues.
    """
    print("\nIniciando sesión de Spark...")
    spark = _build_spark_session()

    try:
        sc = spark.sparkContext
        Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
        fs = Path(f"{HDFS_URI}/").getFileSystem(sc._jsc.hadoopConfiguration())

        for dataset_name in DATASETS:
            dataset_raw_dir = f"{HDFS_URI}/data/raw_json/{dataset_name}"
            dataset_parquet_dir = f"{HDFS_URI}/data/parquet/{dataset_name}"

            print(f"\nLeyendo archivos crudos desde '{dataset_raw_dir}' ...")
            df = spark.read.json(dataset_raw_dir)

            print(f"\nEsquema inferido para '{dataset_name}':")
            df.printSchema()

            total_rows = df.count()
            print(f"\nResumen: Total de registros para '{dataset_name}': {total_rows}")

            print(f"Guardando Parquet distribuido en '{dataset_parquet_dir}' ...")
            df.coalesce(8).write.mode("overwrite").parquet(dataset_parquet_dir)

            # Auto-borrado de crudos en HDFS usando la JVM via Py4J
            print(f"Limpiando capa cruda JSON: borrando {dataset_raw_dir} ...")
            try:
                fs.delete(Path(dataset_raw_dir), True)
            except Exception as jvm_err:
                print(f"Advertencia: no se pudo borrar el raw JSON para {dataset_name}: {jvm_err}")

        print("\n====== Transformación a Parquet distribuido completada ======")
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
        extract_data_to_hdfs(parallel_downloads=args.parallel_downloads, sample=args.sample)

    if run_spark:
        process_with_spark()


if __name__ == "__main__":
    main()
