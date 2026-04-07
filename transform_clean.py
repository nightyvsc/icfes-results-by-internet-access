"""
Limpieza, filtros y transformación inicial (entrega 1, num.I7).

Lee Parquet generado por data.py en data/parquet/ y escribe versiones limpias
en data/parquet_clean/. Solo PySpark (sin pandas).

Variables de entorno: mismas que data.py (SPARK_MASTER_URL, SPARK_DRIVER_MEMORY, etc.).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

from data import PARQUET_DIR, _align_spark_home_with_pyspark_package, _spark_local_driver_memory

PARQUET_CLEAN_DIR = "data/parquet_clean"

ICFES_SCORE_COLS = [
    "punt_c_naturales",
    "punt_global",
    "punt_ingles",
    "punt_lectura_critica",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
]

INTERNET_NUM_COLS = ["no_de_accesos", "velocidad_bajada", "velocidad_subida"]

BACH_NUM_COLS = [
    "cobertura_neta",
    "cobertura_neta_primaria",
    "cobertura_neta_secundaria",
    "cobertura_neta_media",
    "cobertura_bruta",
    "deserci_n",
    "deserci_n_primaria",
    "deserci_n_secundaria",
    "aprobaci_n",
    "reprobaci_n",
    "repitencia",
    "sedes_conectadas_a_internet",
    "tasa_matriculaci_n_5_16",
    "poblaci_n_5_16",
    "tama_o_promedio_de_grupo",
]


def _quiet_log4j_options() -> str | None:
    """Java system property so Spark uses project quiet-log4j2.properties (less stdout/stderr noise)."""
    cfg = Path(__file__).resolve().parent / "quiet-log4j2.properties"
    if not cfg.is_file():
        return None
    return f"-Dlog4j2.configurationFile=file:{cfg}"


class _StageProgressBar:
    """Simple single-line progress bar (no extra dependency)."""

    def __init__(self, total: int, *, enabled: bool) -> None:
        self._total = max(int(total), 1)
        self._current = 0
        self._enabled = enabled

    def advance(self, label: str) -> None:
        if not self._enabled:
            return
        self._current += 1
        width = 32
        filled = min(width, int(width * self._current / self._total))
        bar = "#" * filled + "-" * (width - filled)
        tail = (label[:46] + "…") if len(label) > 47 else label
        sys.stdout.write(f"\r[{bar}] {self._current}/{self._total} {tail}")
        sys.stdout.flush()
        if self._current >= self._total:
            sys.stdout.write("\n")


def build_spark_session() -> SparkSession:
    """Misma lógica de clúster/local que data.py; distinto nombre de app."""
    master_url = os.environ.get("SPARK_MASTER_URL", "").strip()
    app_name = "Transform_Limpieza_ICFES_Internet"
    builder = SparkSession.builder.appName(app_name)

    quiet_java = _quiet_log4j_options()
    builder = (
        builder.config("spark.ui.showConsoleProgress", "false")
        .config("spark.logConf", "false")
        .config("spark.sql.debug.maxToStringFields", "8")
    )
    if quiet_java:
        builder = builder.config("spark.driver.extraJavaOptions", quiet_java)

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
        builder = (
            builder.master(master_url)
            .config("spark.executor.instances", str(instances))
            .config("spark.executor.cores", str(cores))
            .config("spark.executor.memory", exec_mem)
            .config("spark.driver.memory", driver_mem)
            .config("spark.default.parallelism", default_par)
        )
        if quiet_java:
            builder = builder.config("spark.executor.extraJavaOptions", quiet_java)
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
                "Fallo al iniciar Spark (JavaPackage). Revisa JAVA_HOME y SPARK_HOME; "
                "ver mensaje de ayuda en data.py."
            ) from e
        raise


def _norm_muni(col_name: str):
    """Código DANE municipio: trim + relleno a 5 dígitos (si existe valor)."""
    c = F.trim(F.col(col_name).cast(StringType()))
    return F.when(c == "", None).otherwise(F.lpad(c, 5, "0"))


def _parse_string_to_double(col: F.Column) -> F.Column:
    """
    Convierte strings numéricos a double. Soporta:
    - Coma decimal (internet): '6,00'
    - Punto decimal (ICFES / indicadores): '108.73'
    - Miles + coma decimal: '1.234,56'
    - Miles solo con puntos (MEN): '1.174.274', '386.466'
    Normaliza separadores tipo punto en Unicode a ASCII. Cadenas no numéricas → NULL.
    """
    raw = F.trim(col.cast(StringType()))
    c = F.regexp_replace(raw, r"[\u00B7\u2219\u22C5\u2022\u2024\u30FB]", ".")
    c = F.when((c.isNull()) | (c == ""), None).otherwise(c)
    has_comma = F.coalesce(F.instr(c, ",") > F.lit(0), F.lit(False))
    has_dot = F.coalesce(F.instr(c, ".") > F.lit(0), F.lit(False))

    parts = F.split(c, r"[.]")
    n_parts = F.size(parts)
    p1 = F.element_at(parts, 1)
    p2 = F.element_at(parts, 2)
    # Dos o más puntos → '1.174.274'. Un punto y bloque derecho de 3 dígitos (+ izq. 1-3) → '386.466'
    two_or_more_dot_separators = n_parts > F.lit(2)
    one_dot_thousands = (n_parts == F.lit(2)) & p1.rlike(r"^[0-9]{1,3}$") & p2.rlike(r"^[0-9]{3}$")
    thousands_dots_only = (
        (~has_comma) & has_dot & (two_or_more_dot_separators | one_dot_thousands)
    )

    normalized = (
        F.when(c.isNull(), None)
        .when(has_comma & has_dot, F.regexp_replace(F.regexp_replace(c, r"\.", ""), ",", "."))
        .when(has_comma, F.regexp_replace(c, ",", "."))
        .when(thousands_dots_only, F.regexp_replace(c, r"\.", ""))
        .otherwise(c)
    )
    looks_numeric = normalized.rlike(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$") | normalized.rlike(
        r"^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$"
    )
    return (
        F.when(normalized.isNull(), F.lit(None).cast(DoubleType()))
        .when(looks_numeric, normalized.cast(DoubleType()))
        .otherwise(F.lit(None).cast(DoubleType()))
    )


def _cast_existing_numeric(df: DataFrame, cols: list[str]) -> DataFrame:
    present = [c for c in cols if c in df.columns]
    out = df
    for c in present:
        out = out.withColumn(c, _parse_string_to_double(F.col(c)))
    return out


def _icfes_period_year_expr():
    """Saber 11: periodo suele ser YYYYS; primeros 4 = año."""
    p = F.trim(F.col("periodo").cast(StringType()))
    return F.when((p.isNull()) | (p == ""), None).otherwise(
        F.substring(p, 1, 4).cast(IntegerType())
    )


def prepare_icfes(df: DataFrame) -> DataFrame:
    out = _cast_existing_numeric(df, ICFES_SCORE_COLS)
    if "cole_cod_mcpio_ubicacion" in out.columns:
        out = out.withColumn("cod_municipio_norm", _norm_muni("cole_cod_mcpio_ubicacion"))
    else:
        out = out.withColumn("cod_municipio_norm", F.lit(None).cast(StringType()))

    if "periodo" in out.columns:
        out = out.withColumn("year_icfes", _icfes_period_year_expr())
    else:
        out = out.withColumn("year_icfes", F.lit(None).cast(IntegerType()))

    # Filtro: solo estudiantes con resultados publicados (excluye investigación / pendientes).
    if "estu_estadoinvestigacion" in out.columns:
        st = F.upper(F.trim(F.col("estu_estadoinvestigacion").cast(StringType())))
        out = out.filter(st == "PUBLICADO")

    out = out.filter(F.col("cod_municipio_norm").isNotNull())

    # Puntajes Saber 11 ~0–500; eliminar filas fuera de rango (datos alterados).
    if "punt_global" in out.columns:
        out = out.filter(
            F.col("punt_global").isNull()
            | ((F.col("punt_global") >= 0) & (F.col("punt_global") <= 500))
        )

    # Atípicos por IQR sobre punt_global (avance metodológico; entrega 2 puede refinar).
    if "punt_global" in out.columns:
        out = apply_iqr_filter(out, "punt_global")

    # Imputación por media: punt_ingles (ejemplo tipo PDF) — solo donde sigue null.
    if "punt_ingles" in out.columns:
        mean_pi = out.select(F.mean("punt_ingles")).first()[0]
        if mean_pi is not None:
            out = out.fillna({"punt_ingles": float(mean_pi)})

    return out


def prepare_internet(df: DataFrame) -> DataFrame:
    out = _cast_existing_numeric(df, INTERNET_NUM_COLS)

    if "cod_municipio" in out.columns:
        out = out.withColumn("cod_municipio_norm", _norm_muni("cod_municipio"))
    else:
        out = out.withColumn("cod_municipio_norm", F.lit(None).cast(StringType()))

    if "anno" in out.columns:
        out = out.withColumn(
            "year_int",
            F.when(F.trim(F.col("anno").cast(StringType())) == "", None).otherwise(
                F.trim(F.col("anno").cast(StringType())).cast(IntegerType())
            ),
        )
    else:
        print(
            "Aviso: columna 'anno' no está en internet; no se construirá join anual "
            "(¿esquema amplio legacy?).",
            file=sys.stderr,
        )
        out = out.withColumn("year_int", F.lit(None).cast(IntegerType()))

    out = out.filter(F.col("cod_municipio_norm").isNotNull())
    if "no_de_accesos" in out.columns:
        out = out.filter(F.col("no_de_accesos").isNull() | (F.col("no_de_accesos") >= 0))

    if "no_de_accesos" in out.columns:
        out = apply_iqr_filter(out, "no_de_accesos")

    return out


def prepare_bachillerato(df: DataFrame) -> DataFrame:
    out = _cast_existing_numeric(df, BACH_NUM_COLS)
    if "c_digo_municipio" in out.columns:
        out = out.withColumn("cod_municipio_norm", _norm_muni("c_digo_municipio"))
    else:
        out = out.withColumn("cod_municipio_norm", F.lit(None).cast(StringType()))

    if "a_o" in out.columns:
        out = out.withColumn(
            "year_int",
            F.when(F.trim(F.col("a_o").cast(StringType())) == "", None).otherwise(
                F.trim(F.col("a_o").cast(StringType())).cast(IntegerType())
            ),
        )
    else:
        out = out.withColumn("year_int", F.lit(None).cast(IntegerType()))

    out = out.filter(F.col("cod_municipio_norm").isNotNull())

    # Imputación por media: cobertura_neta (ejemplo tipo PDF).
    if "cobertura_neta" in out.columns:
        mean_cn = out.select(F.mean("cobertura_neta")).first()[0]
        if mean_cn is not None:
            out = out.fillna({"cobertura_neta": float(mean_cn)})

    return out


def apply_iqr_filter(df: DataFrame, colname: str) -> DataFrame:
    """
    Elimina extremos fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (regla de Tukey).
    Solo filas con valor no nulo en colname se usan para calcular cuartiles.
    """
    if colname not in df.columns:
        return df
    vals = df.select(colname).filter(F.col(colname).isNotNull())
    if vals.limit(1).count() == 0:
        return df
    qs = df.approxQuantile(colname, [0.25, 0.75], relativeError=0.01)
    if not qs or qs[0] is None or qs[1] is None:
        return df
    q1, q3 = float(qs[0]), float(qs[1])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df.filter(F.col(colname).isNull() | ((F.col(colname) >= low) & (F.col(colname) <= high)))


def build_municipio_internet_cobertura(
    df_internet: DataFrame,
    df_bach: DataFrame,
) -> DataFrame:
    """
    Agrega accesos por municipio-año y une cobertura educativa.
    Transformación: accesos_por_habitante = sum(no_de_accesos) / poblaci_n_5_16.
    """
    key_cols = ["cod_municipio_norm", "year_int"]
    inet_agg = (
        df_internet.filter(F.col("year_int").isNotNull())
        .groupBy(*key_cols)
        .agg(F.sum("no_de_accesos").alias("total_accesos_internet"))
    )

    b_sel = df_bach.filter(F.col("year_int").isNotNull()).select(
        "cod_municipio_norm",
        "year_int",
        "poblaci_n_5_16",
        "cobertura_neta",
        "municipio",
        "departamento",
    )

    joined = inet_agg.join(
        b_sel,
        on=["cod_municipio_norm", "year_int"],
        how="inner",
    )

    joined = joined.withColumn(
        "accesos_por_habitante",
        F.when(
            (F.col("poblaci_n_5_16").isNotNull())
            & (F.col("poblaci_n_5_16") > 0),
            F.col("total_accesos_internet") / F.col("poblaci_n_5_16"),
        ).otherwise(None),
    )
    return joined


def summarize_print(name: str, df: DataFrame, show_rows: int = 20) -> None:
    print("\n" + "=" * 60)
    print(f"RESULTADO: {name}")
    print("=" * 60)
    df.printSchema()
    n = df.count()
    print(f"Filas: {n:,}")
    df.show(show_rows, truncate=False)


def run_pipeline(
    *,
    spark: SparkSession,
    parquet_dir: str = PARQUET_DIR,
    parquet_clean_dir: str = PARQUET_CLEAN_DIR,
    limit_rows: int | None = None,
    write_parquet: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    use_bar = show_progress and not verbose
    total_steps = 5 + (4 if write_parquet else 0) + (0 if verbose else 1)
    bar = _StageProgressBar(total_steps, enabled=use_bar)

    read_label = "Leer Parquet (icfes, internet, bachillerato)"
    if limit_rows is not None:
        read_label = f"Leer Parquet (limit {limit_rows} por tabla)"
    bar.advance(read_label)

    base_icfes = spark.read.parquet(os.path.join(parquet_dir, "icfes"))
    base_internet = spark.read.parquet(os.path.join(parquet_dir, "internet"))
    base_bach = spark.read.parquet(os.path.join(parquet_dir, "bachillerato"))

    if limit_rows is not None:
        base_icfes = base_icfes.limit(limit_rows)
        base_internet = base_internet.limit(limit_rows)
        base_bach = base_bach.limit(limit_rows)

    bar.advance("Transformar ICFES (casts, filtros, IQR, imputación)")
    df_icfes = prepare_icfes(base_icfes)
    bar.advance("Transformar internet (casts, IQR, imputación)")
    df_internet = prepare_internet(base_internet)
    bar.advance("Transformar bachillerato (casts, imputación)")
    df_bach = prepare_bachillerato(base_bach)
    bar.advance("Municipio: agregar internet + join cobertura (accesos/hab.)")
    df_municipio = build_municipio_internet_cobertura(df_internet, df_bach)

    outputs: tuple[tuple[str, DataFrame], ...] = (
        ("icfes", df_icfes),
        ("internet", df_internet),
        ("bachillerato", df_bach),
        ("municipio_internet_cobertura", df_municipio),
    )

    if write_parquet:
        for name, sdf in outputs:
            out_path = os.path.join(parquet_clean_dir, name)
            bar.advance(f"Escribir Parquet limpio: {name}")
            sdf.write.mode("overwrite").parquet(out_path)

    if verbose:
        summarize_print("icfes (limpio)", df_icfes)
        summarize_print("internet (limpio)", df_internet)
        summarize_print("bachillerato (limpio)", df_bach)
        summarize_print(
            "municipio: internet agregado + cobertura (accesos por habitante)",
            df_municipio,
        )
    else:
        bar.advance("Resumen (conteo por tabla)")
        counts: list[tuple[str, int]] = []
        for label, sdf in (
            ("icfes (limpio)", df_icfes),
            ("internet (limpio)", df_internet),
            ("bachillerato (limpio)", df_bach),
            (
                "municipio: internet + cobertura (accesos/hab.)",
                df_municipio,
            ),
        ):
            counts.append((label, sdf.count()))
        print("Listo.")
        for label, n in counts:
            print(f"  • {label}: {n:,} filas")

    return {
        "icfes": df_icfes,
        "internet": df_internet,
        "bachillerato": df_bach,
        "municipio_internet_cobertura": df_municipio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpieza y transformación inicial sobre data/parquet (PySpark)."
    )
    parser.add_argument(
        "--parquet-dir",
        default=PARQUET_DIR,
        help="Directorio base con subcarpetas icfes, internet, bachillerato",
    )
    parser.add_argument(
        "--output-dir",
        default=PARQUET_CLEAN_DIR,
        help="Directorio para Parquet limpio",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        metavar="N",
        help="Limitar filas por tabla tras leer Parquet (desarrollo rápido).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="No escribir data/parquet_clean (solo cálculo y salida por consola).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar schema, conteos y show(20) por tabla (más salida; sin barra).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="No mostrar la barra de progreso por etapas.",
    )
    args = parser.parse_args()

    logging.getLogger("py4j").setLevel(logging.ERROR)
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        run_pipeline(
            spark=spark,
            parquet_dir=args.parquet_dir,
            parquet_clean_dir=args.output_dir,
            limit_rows=args.limit_rows,
            write_parquet=not args.no_write,
            verbose=args.verbose,
            show_progress=not args.no_progress,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
