import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from kmeans_core import kmeans


def prepare_datasets(output_dir: str = "data") -> dict[str, tuple[np.ndarray, int]]:
    """
    Genera y guarda los datasets numéricos en formato CSV.
    Retorna un diccionario {nombre_dataset: (matriz_X, k_real)}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset sintético con 4 clusters bien separados (n=600, d=2)
    X_sep, _ = make_blobs(
        n_samples=600, centers=4, n_features=2, cluster_std=0.65, random_state=42
    )
    np.savetxt(os.path.join(output_dir, "synthetic_separated.csv"), X_sep, delimiter=",")

    # 2. Dataset sintético con 4 clusters parcialmente solapados (n=600, d=2)
    X_ovl, _ = make_blobs(
        n_samples=600, centers=4, n_features=2, cluster_std=1.85, random_state=42
    )
    np.savetxt(os.path.join(output_dir, "synthetic_overlapped.csv"), X_ovl, delimiter=",")

    # 3. Dataset real: Iris estandarizado (n=150, d=4, k_real=3)
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)
    np.savetxt(os.path.join(output_dir, "real_iris_scaled.csv"), X_iris, delimiter=",")

    return {
        "Sintetico_Separado": (X_sep, 4),
        "Sintetico_Solapado": (X_ovl, 4),
        "Real_Iris": (X_iris, 3),
    }


def run_k_variation_experiment(
    datasets: dict[str, tuple[np.ndarray, int]], k_range: range = range(2, 9)
) -> pd.DataFrame:
    """
    Evalúa inercia (WCSS) y coeficiente de silueta al variar k en [2..8].
    """
    records = []
    for ds_name, (X, _) in datasets.items():
        for k in k_range:
            res = kmeans(X, k=k, init="kmeans++", seed=42)
            sil = (
                silhouette_score(X, res["labels"])
                if len(np.unique(res["labels"])) > 1
                else -1.0
            )
            records.append(
                {
                    "dataset": ds_name,
                    "k": k,
                    "inertia": round(res["inertia"], 4),
                    "silhouette": round(float(sil), 4),
                    "iterations": res["iterations"],
                    "converged": res["converged"],
                }
            )
    return pd.DataFrame(records)


def run_initialization_stability_experiment(
    datasets: dict[str, tuple[np.ndarray, int]], n_runs: int = 30
) -> pd.DataFrame:
    """
    Compara 'random' vs 'kmeans++' a lo largo de n_runs semillas distintas
    para medir estabilidad (desviación estándar de WCSS) y velocidad de convergencia.
    """
    records = []
    for ds_name, (X, k_true) in datasets.items():
        for init_method in ["random", "kmeans++"]:
            for seed in range(n_runs):
                t0 = time.perf_counter()
                res = kmeans(X, k=k_true, init=init_method, seed=seed)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                sil = (
                    silhouette_score(X, res["labels"])
                    if len(np.unique(res["labels"])) > 1
                    else -1.0
                )
                records.append(
                    {
                        "dataset": ds_name,
                        "k": k_true,
                        "init": init_method,
                        "seed": seed,
                        "inertia": res["inertia"],
                        "silhouette": float(sil),
                        "iterations": res["iterations"],
                        "time_ms": elapsed_ms,
                    }
                )
    return pd.DataFrame(records)


def run_scalability_experiment() -> pd.DataFrame:
    """
    Mide el tiempo de ejecución al escalar el número de puntos n,
    validando empíricamente la complejidad O(t * k * n * d).
    """
    sizes = [500, 1000, 2500, 5000, 10000, 20000]
    k_fixed = 5
    d_fixed = 4
    records = []

    for n in sizes:
        X_scale, _ = make_blobs(
            n_samples=n, centers=k_fixed, n_features=d_fixed, random_state=42
        )
        for init_method in ["random", "kmeans++"]:
            # Promedio sobre 5 ejecuciones para reducir ruido del sistema operativo
            times = []
            iters = []
            for s in range(5):
                t0 = time.perf_counter()
                res = kmeans(X_scale, k=k_fixed, init=init_method, seed=s)
                times.append((time.perf_counter() - t0) * 1000.0)
                iters.append(res["iterations"])

            mean_time = float(np.mean(times))
            mean_iter = float(np.mean(iters))
            records.append(
                {
                    "n": n,
                    "d": d_fixed,
                    "k": k_fixed,
                    "init": init_method,
                    "mean_iterations": round(mean_iter, 2),
                    "mean_time_ms": round(mean_time, 3),
                    "time_per_iter_ms": round(mean_time / mean_iter, 4),
                }
            )
    return pd.DataFrame(records)


def main():
    os.makedirs("results", exist_ok=True)
    print("1. Preparando datasets en data/...")
    datasets = prepare_datasets("data")

    print("2. Ejecutando experimento de variación de k...")
    df_k = run_k_variation_experiment(datasets)
    df_k.to_csv("results/exp1_k_variation.csv", index=False)

    print("3. Ejecutando experimento de estabilidad por inicialización (30 semillas)...")
    df_init = run_initialization_stability_experiment(datasets, n_runs=30)
    df_init.to_csv("results/exp2_init_stability.csv", index=False)

    # Tabla resumen agregada lista para el informe LaTeX
    summary_init = (
        df_init.groupby(["dataset", "init"])
        .agg(
            wcss_mean=("inertia", "mean"),
            wcss_std=("inertia", "std"),
            wcss_min=("inertia", "min"),
            wcss_max=("inertia", "max"),
            sil_mean=("silhouette", "mean"),
            iter_mean=("iterations", "mean"),
            time_ms_mean=("time_ms", "mean"),
        )
        .round(4)
        .reset_index()
    )
    summary_init.to_csv("results/exp2_init_summary.csv", index=False)

    print("4. Ejecutando experimento de escalabilidad temporal...")
    df_scale = run_scalability_experiment()
    df_scale.to_csv("results/exp3_scalability.csv", index=False)

    print("\n--- RESUMEN ESTABILIDAD (exp2_init_summary.csv) ---")
    print(summary_init.to_string(index=False))
    print("\n--- RESUMEN ESCALABILIDAD (exp3_scalability.csv) ---")
    print(df_scale.to_string(index=False))


if __name__ == "__main__":
    main()