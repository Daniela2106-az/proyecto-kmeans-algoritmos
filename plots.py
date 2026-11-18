import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans_core import kmeans
os.makedirs("figures", exist_ok=True)
DATOS = [
    ("Sintetico_Separado", "data/synthetic_separated.csv", 4),
    ("Sintetico_Solapado", "data/synthetic_overlapped.csv", 4),
    ("Real_Iris", "data/real_iris_scaled.csv", 3),
]
def dispersion():
    plt.figure(figsize=(12, 4))
    for i, (nombre, archivo, k) in enumerate(DATOS):
        X = np.loadtxt(archivo, delimiter=",")
        r = kmeans(X, k=k, init="kmeans++", seed=42)
        plt.subplot(1, 3, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=r["labels"], s=10)
        plt.scatter(r["centroids"][:, 0], r["centroids"][:, 1], c="black", marker="X", s=120)
        plt.title(nombre)
    plt.tight_layout()
    plt.savefig("figures/dispersion.png", dpi=200)
    plt.close()
def trayectoria():
    X = np.loadtxt("data/synthetic_separated.csv", delimiter=",")
    r = kmeans(X, k=4, init="random", seed=2)
    historia = np.array(r["history"])  # (pasos, k, 2)
    plt.scatter(X[:, 0], X[:, 1], c=r["labels"], s=8, alpha=0.4)
    for j in range(4):
        plt.plot(historia[:, j, 0], historia[:, j, 1], "-o", color="black")
    plt.title("Trayectoria de los centroides (init=random)")
    plt.savefig("figures/trayectoria.png", dpi=200)
    plt.close()
def codo_y_silueta():
    df = pd.read_csv("results/exp1_k_variation.csv")
    for columna, nombre in [("inertia", "codo"), ("silhouette", "silueta")]:
        plt.figure(figsize=(6, 4))
        for ds in df["dataset"].unique():
            d = df[df["dataset"] == ds]
            plt.plot(d["k"], d[columna], "-o", label=ds)
        plt.xlabel("k")
        plt.ylabel(columna)
        plt.legend()
        plt.savefig(f"figures/{nombre}.png", dpi=200)
        plt.close()
def tiempo_vs_n():
    df = pd.read_csv("results/exp3_scalability.csv")
    for init in ["random", "kmeans++"]:
        d = df[df["init"] == init]
        plt.plot(d["n"], d["time_per_iter_ms"], "-o", label=init)
    plt.xlabel("n")
    plt.ylabel("ms por iteración")
    plt.legend()
    plt.savefig("figures/tiempo_vs_n.png", dpi=200)
    plt.close()
def generar_todo():
    dispersion()
    trayectoria()
    codo_y_silueta()
    tiempo_vs_n()
    print("Figuras guardadas en figures/")
if __name__ == "__main__":
    generar_todo()