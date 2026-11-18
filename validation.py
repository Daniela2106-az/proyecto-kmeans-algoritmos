import numpy as np
from sklearn.cluster import KMeans
from kmeans_core import kmeans, _initialize_centroids

DATOS = [
    ("Sintético Separado", "data/synthetic_separated.csv", 4),
    ("Sintético Solapado", "data/synthetic_overlapped.csv", 4),
    ("Real Iris", "data/real_iris_scaled.csv", 3),
]
def validar():
    print(f"{'Dataset':20s} {'WCSS propio':>14s} {'WCSS sklearn':>14s}")
    for nombre, archivo, k in DATOS:
        X = np.loadtxt(archivo, delimiter=",")
        # mismos centroides iniciales para las dos implementaciones
        inicio = _initialize_centroids(X, k, "kmeans++", np.random.default_rng(0))
        propio = kmeans(X, k=k, init="kmeans++", seed=0)
        sk = KMeans(n_clusters=k, init=inicio, n_init=1).fit(X)
        print(f"{nombre:20s} {propio['inertia']:14.4f} {sk.inertia_:14.4f}")

if __name__ == "__main__":
    validar()